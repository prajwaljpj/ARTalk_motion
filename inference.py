#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import json
import torch
import argparse
import torchaudio
import numpy as np
import gradio as gr
from gtts import gTTS
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
import mediapipe as mp
from app.utils_videos import write_video


class ARTAvatarInferEngine:
    def __init__(self, load_gaga=False, fix_pose=False, clip_length=750, device="cuda", resolution="1080p", projection_mode='perspective', zoom=1.0, expression_scale=1.0):
        self.device = device
        self.fix_pose = fix_pose
        self.clip_length = clip_length
        self.projection_mode = projection_mode
        self.zoom = zoom
        self.expression_scale = expression_scale
        
        self.resolution_map = {
            "1080p": (1920, 1080),
            "720p": (1280, 720),
        }
        self.output_size = self.resolution_map.get(resolution, (1920, 1080))

        audio_encoder = "wav2vec"
        ckpt = torch.load(
            "./assets/ARTalk_{}.pt".format(audio_encoder),
            map_location="cpu",
            weights_only=True,
        )
        configs = json.load(open("./assets/config.json"))
        configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
        self.ARTalk = BitwiseARModel(configs).eval().to(device)
        self.ARTalk.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(
            n_shape=300, n_exp=100, scale=1.0, no_lmks=True
        ).to(device)
        self.mesh_renderer = RenderMesh(
            image_size=512, faces=self.flame_model.get_faces(), scale=1.0, projection_mode=self.projection_mode
        )

        self.output_dir = "render_results/ARTAvatar_{}".format(audio_encoder)
        os.makedirs(self.output_dir, exist_ok=True)
        self.style_motion = None
        self.segmentation_model = None

        if load_gaga:
            from app.GAGAvatar import GAGAvatar

            self.GAGAvatar = GAGAvatar(projection_mode=self.projection_mode, zoom=self.zoom).to(device)
            self.GAGAvatar_flame = FLAMEModel(
                n_shape=300, n_exp=100, scale=5.0, no_lmks=True
            ).to(device)

    def _initialize_segmentation_model(self):
        if self.segmentation_model is None:
            print("Initializing MediaPipe Selfie Segmentation...")
            self.segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
            print("Done.")

    def set_style_motion(self, style_motion):
        if isinstance(style_motion, str):
            style_motion = torch.load(
                "assets/style_motion/{}.pt".format(style_motion),
                map_location="cpu",
                weights_only=True,
            )
        assert style_motion.shape == (
            50,
            106,
        ), f"Invalid style_motion shape: {style_motion.shape}."
        self.style_motion = style_motion[None].to(self.device)

    def inference(self, audio, clip_length=None):
        audio_batch = {
            "audio": audio[None].to(self.device),
            "style_motion": self.style_motion,
        }
        print("Inferring motion...")
        pred_motions = self.ARTalk.inference(audio_batch, with_gtmotion=False)[0]
        clip_length = clip_length if clip_length is not None else self.clip_length
        pred_motions = self.smooth_motion_savgol(pred_motions)[:clip_length]
        
        # Apply expression scale
        pred_motions[..., :100] *= self.expression_scale

        if self.fix_pose:
            pred_motions[..., 100:103] *= 0.0
        print("Done!")
        pred_motions[..., 104:] *= 0.0
        return pred_motions

    def rendering(
        self,
        audio,
        pred_motions,
        shape_id="mesh",
        shape_code=None,
        save_name="ARTAvatar.mp4",
        background_path=None,
        background_blur=0.5,
    ):
        print("Rendering...")
        pred_images = []
        
        background_image_full = None
        if background_path:
            try:
                from PIL import Image
                background_image_full = Image.open(background_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load background image due to '{e}'. Rendering without background.")
                background_path = None

        if shape_id == "mesh":
            # This part remains the same, using the high-quality internal alpha mask
            background_tensor = None
            if background_image_full:
                background_tensor = torch.from_numpy(np.array(background_image_full)).permute(2, 0, 1).float()

            if shape_code is None:
                shape_code = audio.new_zeros(1, 300).to(self.device).expand(pred_motions.shape[0], -1)
            else:
                shape_code = shape_code.to(self.device).expand(pred_motions.shape[0], -1)
            
            verts = self.ARTalk.basic_vae.get_flame_verts(
                self.flame_model, shape_code, pred_motions, with_global=True
            )
            for v in tqdm(verts):
                rgb, alpha = self.mesh_renderer(v[None])
                rgb = rgb[0].cpu()
                alpha = alpha[0].cpu()

                if background_tensor is not None:
                    bg_c, bg_h, bg_w = background_tensor.shape
                    head_c, head_h, head_w = rgb.shape
                    y_offset = bg_h - head_h
                    x_offset = (bg_w - head_w) // 2
                    if y_offset < 0 or x_offset < 0:
                        final_frame = rgb
                        background_tensor = None
                    else:
                        canvas = background_tensor.clone()
                        roi = canvas[:, y_offset : y_offset + head_h, x_offset : x_offset + head_w]
                        composited_roi = rgb * alpha + roi * (1 - alpha)
                        canvas[:, y_offset : y_offset + head_h, x_offset : x_offset + head_w] = composited_roi
                        final_frame = canvas
                else:
                    final_frame = rgb
                pred_images.append(final_frame)
        else:
            # New GAGAvatar logic with MediaPipe segmentation
            self._initialize_segmentation_model()
            self.GAGAvatar.set_avatar_id(shape_id)
            
            bg_image_resized_np = None
            if background_image_full:
                # Resize and blur the background just once
                bg_image_resized = background_image_full.resize(self.output_size)
                if background_blur > 0:
                    import cv2
                    kernel_size = int(background_blur * 20) * 2 + 1
                    bg_image_resized_np = cv2.GaussianBlur(np.array(bg_image_resized), (kernel_size, kernel_size), 0)
                else:
                    bg_image_resized_np = np.array(bg_image_resized)

            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                num_frames = len(pred_motions)
                rendering_task = progress.add_task("[cyan]Rendering Avatar", total=num_frames)
                segmentation_task = progress.add_task("[magenta]Segmentation", total=num_frames)
                compositing_task = progress.add_task("[yellow]Compositing", total=num_frames)

                for motion in pred_motions:
                    # --- Stage 1: Render avatar ---
                    batch = self.GAGAvatar.build_forward_batch(motion[None], self.GAGAvatar_flame)
                    avatar_tensor = self.GAGAvatar.forward_expression(batch)
                    avatar_np = (avatar_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    progress.update(rendering_task, advance=1)

                    # --- Stage 2: Segmentation and Compositing ---
                    if background_image_full:
                        results = self.segmentation_model.process(avatar_np)
                        mask = np.stack((results.segmentation_mask,) * 3, axis=-1)
                        progress.update(segmentation_task, advance=1)

                        # Create a 1080p canvas
                        canvas_np = bg_image_resized_np.copy()
                        avatar_h, avatar_w, _ = avatar_np.shape
                        canvas_h, canvas_w, _ = canvas_np.shape
                        y_offset = canvas_h - avatar_h
                        x_offset = (canvas_w - avatar_w) // 2
                        roi = canvas_np[y_offset:y_offset + avatar_h, x_offset:x_offset + avatar_w]
                        
                        avatar_fg = avatar_np.astype(np.float32) * mask
                        background_bg = roi.astype(np.float32) * (1 - mask)
                        composited_float = avatar_fg + background_bg
                        composited_roi_np = np.clip(composited_float, 0, 255).astype(np.uint8)
                        
                        canvas_np[y_offset:y_offset + avatar_h, x_offset:x_offset + avatar_w] = composited_roi_np
                        final_image_tensor = torch.from_numpy(canvas_np).permute(2, 0, 1)
                        progress.update(compositing_task, advance=1)
                    else:
                        # If no background, skip segmentation and just place on canvas
                        progress.update(segmentation_task, advance=1, visible=False) # Hide segmentation bar
                        avatar_h, avatar_w, _ = avatar_np.shape
                        canvas_h, canvas_w = self.output_size[1], self.output_size[0]
                        canvas_np = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        y_offset = canvas_h - avatar_h
                        x_offset = (canvas_w - avatar_w) // 2
                        canvas_np[y_offset:y_offset + avatar_h, x_offset:x_offset + avatar_w] = avatar_np
                        final_image_tensor = torch.from_numpy(canvas_np).permute(2, 0, 1)
                        progress.update(compositing_task, advance=1)
                    
                    pred_images.append(final_image_tensor)
        
        print("Done!")
        # save video
        print("Saving video...")
        pred_images = torch.stack(pred_images)
        dump_path = os.path.join(self.output_dir, "{}.mp4".format(save_name))
        audio = audio[: int(pred_images.shape[0] / 25.0 * 16000)]
        write_video(
            pred_images, dump_path, 25.0, audio_samples=audio, sample_rate=16000, acodec="aac"
        )
        print("Done!")

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        from scipy.signal import savgol_filter

        motion_np = motion_codes.clone().detach().cpu().numpy()
        motion_np_smoothed = savgol_filter(
            motion_np, window_length=5, polyorder=2, axis=0
        )
        motion_np_smoothed[..., 100:103] = savgol_filter(
            motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
        )
        return torch.tensor(motion_np_smoothed).type_as(motion_codes)


def run_gradio_app(engine):
    def process_audio(
        input_type, audio_input, text_input, text_language, shape_id, style_id, resolution, projection, background_path, background_blur, zoom, expression_scale
    ):
        if input_type == "Audio" and audio_input is None:
            gr.Warning("Please upload an audio file")
            return None, None
        if input_type == "Text" and (
            text_input is None or len(text_input.strip()) == 0
        ):
            gr.Warning("Please input text content")
            return None, None
        if input_type == "Text":
            gtts_lang = {
                "English": "en",
                "中文": "zh",
                "日本語": "ja",
                "Deutsch": "de",
                "Français": "fr",
                "Español": "es",
            }
            tts = gTTS(text=text_input, lang=gtts_lang[text_language])
            tts.save("./render_results/tts_output.wav")
            audio_input = "./render_results/tts_output.wav"
        # load audio
        audio, sr = torchaudio.load(audio_input)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        # inference
        if style_id == "default":
            engine.style_motion = None
        else:
            engine.set_style_motion(style_id)
        
        # Update engine resolution and projection based on UI
        engine.output_size = engine.resolution_map.get(resolution, (1920, 1080))
        engine.mesh_renderer.projection_mode = projection
        engine.expression_scale = expression_scale
        if hasattr(engine, 'GAGAvatar'):
            engine.GAGAvatar.projection_mode = projection
            engine.GAGAvatar.cam_params['focal_x'] = 12.0 * zoom
            engine.GAGAvatar.cam_params['focal_y'] = 12.0 * zoom

        pred_motions = engine.inference(audio)
        # render
        save_name = f'{audio_input.split("/")[-1].split(".")[0]}_{style_id.replace(".", "_")}_{shape_id.replace(".", "_")}'
        engine.rendering(
            audio, pred_motions, shape_id=shape_id, save_name=save_name, 
            background_path=background_path, background_blur=background_blur
        )
        # save pred_motions
        torch.save(
            pred_motions.float().cpu(),
            os.path.join(engine.output_dir, "{}_motions.pt".format(save_name)),
        )
        return os.path.join(
            engine.output_dir, "{}.mp4".format(save_name)
        ), os.path.join(engine.output_dir, "{}_motions.pt".format(save_name))

    # create the gradio app
    if hasattr(engine, "GAGAvatar"):
        all_gagavatar_id = list(engine.GAGAvatar.all_gagavatar_id.keys())
        all_gagavatar_id = sorted(all_gagavatar_id)
    else:
        all_gagavatar_id = []
    all_style_id = [os.path.basename(i) for i in os.listdir("assets/style_motion")]
    all_style_id = sorted([i.split(".")[0] for i in all_style_id if i.endswith(".pt")])
    with gr.Blocks(
        title="ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model"
    ) as demo:
        gr.Markdown(
            """
            <center>
            <h1>ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model</h1>
            </center>

            **ARTalk generates realistic 3D head motions from given audio, including accurate lip sync, natural facial animations, eye blinks, and head poses.**
            Please refer to our [paper](https://arxiv.org/abs/2502.20323), [project page](https://xg-chu.site/project_artalk), and [github](https://github.com/xg-chu/ARTalk) for more details about ARTalk.
            The apperance is powered by [GAGAvatar](https://xg-chu.site/project_gagavatar).
            
            Usage: Upload an audio file or input text -> Select an appearance and style -> Click generate!
        """
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Audio & Text")
                input_type = gr.Radio(
                    choices=["Audio", "Text"], value="Audio", label="Choose input type"
                )
                audio_group = gr.Group()
                with audio_group:
                    audio_input = gr.Audio(type="filepath", label="Input Audio")
                text_group = gr.Group(visible=False)
                with text_group:
                    text_input = gr.Textbox(label="Input Text")
                    text_language = gr.Dropdown(
                        choices=[
                            "English",
                            "中文",
                            "日本語",
                            "Deutsch",
                            "Français",
                            "Español",
                        ],
                        value="English",
                        label="Choose the language of the input text",
                    )
            with gr.Column():
                gr.Markdown("### Avatar Control")
                appearance = gr.Dropdown(
                    choices=["mesh"] + all_gagavatar_id,
                    value="mesh",
                    label="Choose the apperance of the speaker",
                )
                style = gr.Dropdown(
                    choices=["default"] + all_style_id,
                    value="natural_0",
                    label="Choose the style of the speaker",
                )
                resolution = gr.Dropdown(
                    choices=["1080p", "720p"],
                    value="1080p",
                    label="Choose the output resolution",
                )
                projection = gr.Dropdown(
                    choices=["perspective", "orthographic"],
                    value="perspective",
                    label="Choose the projection mode",
                )
                zoom = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Zoom")
                expression_scale = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Expression Scale")
                background_input = gr.Image(type="filepath", label="Background Image (optional)")
                background_blur = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.1, label="Background Blur (GAGAvatar only)")
            with gr.Column():
                gr.Markdown("### Generated Video")
                video_output = gr.Video(autoplay=True)
                motion_output = gr.File(label="motion sequence", file_types=[".pt"])

        inputs = [input_type, audio_input, text_input, text_language, appearance, style, resolution, projection, background_input, background_blur, zoom, expression_scale]
        btn = gr.Button("Generate")
        btn.click(
            fn=process_audio, inputs=inputs, outputs=[video_output, motion_output]
        )

        if hasattr(engine, "GAGAvatar"):
            examples = [
                ["Audio", "demo/jp1.wav", None, None, "12.jpg", "curious_0", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/jp2.wav", None, None, "12.jpg", "natural_3", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/eng1.wav", None, None, "12.jpg", "natural_2", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/eng2.wav", None, None, "12.jpg", "happy_1", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/cn1.wav", None, None, "11.jpg", "natural_1", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/cn2.wav", None, None, "12.jpg", "happy_2", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                [
                    "Text",
                    None,
                    "Hello, this is a demo of ARTalk! Let's create something fun together.",
                    "English",
                    "12.jpg",
                    "happy_0",
                    "1080p",
                    "perspective",
                    None,
                    0.5,
                    1.0,
                    1.0,
                ],
                ["Text", None, "让我们一起创造一些有趣的东西吧。", "中文", "12.jpg", "natural_0", "1080p", "perspective", None, 0.5, 1.0, 1.0],
            ]
        else:
            examples = [
                ["Audio", "demo/jp1.wav", None, None, "mesh", "curious_0", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/jp2.wav", None, None, "mesh", "natural_3", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/eng1.wav", None, None, "mesh", "natural_2", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/eng2.wav", None, None, "mesh", "happy_1", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/cn1.wav", None, None, "mesh", "natural_1", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                ["Audio", "demo/cn2.wav", None, None, "mesh", "happy_2", "1080p", "perspective", None, 0.5, 1.0, 1.0],
                [
                    "Text",
                    None,
                    "Hello, this is a demo of ARTalk! Let's create something fun together.",
                    "English",
                    "mesh",
                    "happy_0",
                    "1080p",
                    "perspective",
                    None,
                    0.5,
                    1.0,
                    1.0,
                ],
                ["Text", None, "让我们一起创造一些有趣的东西吧。", "中文", "mesh", "natural_0", "1080p", "perspective", None, 0.5, 1.0, 1.0],
            ]
        gr.Examples(examples=examples, inputs=inputs, outputs=video_output)

        def toggle_input(choice):
            if choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        input_type.change(
            fn=toggle_input, inputs=[input_type], outputs=[audio_group, text_group]
        )

    demo.launch(server_name="0.0.0.0", server_port=8960)



if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", "-a", default=None, type=str)
    parser.add_argument("--clip_length", "-l", default=750, type=int)
    parser.add_argument("--shape_id", "-i", default="mesh", type=str)
    parser.add_argument("--style_id", "-s", default="default", type=str)
    parser.add_argument("--background_path", "-bg", default=None, type=str)
    parser.add_argument("--background_blur", "-bb", default=0.5, type=float)
    parser.add_argument("--resolution", "-res", default="1080p", type=str, choices=["1080p", "720p"])
    parser.add_argument("--projection", "-proj", default="perspective", type=str, choices=["perspective", "orthographic"])
    parser.add_argument("--zoom", "-z", default=1.0, type=float)
    parser.add_argument("--expression_scale", "-es", default=1.0, type=float)

    parser.add_argument("--run_app", action="store_true")
    args = parser.parse_args()

    engine = ARTAvatarInferEngine(
        load_gaga=True, fix_pose=False, clip_length=args.clip_length, resolution=args.resolution, projection_mode=args.projection, zoom=args.zoom, expression_scale=args.expression_scale
    )
    if args.run_app:
        run_gradio_app(engine)
    else:
        shape_id = (
            "mesh"
            if args.shape_id not in engine.GAGAvatar.all_gagavatar_id.keys()
            else args.shape_id
        )
        audio, sr = torchaudio.load(args.audio_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

        base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        save_name = f'{base_name}_{args.style_id.replace(".", "_")}_{args.shape_id.replace(".", "_")}'
        engine.set_style_motion(args.style_id)
        pred_motions = engine.inference(audio)
        engine.rendering(
            audio, pred_motions, shape_id=args.shape_id, save_name=save_name, 
            background_path=args.background_path, background_blur=args.background_blur
        )
