#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import av
import torch
import numpy as np

def write_video(video_frames, output_path, fps, background_path=None, audio_samples=None, sample_rate=None, acodec="aac"):
    assert video_frames.ndim == 4, "Input frames should be a 4D array."
    if background_path:
        assert video_frames.shape[1] in [3, 4], "Input frames should have 3 (RGB) or 4 (RGBA) channels when a background is provided."
    else:
        assert video_frames.shape[1] == 3, "Input frames should have 3 channels (RGB)."

    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()

    # Get video dimensions
    _, num_channels, video_height, video_width = video_frames.shape

    # Set output dimensions
    output_width, output_height = video_width, video_height

    if background_path:
        try:
            from PIL import Image
            background_img = Image.open(background_path).convert("RGB")
            output_width, output_height = background_img.size
            background_np = np.array(background_img).astype(np.float32).transpose(2, 0, 1)
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"Warning: Could not load background image due to '{e}'. Writing video without background.")
            background_path = None # Disable background processing

    # Initialize video container
    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = output_width
    stream.height = output_height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    # Initialize audio stream if needed
    if audio_samples is not None:
        if acodec == "aac":
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.format = "fltp"
        elif acodec == "vs_preview" or acodec == "debug":
            audio_stream = container.add_stream("mp3", rate=sample_rate)
            audio_stream.format = "fltp"
        else:
            raise ValueError("Unsupported audio codec.")

    # Frame-by-frame processing
    for i in range(video_frames.shape[0]):
        if background_path:
            # Create a fresh canvas for each frame
            canvas = background_np.copy()
            
            # Calculate position (bottom center)
            y_offset = output_height - video_height
            x_offset = (output_width - video_width) // 2

            if y_offset < 0 or x_offset < 0:
                print("Warning: Video frames are larger than the background image. Skipping background.")
                frame_to_write = video_frames[i]
            else:
                video_frame_np = video_frames[i].astype(np.float32)
                roi = canvas[:, y_offset:y_offset + video_height, x_offset:x_offset + video_width]

                if num_channels == 4:  # RGBA
                    alpha = video_frame_np[3:4, :, :] / 255.0
                    rgb_frame = video_frame_np[:3, :, :]
                    composited_roi = rgb_frame * alpha + roi * (1 - alpha)
                else:  # RGB with black background
                    mask = (np.sum(video_frame_np, axis=0, keepdims=True) > 30).astype(np.float32)
                    composited_roi = video_frame_np * mask + roi * (1 - mask)
                
                canvas[:, y_offset:y_offset + video_height, x_offset:x_offset + video_width] = composited_roi
                frame_to_write = canvas
        else:
            frame_to_write = video_frames[i]

        # Prepare and write the frame
        final_frame = np.clip(frame_to_write, 0, 255).astype(np.uint8)
        final_frame = final_frame.transpose(1, 2, 0)
        video_frame_av = av.VideoFrame.from_ndarray(final_frame, format="rgb24")
        for packet in stream.encode(video_frame_av):
            container.mux(packet)

    # Encode and write audio
    if audio_samples is not None:
        if isinstance(audio_samples, torch.Tensor):
            audio_samples = audio_samples.cpu().numpy()
        assert audio_samples.ndim == 1, "Input audio samples should be a 1D array."
        num_samples_per_frame = int(sample_rate // fps)
        for i in range(0, audio_samples.shape[0], num_samples_per_frame):
            chunk = audio_samples[i:i + num_samples_per_frame]
            if chunk.shape[0] < num_samples_per_frame:
                chunk = np.pad(chunk, (0, num_samples_per_frame - chunk.shape[0]), mode="constant")
            audio_frame = av.AudioFrame.from_ndarray(chunk[None], format="fltp", layout="mono")
            audio_frame.rate = sample_rate
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

    # Flush streams and close container
    for packet in stream.encode():
        container.mux(packet)
    if audio_samples is not None:
        for packet in audio_stream.encode():
            container.mux(packet)
    
    container.close()


def read_video_frames(video_path):
    container = av.open(video_path)
    for frame in container.decode(video=0):
        yield torch.tensor(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)


def get_video_info(video_path):
    info_dict = {}
    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    if video_stream is None:
        info_dict["video"] = None
    else:
        info_dict["video"] = {
            "width": video_stream.width,
            "height": video_stream.height,
            "frame_rate": float(video_stream.average_rate),
            "num_frames": video_stream.frames,
        }
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        info_dict["audio"] = None
    else:
        info_dict["audio"] = {
            "channels": audio_stream.channels,
            "sample_rate": audio_stream.rate,
            "duration": audio_stream.duration,
        }
    return info_dict


def read_all_video_frames(video_path):
    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    if video_stream is None:
        print("No video stream found in the file.")
        return np.zeros((0), dtype=np.uint8), 0
    frames = []
    for frame in container.decode(video=0):  # Decode only video stream
        if frame.pts is None:  # Ignore invalid frames
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        # frame_id = int(frame.pts * video_stream.time_base * float(video_stream.average_rate))
    frames = torch.tensor(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
    return frames, float(video_stream.average_rate)


def read_audio_samples(video_path, stero=False):
    container = av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        print("No audio stream found in the file.")
        return None, None
    audio_samples = []
    for frame in container.decode(audio=0):  # Decode all audio frames
        audio_samples.append(frame.to_ndarray())  # Convert to NumPy array
    # Concatenate all audio frames into a single array
    audio_data = np.concatenate(audio_samples, axis=-1)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0  # for PCM (WAV)
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / (2**31) # for FLAC
    if not stero:
        audio_data = audio_data.mean(axis=0)
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        print("Warning: Audio samples are not normalized, max={}, min={}.".format(audio_data.max(), audio_data.min()))
    return audio_data, audio_stream.rate


if __name__ == "__main__":
    from tqdm import tqdm
    # Example Usage
    video_path = '../MultiTalk_dataset/multitalk_dataset/english/-OknSRRyFJE_0.mp4'
    vres, fps = read_all_video_frames(video_path)
    print(vres.shape, fps)
    
    ares, sample_rate = read_audio_samples(video_path)
    print(ares.shape, sample_rate)
    
    video_length = get_video_info(video_path)['video']['num_frames']
    print(get_video_info(video_path))

    for frame in tqdm(read_video_frames(video_path), total=video_length):
        pass

    # write_video(vres, "output_debug.mp4", fps)
