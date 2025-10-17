
import os
import torch
import argparse
import torchaudio
import json
from app import BitwiseARModel

# This script is designed to be run in the ARTalk environment.
# It generates motion parameters from an audio file and saves them.

class MotionGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load the ARTalk model configuration
        audio_encoder = "wav2vec"
        ckpt_path = "./assets/ARTalk_{}.pt".format(audio_encoder)
        config_path = "./assets/config.json"
        
        print("Loading ARTalk model from {}...".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        configs = json.load(open(config_path))
        configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
        
        # Initialize the ARTalk model
        self.artalk = BitwiseARModel(configs).eval().to(device)
        self.artalk.load_state_dict(ckpt, strict=True)
        print("ARTalk model loaded successfully.")
        
        self.style_motion = None

    def set_style_motion(self, style_id):
        """Loads a style motion template."""
        if style_id is None or style_id.lower() == 'default':
            print("Using default style motion.")
            self.style_motion = None
            return
            
        style_path = "assets/style_motion/{}.pt".format(style_id)
        if not os.path.exists(style_path):
            print(f"Warning: Style motion file not found at {style_path}. Using default style.")
            self.style_motion = None
            return
            
        print(f"Loading style motion from {style_path}...")
        style_motion_data = torch.load(style_path, map_location="cpu", weights_only=True)
        self.style_motion = style_motion_data[None].to(self.device)

    def generate_motion(self, audio_path):
        """Generates and returns motion parameters and the audio tensor."""
        print(f"Processing audio file: {audio_path}")
        audio, sr = torchaudio.load(audio_path)
        # Resample and convert to mono
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        
        audio_batch = {
            "audio": audio[None].to(self.device),
            "style_motion": self.style_motion,
        }
        
        print("Inferring motion with ARTalk...")
        with torch.no_grad():
            pred_motions = self.artalk.inference(audio_batch, with_gtmotion=False)[0]
        print("Motion inference complete.")
        
        # Smooth the motion for more natural results
        pred_motions = self.smooth_motion_savgol(pred_motions)
        print("Motion smoothing complete.")
        
        return pred_motions, audio

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        """Applies a Savitzky-Golay filter to smooth the motion."""
        from scipy.signal import savgol_filter
        
        motion_np = motion_codes.clone().detach().cpu().numpy()
        
        # Smooth expression and pose separately with different window sizes
        motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
        motion_np_smoothed[..., 100:103] = savgol_filter(
            motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
        )
        
        return torch.tensor(motion_np_smoothed, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Generate motion parameters from audio using ARTalk.")
    parser.add_argument("--audio_path", "-a", required=True, type=str, help="Path to the input audio file.")
    parser.add_argument("--style_id", "-s", default="default", type=str, help="Style ID to use (e.g., 'natural_0'). Corresponds to files in assets/style_motion.")
    parser.add_argument("--output_path", "-o", default="output_motion.pt", type=str, help="Path to save the generated motion tensor file.")
    parser.add_argument("--device", "-d", default="cuda", type=str, help="Device to run the model on ('cuda' or 'cpu').")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return

    generator = MotionGenerator(device=args.device)
    generator.set_style_motion(args.style_id)
    
    pred_motions, _ = generator.generate_motion(args.audio_path)
    
    # Save the motion data
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save(pred_motions.cpu(), args.output_path)
    print(f"Successfully saved motion data to {args.output_path}")
    print(f"Motion tensor shape: {pred_motions.shape}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
