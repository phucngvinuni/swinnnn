import torch
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import re

# --- 1. THÊM ĐƯỜNG DẪN VÀ IMPORT CÁC MODULE SWINJSCC ---
SWINJSCC_CODE_PATH = "." 
sys.path.insert(0, os.path.abspath(SWINJSCC_CODE_PATH))

try:
    from net.network import SwinJSCC
    from utils import seed_torch
    import torch.nn as nn
except ImportError as e:
    print(f"Lỗi import từ SwinJSCC: {e}"); sys.exit(1)

# --- HÀM TÁI TẠO ---
def reconstruct_single_image_swinjscc(args):
    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    class Config: pass
    config = Config()
    config.device = device
    config.pass_channel = True
    config.logger = None
    config.downsample = 4 
    config.image_dims = (3, args.input_size, args.input_size)
    config.norm = False
    config.CUDA = True
    try:
        channel_number_int = int(args.C)
    except ValueError:
        print(f"ERROR: --C value '{args.C}' cannot be converted to an integer."); return

    encoder_kwargs = dict(
        img_size=(args.input_size, args.input_size), patch_size=2, in_chans=3,
        embed_dims=args.embed_dims, depths=args.depths, num_heads=args.num_heads, C=channel_number_int,
        window_size=args.window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    decoder_kwargs = dict(
        img_size=(args.input_size, args.input_size),
        embed_dims=args.embed_dims[::-1], depths=args.depths[::-1], num_heads=args.num_heads[::-1], C=channel_number_int,
        window_size=args.window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    config.encoder_kwargs = encoder_kwargs
    config.decoder_kwargs = decoder_kwargs
    
    print(f"Initializing model '{args.model}' with C={args.C}...")
    net = SwinJSCC(args, config)
    
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found at {args.checkpoint_path}"); return

    print(f"Loading weights from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    if next(iter(state_dict)).startswith('module.'):
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    
    net.to(device)
    net.eval()
    print("Model loaded successfully and set to evaluation mode.")

    if not os.path.exists(args.input_image_path):
        print(f"ERROR: Input image not found at {args.input_image_path}"); return

    try:
        with Image.open(args.input_image_path) as img_pil:
            original_size = img_pil.size
            new_H = original_size[1] - original_size[1] % 128
            new_W = original_size[0] - original_size[0] % 128
            if new_H == 0 or new_W == 0:
                print(f"ERROR: Image size {original_size} is too small to be cropped to a multiple of 128."); return

            transform = transforms.Compose([transforms.CenterCrop((new_H, new_W)), transforms.ToTensor()])
            img_tensor_input = transform(img_pil.convert('RGB')).unsqueeze(0).to(device)
            print(f"Input image loaded and processed to size {img_tensor_input.shape}")
    except Exception as e:
        print(f"Error loading or transforming input image: {e}"); return

    with torch.no_grad():
        recon_image, _, _, _, _ = net(img_tensor_input, given_SNR=args.snr_eval, given_rate=channel_number_int)
        recon_image = recon_image.clamp(0., 1.)

    # SỬA LỖI Ở ĐÂY:
    output_path_str = args.output_image_path
    if not output_path_str:
        input_p = Path(args.input_image_path)
        output_dir = Path(args.output_dir_overall)
        # Dòng quan trọng: tạo thư mục output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path_str = str(output_dir / f"{input_p.stem}_swinjscc_recon_C{args.C}_snr{args.snr_eval:.0f}dB.png")
    else:
        # Nếu người dùng cung cấp đường dẫn đầy đủ, cũng đảm bảo thư mục cha tồn tại
        Path(output_path_str).parent.mkdir(parents=True, exist_ok=True)

    try:
        to_pil = transforms.ToPILImage()
        recon_pil = to_pil(recon_image.squeeze(0).cpu())
        recon_pil.save(output_path_str)
        print(f"Reconstructed image saved to: {output_path_str}")
    except Exception as e:
        print(f"Error saving reconstructed image: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Image Inference with SwinJSCC")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--input_image_path', type=str, required=True)
    parser.add_argument('--output_image_path', type=str, default="")
    parser.add_argument('--output_dir_overall', default="./swinjscc_inference_output")
    parser.add_argument('--model', type=str, default='SwinJSCC_w/o_SAandRA')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--C', type=str, required=True)
    parser.add_argument('--embed_dims', type=int, nargs='+', default=[128, 192, 256, 320])
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2])
    parser.add_argument('--num_heads', type=int, nargs='+', default=[4, 6, 8, 10])
    parser.add_argument('--distortion-metric', type=str, default='MSE')
    parser.add_argument('--channel_type', type=str, default='rayleigh')
    parser.add_argument('--multiple-snr', type=str, default='10')
    parser.add_argument('--snr_eval', type=float, default=15.0)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    
    args = parser.parse_args()
    reconstruct_single_image_swinjscc(args)