import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import yaml
import io
import shutil
import sys
from torchvision import transforms
from simplejpeg import encode_jpeg, decode_jpeg

# --- 1. THÊM ĐƯỜNG DẪN VÀ IMPORT CÁC MODULE ---
AQUASC_CODE_PATH = "./aquasc_code"
sys.path.insert(0, os.path.abspath(AQUASC_CODE_PATH))

try:
    from datasets import build_dataset, yolo_collate_fn
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    from utils import AverageMeter, seed_torch # Import AverageMeter để tính trung bình
    TORCHMETRICS_AVAILABLE = True
except ImportError as e:
    print(f"Lỗi import: {e}"); sys.exit(1)

# --- 2. HÀM TIỆN ÍCH ---

def seed_initial(seed=0):
    # ... (giữ nguyên)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

def pass_channel_custom(image_tensor: torch.Tensor, snr_db: float, channel_type: str):
    # ... (giữ nguyên)
    signal = image_tensor * 2 - 1
    signal_power = torch.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    noise_std = torch.sqrt(noise_variance)

    if channel_type == 'awgn':
        noise = torch.normal(0, noise_std, size=signal.shape, device=signal.device)
        noisy_signal = signal + noise
    elif channel_type == 'rayleigh':
        B, C, H, W = signal.shape
        h_real = torch.normal(0, np.sqrt(0.5), size=(B, 1, 1, 1), device=signal.device)
        h_imag = torch.normal(0, np.sqrt(0.5), size=(B, 1, 1, 1), device=signal.device)
        h_magnitude = torch.sqrt(h_real**2 + h_imag**2)
        
        faded_signal = h_magnitude * signal
        noise = torch.normal(0, noise_std, size=signal.shape, device=signal.device)
        noisy_faded_signal = faded_signal + noise
        
        equalized_signal = noisy_faded_signal / (h_magnitude + 1e-9)
        noisy_signal = equalized_signal
    else:
        raise ValueError("Channel type not supported.")

    output_tensor = (noisy_signal + 1) / 2
    return output_tensor.clamp(0., 1.)

# THÊM HÀM TÍNH PSNR
def calculate_psnr(original_batch: torch.Tensor, recon_batch: torch.Tensor) -> float:
    """
    Tính toán PSNR trung bình cho một batch ảnh.
    Input tensors should be in range [0, 1].
    """
    original_batch = original_batch.to(recon_batch.device)
    mse = torch.mean((original_batch * 255. - recon_batch * 255.)**2)
    if mse.item() < 1e-10:
        return 100.0
    psnr = 10 * torch.log10(255.**2 / mse)
    return psnr.item()

# --- 3. SCRIPT BENCHMARK CHÍNH ---

def main(args):
    seed_initial(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ... (phần Load YOLO và Dataloader giữ nguyên) ...
    seed_torch(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Load YOLO model một lần
    if not os.path.exists(args.yolo_model_path):
        print(f"ERROR: YOLO weights not found at {args.yolo_model_path}"); return
    yolo_model = YOLO(args.yolo_model_path).to(device)
    print("YOLO model loaded.")

    # Chuẩn bị Dataloader một lần
    args_for_dataset = argparse.Namespace(
        data_path=args.dataset_path, data_set='fish',
        input_size=args.input_size, patch_size=8,
        mask_ratio=0.0, num_object_classes=1, eval=True
    )
    val_dataset = build_dataset(is_train=False, args=args_for_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=yolo_collate_fn)
    print(f"Validation dataloader created with {len(val_loader)} batches.")
    
    # Định nghĩa các cấu hình JPEG để benchmark
    jpeg_qualities_to_test = [95]
    
    # ... (phần Chuẩn bị thư mục output giữ nguyên) ...

    all_results = {}
    to_tensor_transform = transforms.ToTensor()
    to_pil_transform = transforms.ToPILImage()

    for quality in jpeg_qualities_to_test:
        config_name = f"JPEG_Q{quality}"
        print(f"\n\n===== Starting Benchmark for Config: {config_name} =====")
        all_results[config_name] = {}
        
        for snr in args.snr_db_range:
            print(f"\n--- Processing SNR = {snr} dB for config '{config_name}' ---")
            
            # Khởi tạo lại các meters cho mỗi lần chạy
            if TORCHMETRICS_AVAILABLE:
                map_calculator = MeanAveragePrecision(iou_type="bbox").to(device)
            else: map_calculator = None
            
            psnr_meter = AverageMeter() # Dùng AverageMeter để tính PSNR trung bình

            pbar = tqdm(val_loader, desc=f"JPEG-Q{quality} SNR {snr}dB")
            for batch in pbar:
                (original_imgs_tensor, _), (target_imgs, yolo_gt_targets_list, _) = batch
                
                # Chuyển batch gốc lên device để tính PSNR sau này
                original_imgs_tensor = original_imgs_tensor.to(device)

                reconstructed_batch_list = []
                for i in range(original_imgs_tensor.size(0)):
                    pil_img = np.asarray(to_pil_transform(original_imgs_tensor[i].cpu()))
                    buffer = encode_jpeg(pil_img, quality=quality, colorspace='RGB')
                    jpeg_pil_img = decode_jpeg(buffer, colorspace='RGB')
                    
                    jpeg_tensor = to_tensor_transform(jpeg_pil_img).to(device)
                    reconstructed_batch_list.append(jpeg_tensor)
                
                jpeg_batch_tensor = torch.stack(reconstructed_batch_list)
                noisy_batch_tensor = pass_channel_custom(jpeg_batch_tensor, snr, args.channel_type)
                
                # --- TÍNH TOÁN CÁC METRICS ---
                # 1. Tính PSNR
                current_psnr = calculate_psnr(original_imgs_tensor, noisy_batch_tensor)
                psnr_meter.update(current_psnr, original_imgs_tensor.size(0))
                
                # 2. Đánh giá YOLO
                if yolo_model and map_calculator:
                    # ... (phần YOLO giữ nguyên) ...
                    with torch.no_grad():
                        yolo_preds = yolo_model(noisy_batch_tensor, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
                    
                    preds_for_metric = [{"boxes": r.boxes.xyxy, "scores": r.boxes.conf, "labels": r.boxes.cls.long()} for r in yolo_preds]
                    gts_for_metric = [{"boxes": t['boxes'].to(device), "labels": t['labels'].to(device)} for t in yolo_gt_targets_list]
                    
                    map_calculator.update(preds_for_metric, gts_for_metric)
                    
                # Tính toán và lưu kết quả
                final_metrics = {"psnr": psnr_meter.avg}
                if map_calculator:
                    map_results = map_calculator.compute()
                    final_metrics["map50"] = map_results['map_50'].item()
                    final_metrics["map"] = map_results['map'].item()
                    print(f"Results for {config_name}, SNR {snr}dB: PSNR={final_metrics['psnr']:.2f}, mAP@50={final_metrics['map50']:.4f}, mAP@[.5:.95]={final_metrics['map']:.4f}")
                else:
                    print(f"Results for {config_name}, SNR {snr}dB: PSNR={final_metrics['psnr']:.2f}")    

            all_results[config_name][snr] = final_metrics

    # --- Lưu báo cáo cuối cùng ---
    report_path = "jpeg_only_benchmark_summary.yaml"
    with open(report_path, 'w') as f:
        yaml.dump(all_results, f, indent=2, sort_keys=True)
    
    print("\n\n" + "="*25 + " FINAL JPEG-ONLY BENCHMARK RESULTS " + "="*25)
    print(all_results)
    print(f"Benchmark finished. Full results saved to '{report_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run JPEG (no channel coding) + Channel benchmark.")
    
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to your dataset root (containing 'valid').")
    parser.add_argument('--yolo_model_path', type=str, default="best.pt")
    parser.add_argument('--output_dir', type=str, default="./jpeg_only_benchmark_results")
    
    parser.add_argument('--snr_db_range', nargs='+', type=float, default=[-5, 0, 5, 10, 15, 20, 25, 30])
    parser.add_argument('--channel_type', type=str, default='rayleigh', choices=['awgn', 'rayleigh'])
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--yolo_conf_thres', type=float, default=0.25)
    parser.add_argument('--yolo_iou_thres', type=float, default=0.45)
    
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)