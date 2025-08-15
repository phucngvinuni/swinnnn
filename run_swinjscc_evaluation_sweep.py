import torch
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import yaml
import re # Thư viện cho regular expression để trích xuất C và SNR từ tên thư mục

# --- 1. THÊM ĐƯỜNG DẪN VÀ IMPORT CÁC MODULE ---
# (Giữ nguyên như script huấn luyện)
SWINJSCC_CODE_PATH = "./swinjscc_code"
AQUASC_CODE_PATH = "./aquasc_code"
sys.path.insert(0, os.path.abspath(SWINJSCC_CODE_PATH))
sys.path.insert(0, os.path.abspath(AQUASC_CODE_PATH))

try:
    from net.network import SwinJSCC
    from utils import AverageMeter, seed_torch
    import torch.nn as nn
    from datasets import build_dataset, yolo_collate_fn
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError as e:
    print(f"Lỗi import: {e}"); sys.exit(1)

# --- 2. HÀM ĐÁNH GIÁ (giống như trong script huấn luyện) ---
def evaluate_swinjscc_with_yolo(swinjscc_net, yolo_net, dataloader, device, snr_eval, rate_eval, args_eval):
    # ... (Copy y nguyên hàm này từ script huấn luyện của bạn) ...
    swinjscc_net.eval()
    if yolo_net: yolo_net.eval()
    psnr_meter = AverageMeter()
    map_calculator = None
    if yolo_net and TORCHMETRICS_AVAILABLE:
        map_calculator = MeanAveragePrecision(iou_type="bbox").to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at SNR={snr_eval}dB, C={rate_eval}", leave=False):
            (original_imgs, _), (target_imgs, yolo_gt_targets_list, _) = batch
            original_imgs = original_imgs.to(device)
            
            recon_image, _, _, _, _ = swinjscc_net(original_imgs, given_SNR=snr_eval, given_rate=rate_eval)
            recon_image = recon_image.clamp(0., 1.)

            mse = torch.mean((original_imgs * 255. - recon_image * 255.)**2)
            if mse.item() > 1e-10:
                psnr = 10 * torch.log10(255.**2 / mse)
                psnr_meter.update(psnr.item(), original_imgs.size(0))
            else:
                psnr_meter.update(100, original_imgs.size(0))

            if yolo_net and map_calculator:
                yolo_preds = yolo_net(recon_image, verbose=False, conf=args_eval.yolo_conf_thres, iou=args_eval.yolo_iou_thres)
                preds_for_metric = [{"boxes": r.boxes.xyxy.to(device), "scores": r.boxes.conf.to(device), "labels": r.boxes.cls.to(device).long()} for r in yolo_preds]
                gts_for_metric = [{"boxes": t['boxes'].to(device), "labels": t['labels'].to(device)} for t in yolo_gt_targets_list]
                try:
                    map_calculator.update(preds_for_metric, gts_for_metric)
                except Exception as e:
                    pass # Bỏ qua lỗi nếu một batch không có box
    
    final_stats = {"psnr": psnr_meter.avg}
    if map_calculator:
        map_results = map_calculator.compute()
        final_stats["map"] = map_results['map'].item()
        final_stats["map_50"] = map_results['map_50'].item()
    
    return final_stats

# --- 3. SCRIPT ĐÁNH GIÁ HÀNG LOẠT CHÍNH ---
def main_sweep(args):
    seed_torch(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Load YOLO model một lần
    if not os.path.exists(args.yolo_weights_path):
        print(f"ERROR: YOLO weights not found at {args.yolo_weights_path}"); return
    yolo_net = YOLO(args.yolo_weights_path).to(device)
    print("YOLO model loaded.")

    # Chuẩn bị Dataloader một lần
    args_for_dataset = argparse.Namespace(
        data_path=args.fish_dataset_path, data_set='fish',
        input_size=args.input_size, patch_size=8,
        mask_ratio=0.0, num_object_classes=1, eval=True
    )
    val_dataset = build_dataset(is_train=False, args=args_for_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=yolo_collate_fn)
    print(f"Validation dataloader created with {len(val_loader)} batches.")

    # Tìm tất cả các thư mục lần chạy trong thư mục history
    history_dir = Path(args.history_dir)
    run_dirs = [d for d in history_dir.iterdir() if d.is_dir()]
    
    if not run_dirs:
        print(f"ERROR: No run directories found in '{args.history_dir}'")
        return

    all_results = {}

    # Lặp qua từng thư mục lần chạy
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        run_name = run_dir.name
        print(f"\n===== Processing Run: {run_name} =====")
        
        # Tìm checkpoint
        checkpoint_path = run_dir / "models" / args.checkpoint_name
        if not checkpoint_path.exists():
            print(f"  - Warning: Checkpoint '{args.checkpoint_name}' not found in {run_dir}. Skipping.")
            continue

        # Trích xuất C và SNR huấn luyện từ tên thư mục
        try:
            c_match = re.search(r'_C(\d+)_', run_name)
            snr_train_match = re.search(r'_SNR([\d\._]+)$', run_name)
            if not c_match or not snr_train_match:
                print(f"  - Warning: Cannot parse C and training SNR from directory name '{run_name}'. Skipping.")
                continue
            
            C_val = int(c_match.group(1))
            snr_train_val = snr_train_match.group(1) # Giữ dưới dạng chuỗi
            print(f"  - Parsed params: C={C_val}, Trained on SNR={snr_train_val}")
        except Exception as e:
            print(f"  - Warning: Error parsing params from '{run_name}': {e}. Skipping.")
            continue
        
        # Tạo đối tượng args và config tạm thời cho model này
        temp_args = argparse.Namespace(
            model=args.model, # Giả sử model arch giống nhau cho tất cả
            channel_type=args.channel_type,
            C=str(C_val),
            multiple_snr=snr_train_val,
            distortion_metric='MSE' # Giả định
        )
        class TempConfig: pass
        temp_config = TempConfig()
        temp_config.logger = None # Không cần logger trong eval
        temp_config.pass_channel = True
        temp_config.device = device
        temp_config.downsample = 4
        temp_config.image_dims = (3, args.input_size, args.input_size)
        temp_config.CUDA = torch.cuda.is_available() and args.device == 'cuda'
        temp_config.norm = False
        temp_config.encoder_kwargs = dict(
            img_size=(args.input_size, args.input_size), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=C_val,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        temp_config.decoder_kwargs = dict(
            img_size=(args.input_size, args.input_size),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=C_val,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

        # Load model với cấu hình đã trích xuất
        net = SwinJSCC(temp_args, temp_config)
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        net.to(device)
        
        # Đánh giá trên dải SNR
        run_results_per_snr = {}
        for snr_eval in args.snr_db_range:
            eval_stats = evaluate_swinjscc_with_yolo(net, yolo_net, val_loader, device,
                                                     snr_eval, C_val, args)
            run_results_per_snr[snr_eval] = eval_stats
        
        all_results[run_name] = run_results_per_snr

    # --- Lưu báo cáo cuối cùng ---
    report_dir = Path(args.output_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "swinjscc_benchmark_summary.yaml"
    with open(report_path, 'w') as f:
        yaml.dump(all_results, f, indent=2, sort_keys=True)
    
    print("\n\n" + "="*25 + " FINAL SWEEP RESULTS " + "="*25)
    # In kết quả ra màn hình cho dễ xem
    for run_name, snr_data in all_results.items():
        print(f"\n--- Results for: {run_name} ---")
        for snr, metrics in snr_data.items():
            print(f"  SNR={snr:<5.1f} | PSNR={metrics.get('psnr', 0):.2f} | mAP@50={metrics.get('map_50', 0):.4f} | mAP@[.5:.95]={metrics.get('map', 0):.4f}")
    
    print(f"\nBenchmark sweep finished. Full results saved to '{report_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Evaluation Sweep for multiple SwinJSCC checkpoints.')
    
    # Paths
    parser.add_argument('--history_dir', type=str, required=True, help="Directory containing all training run folders.")
    parser.add_argument('--fish_dataset_path', type=str, required=True, help="Path to your fish dataset root (containing 'valid' folder).")
    parser.add_argument('--yolo_weights_path', type=str, default="best.pt", help="Path to pretrained YOLO weights.")
    parser.add_argument('--checkpoint_name', type=str, default="model_best_psnr.pth", help="Name of the checkpoint file to load from each run folder.")
    parser.add_argument('--output_report_dir', type=str, default="./swinjscc_sweep_results")
    
    # Evaluation Params
    parser.add_argument('--snr_db_range', nargs='+', type=float, default=[-5, 0, 5, 10, 15, 20, 25])
    parser.add_argument('--batch_size_eval', type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument('--yolo_conf_thres', type=float, default=0.4)
    parser.add_argument('--yolo_iou_thres', type=float, default=0.45)
    
    # Model Architecture Params (phải giống nhau cho các model được so sánh)
    parser.add_argument('--model', type=str, default='SwinJSCC_w/o_SAandRA') # Giả sử chỉ sweep các model loại này
    parser.add_argument('--channel-type', type=str, default='rayleigh')
    parser.add_argument('--input_size', type=int, default=256)
    
    # Common
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)

    cli_args_sweep = parser.parse_args()
    main_sweep(cli_args_sweep)