# train_swinjscc_with_resume.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import time
from ultralytics import YOLO
from tqdm import tqdm
import logging
from collections import OrderedDict

# --- 1. THÊM ĐƯỜNG DẪN ĐỂ IMPORT CÁC MODULE ---
SWINJSCC_CODE_PATH = "./swinjscc_code"
AQUASC_CODE_PATH = "./aquasc_code"
sys.path.insert(0, os.path.abspath(SWINJSCC_CODE_PATH))
sys.path.insert(0, os.path.abspath(AQUASC_CODE_PATH))

# --- 2. IMPORT CÁC THÀNH PHẦN CẦN THIẾT ---
try:
    from net.network import SwinJSCC
    from utils import AverageMeter, seed_torch
    import torch.nn as nn
except ImportError as e:
    print(f"Lỗi import từ SwinJSCC: {e}"); sys.exit(1)

try:
    from datasets import build_dataset, yolo_collate_fn
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError as e:
    print(f"Lỗi import từ AquaSC: {e}"); sys.exit(1)

# --- 3. CÁC HÀM TIỆN ÍCH VÀ ĐÁNH GIÁ ---
def save_checkpoint(epoch, model, optimizer, save_path, best_metric):
    """Lưu checkpoint bao gồm epoch, model, optimizer state, và best metric."""
    print(f"DEBUG: Saving checkpoint for epoch {epoch} to {save_path}")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(), # << Key này rất quan trọng
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_metric, 
    }
    torch.save(state, save_path)

def evaluate_swinjscc_with_yolo(swinjscc_net, yolo_net, dataloader, device, snr_eval, rate_eval, args_eval):
    swinjscc_net.eval()
    if yolo_net: yolo_net.eval()
    psnr_meter = AverageMeter()
    map_calculator = None
    if yolo_net and TORCHMETRICS_AVAILABLE:
        map_calculator = MeanAveragePrecision(iou_type="bbox").to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at SNR={snr_eval}dB", leave=False):
            (original_imgs, _), (_, yolo_gt_targets_list, _) = batch
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
                try: map_calculator.update(preds_for_metric, gts_for_metric)
                except Exception: pass
    final_stats = {"psnr": psnr_meter.avg}
    if map_calculator:
        map_results = map_calculator.compute()
        final_stats["map"] = map_results['map'].item()
        final_stats["map_50"] = map_results['map_50'].item()
    return final_stats

# --- 4. SCRIPT HUẤN LUYỆN CHÍNH ---
def main(args):
    safe_model_name = args.model.replace('/', '-')
    filename = f"SwinJSCC_Fish_{args.distortion_metric}_{safe_model_name}_{args.channel_type}_C{args.C}_SNR{args.multiple_snr.replace(',', '_')}"
    workdir = Path(f'./history/{filename}')
    models_dir = workdir / 'models'
    workdir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    log_file_path = workdir / f'Log_{filename}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file_path, mode='a'), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()
    
    class Config: pass
    config = Config(); config.logger = logger
    config.seed = 42; config.pass_channel = True; config.CUDA = True
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.learning_rate = 1e-5
    config.batch_size = args.batch_size
    config.downsample = 4; config.image_dims = (3, args.input_size, args.input_size); config.norm = False
    
    logger.info("="*50); logger.info(f"STARTING/RESUMING SESSION: {filename}"); logger.info(args); seed_torch(config.seed)

    channel_number = int(args.C) if args.C.isdigit() else None
    encoder_kwargs = dict(img_size=(args.input_size, args.input_size), patch_size=2, in_chans=3, embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    decoder_kwargs = dict(img_size=(args.input_size, args.input_size), embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    config.encoder_kwargs = encoder_kwargs; config.decoder_kwargs = decoder_kwargs

    logger.info("Initializing SwinJSCC model..."); net = SwinJSCC(args, config).to(config.device)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    start_epoch = 0; best_psnr = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=config.device)
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
            if 'best_psnr' in checkpoint: best_psnr = checkpoint['best_psnr']
            logger.info(f"Resumed from epoch {start_epoch}. Previous best PSNR: {best_psnr:.2f}")
        else: logger.error(f"Checkpoint file not found at '{args.resume}'. Starting from scratch.")
    else: logger.info("No checkpoint provided, starting from scratch.")

    logger.info("Building datasets and dataloaders...")
    args_for_dataset = argparse.Namespace(data_path=args.fish_dataset_path, data_set='fish', input_size=args.input_size, patch_size=8, mask_ratio=0.0, num_object_classes=1, eval=False)
    train_dataset = build_dataset(is_train=True, args=args_for_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=yolo_collate_fn, drop_last=True)
    args_for_dataset.eval = True
    val_dataset = build_dataset(is_train=False, args=args_for_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=yolo_collate_fn)
    logger.info("Loading YOLO model..."); yolo_net = YOLO(args.yolo_weights_path).to(config.device)
    
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs):
        net.train(); losses = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            (original_imgs, _), (_, _, _) = batch
            original_imgs = original_imgs.to(config.device)
            _, _, _, _, loss_dist = net(original_imgs)
            total_loss = loss_dist
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            losses.update(total_loss.item())
            pbar.set_postfix(mse_loss=f"{losses.avg:.4f}")

        eval_stats = evaluate_swinjscc_with_yolo(net, yolo_net, val_loader, config.device, snr_eval=args.snr_eval, rate_eval=int(args.C), args_eval=args)
        logger.info(f"Epoch {epoch+1} Eval Results: PSNR={eval_stats.get('psnr', 0):.2f}, mAP@50={eval_stats.get('map_50', 0):.4f}, mAP@[.5:.95]={eval_stats.get('map', 0):.4f}")

        current_psnr = eval_stats.get('psnr', 0.0)
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            logger.info(f"*** New best PSNR: {best_psnr:.2f}. Saving model... ***")
            save_checkpoint(epoch, net, optimizer, 
                            save_path=os.path.join(models_dir, 'model_best_psnr.pth'), 
                            best_metric=best_psnr)
        
        if (epoch + 1) % args.save_freq == 0:
            logger.info(f"Saving periodic checkpoint at epoch {epoch+1}...")
            save_checkpoint(epoch, net, optimizer, 
                            save_path=os.path.join(models_dir, f'model_epoch_{epoch+1}.pth'), 
                            best_metric=best_psnr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SwinJSCC (Standard MSE Loss) on Custom Dataset')
    parser.add_argument('--model', type=str, default='SwinJSCC_w/o_SAandRA')
    parser.add_argument('--channel-type', type=str, default='rayleigh')
    parser.add_argument('--C', type=str, required=True)
    parser.add_argument('--multiple-snr', type=str, required=True)
    parser.add_argument('--distortion-metric', type=str, default='MSE')
    parser.add_argument('--fish_dataset_path', type=str, required=True)
    parser.add_argument('--yolo_weights_path', type=str, default="best.pt")
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--snr_eval', type=float, default=15.0)
    parser.add_argument('--yolo_conf_thres', type=float, default=0.4)
    parser.add_argument('--yolo_iou_thres', type=float, default=0.5)
    # THÊM THAM SỐ RESUME
    parser.add_argument('--resume', type=str, default="", help="Path to checkpoint to resume training from.")
    
    cli_args = parser.parse_args()
    try: int(cli_args.C)
    except ValueError: print("ERROR: --C must be a single integer for this script."); sys.exit(1)
    try: float(cli_args.multiple_snr)
    except ValueError: print("ERROR: --multiple-snr must be a single number for this script."); sys.exit(1)
    main(cli_args)