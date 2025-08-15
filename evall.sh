python run_swinjscc_evaluation_sweep.py \
    --history_dir "./history" \
    --fish_dataset_path "" \
    --yolo_weights_path "best.pt" \
    --checkpoint_name "model_best_psnr.pth" \
    --output_report_dir "./final_benchmark_data" \
    --snr_db_range -5 0 5 10 15 20 25 \
    --batch_size_eval 16