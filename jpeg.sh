python run_jpeg_only_benchmark.py \
    --dataset_path "" \
    --yolo_model_path "best.pt" \
    --output_dir "jpeg_only_results" \
    --snr_db_range -5 0 5 10 15 20 25 \
    --channel_type "rayleigh" \
    --batch_size 16 \
    --input_size 256