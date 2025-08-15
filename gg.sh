python train_swinjscc_on_fish_dataset.py \
    --fish_dataset_path "" \
    --yolo_weights_path "best.pt" \
    --model "SwinJSCC_w/o_SAandRA" \
    --channel-type "rayleigh" \
    --C "20" \
    --multiple-snr "10" \
    --input_size 256 \
    --batch_size 8 \
    --epochs 500 \
    --snr_eval 10.0 \
    --save_freq 20 \
    # --resume "./history/SwinJSCC_Fish_StdMSE_SwinJSCC_w-o_SAandRA_rayleigh_C128_SNR10/models/model_epoch_36.pth" \