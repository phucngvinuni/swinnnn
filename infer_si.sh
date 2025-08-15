

echo "Starting SwinJSCC single image inference script..."

# --- CÁC THAM SỐ CẤU HÌNH ---

# 1. Đường dẫn
PROJECT_ROOT="." # Giả sử đang ở thư mục gốc của SwinJSCC
HISTORY_DIR="history"
INPUT_IMAGE="ori_resize.jpg" # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY

# 2. Chọn Checkpoint và Cấu hình tương ứng
# Ví dụ: bạn muốn tái tạo ảnh bằng model C=27 đã huấn luyện ở SNR 20dB
RUN_NAME="SwinJSCC_Fish_StdMSE_SwinJSCC_w-o_SAandRA_rayleigh_C128_SNR10" # Tên thư mục lần chạy
CHECKPOINT_NAME="model_epoch_160.pth" # Tên file checkpoint

# Tham số kiến trúc của checkpoint này
MODEL_TYPE="SwinJSCC_w/o_SAandRA"
MODEL_C=128
# Giả sử đây là model 'base'
MODEL_EMBED_DIMS="128 192 256 320"
MODEL_DEPTHS="2 2 6 2"
MODEL_NUM_HEADS="4 6 8 10"

# 3. Tham số Inference
SNR_EVAL=0.0 # SNR bạn muốn thử nghiệm
OUTPUT_DIR="./swinjscc_inference_output"

# ------------------------------------

FULL_CHECKPOINT_PATH="${HISTORY_DIR}/${RUN_NAME}/models/${CHECKPOINT_NAME}"

# Tạo tên file output tự động
FILENAME=$(basename -- "$INPUT_IMAGE")
OUTPUT_IMAGE_PATH="${OUTPUT_DIR}/${FILENAME%.*}_recon_C${MODEL_C}_snr${SNR_EVAL}dB.png"

# Kiểm tra các file cần thiết
if [ ! -f "$FULL_CHECKPOINT_PATH" ]; then
    echo "Lỗi: Không tìm thấy checkpoint tại '$FULL_CHECKPOINT_PATH'"
    exit 1
fi
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Lỗi: Không tìm thấy ảnh đầu vào tại '$INPUT_IMAGE'"
    exit 1
fi

# Gọi script Python
python3 infer_single_swinjscc.py \
    --checkpoint_path "${FULL_CHECKPOINT_PATH}" \
    --input_image_path "${INPUT_IMAGE}" \
    --output_image_path "${OUTPUT_IMAGE_PATH}" \
    --model "${MODEL_TYPE}" \
    --C ${MODEL_C} \
    --embed_dims ${MODEL_EMBED_DIMS} \
    --depths ${MODEL_DEPTHS} \
    --num_heads ${MODEL_NUM_HEADS} \
    --snr_eval ${SNR_EVAL} \
    --channel_type "rayleigh" # Phải khớp với kênh model đã được huấn luyện trên đó

echo "Script finished. Reconstructed image saved to ${OUTPUT_IMAGE_PATH}"