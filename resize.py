from PIL import Image

def resize_image(input_path, output_path, size=(256, 256)):
    """
    Resize một ảnh về kích thước mong muốn và lưu lại.
    
    :param input_path: Đường dẫn tới ảnh gốc.
    :param output_path: Đường dẫn để lưu ảnh đã resize.
    :param size: Tuple chứa chiều rộng và chiều cao mới, ví dụ (256, 256).
    """
    try:
        # Mở ảnh từ đường dẫn
        with Image.open(input_path) as img:
            # Resize ảnh
            # Image.Resampling.LANCZOS là một bộ lọc giúp ảnh sau khi thu nhỏ
            # có chất lượng tốt nhất, tránh bị răng cưa.
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            
            # Lưu ảnh đã resize
            img_resized.save(output_path)
            
            print(f"Đã resize ảnh thành công và lưu tại: {output_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại '{input_path}'")
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")

# --- Cách sử dụng ---

# 1. Thay 'anh_goc.jpg' bằng tên file ảnh của bạn
input_image_path = 'frame_028500.png'

# 2. Đặt tên cho file ảnh sau khi resize
output_image_path = 'ori_resize.jpg'

# 3. Gọi hàm để thực hiện
resize_image(input_image_path, output_image_path)