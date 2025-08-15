import numpy as np
import torch
import random
import os
import logging
import time
from pathlib import Path # Thêm import Path

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def makedirs(directory):
    """Tạo thư mục nếu nó chưa tồn tại."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def logger_configuration(config, save_log=False, test_mode=False):
    """
    Hàm logger gốc và đơn giản từ SwinJSCC.
    Nó chỉ tạo các thư mục cơ bản cần thiết cho việc huấn luyện.
    """
    logger = logging.getLogger("SwinJSCC_Training")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        # Chỉ tạo các thư mục mà quá trình huấn luyện thực sự dùng
        if hasattr(config, 'samples'): makedirs(config.samples)
        if hasattr(config, 'models'): makedirs(config.models)

    # Cấu hình handler và formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    
    # Xóa các handler cũ để tránh log bị lặp
    if logger.hasHandlers():
        logger.handlers.clear()

    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)

    if save_log:
        if hasattr(config, 'log') and os.path.exists(config.log):
            os.remove(config.log)
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
        
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def save_model(model, save_path):
    """Lưu state_dict của model."""
    torch.save(model.state_dict(), save_path)


def seed_torch(seed=42):
    """Thiết lập seed cho các thư viện để đảm bảo tính tái lập."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True