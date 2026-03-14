import warnings
# Suppress torchvision weights deprecation warnings triggered by lpips globally
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import structural_similarity as ssim_calc

class Metrics:
    _lpips_model = None

    @classmethod
    def get_lpips_model(cls):
        if cls._lpips_model is None:
            cls._lpips_model = lpips.LPIPS(net='alex') 
        return cls._lpips_model

    @staticmethod
    def calculate_lpips(ref_img, eval_img):
        try:
            model = Metrics.get_lpips_model()
            img0 = lpips.im2tensor(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            img1 = lpips.im2tensor(cv2.cvtColor(eval_img, cv2.COLOR_BGR2RGB))
            dist = model(img0, img1)
            return {"LPIPS": float(dist.item())}
        except Exception as e:
            return {"LPIPS": -1.0}

    @staticmethod
    def calculate_deterministic(ref_img, eval_img):
        if ref_img.shape != eval_img.shape:
            eval_img = cv2.resize(eval_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        p = psnr_calc(ref_img, eval_img)
        s = ssim_calc(ref_img, eval_img, channel_axis=2)
        return {"PSNR": p, "SSIM": s}

    @classmethod
    def run_all(cls, ref_img, eval_img):
        if ref_img.shape != eval_img.shape:
            eval_img = cv2.resize(eval_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        results = cls.calculate_deterministic(ref_img, eval_img)
        results.update(cls.calculate_lpips(ref_img, eval_img))
        return results
