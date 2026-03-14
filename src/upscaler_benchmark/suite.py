import os
import requests
import numpy as np
import cv2
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from upscaler_benchmark.metrics import Metrics
from upscaler_benchmark.models import Models

class Suite:
    def __init__(self, project_id, dataset, max_workers=12):
        self.models = Models(project_id)
        self.dataset = dataset
        self.results = []
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        os.makedirs("benchmark_results", exist_ok=True)

    def prepare_inputs(self, label, path_or_url):
        """Downloads or loads local image variants."""
        try:
            # Check if it's a local file first
            if os.path.exists(path_or_url):
                original = cv2.imread(path_or_url)
            else:
                resp = requests.get(path_or_url, timeout=15)
                resp.raise_for_status()
                original = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            
            return {"Original": original} if original is not None else None
        except Exception as e:
            # print(f"  Error loading input for {label}: {e}") # Silenced for tqdm
            return None

    def evaluate_task(self, label, tier, model_name, lr_bytes, ref_img, distorted_img):
        # Create subfolder for the image label
        img_results_dir = f"benchmark_results/{label.replace(' ', '_')}"
        os.makedirs(img_results_dir, exist_ok=True)
        
        # Save reference assets once
        cv2.imwrite(f"{img_results_dir}/original.png", ref_img)
        cv2.imwrite(f"{img_results_dir}/distorted.png", distorted_img)

        # print(f"    [RUNNING] {label} | {tier} | {model_name}...") # Silenced for tqdm
        output = self.models.run_tier(model_name, lr_bytes, tier=tier)
        if output is not None:
            # print(f"    [METRICS] Calculating for {label} {tier} {model_name}...") # Silenced for tqdm
            metrics = Metrics.run_all(ref_img, output)
            metrics.update({"Label": label, "Tier": tier, "Model": model_name})
            with self.results_lock:
                self.results.append(metrics)
            
            # Save upscaled result
            cv2.imwrite(f"{img_results_dir}/{model_name.lower()}_{tier.lower()}.png", output)
            # print(f"    [DONE] {label} | {tier} | {model_name} complete.") # Silenced for tqdm
        else:
            # print(f"    [FAILED] {label} | {tier} | {model_name} (No output)") # Silenced for tqdm
            pass

    def run_benchmark(self):
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            print(f"📦 Preparing benchmark tasks for {len(self.dataset)} images...")
            for label, url in self.dataset.items():
                inputs = self.prepare_inputs(label, url)
                if not inputs: continue
                tiers = {"1K": 1024} # Target output height
                for tier, target_h in tiers.items():
                    # 1. Prepare Reference (1K ground truth)
                    h, w = inputs["Original"].shape[:2]
                    ref_scale = target_h / h
                    ref_1k = cv2.resize(inputs["Original"], (int(w*ref_scale), target_h), interpolation=cv2.INTER_AREA)

                    # 2. Apply Real-World Distortion
                    # Downsample to 500px height (Classic Upscaling task)
                    low_res_h = 500
                    scale_lr = low_res_h / h
                    distorted = cv2.resize(inputs["Original"], (int(w*scale_lr), low_res_h), interpolation=cv2.INTER_AREA)
                    
                    # Add Blur (Simulate sensor/lens blur)
                    distorted = cv2.GaussianBlur(distorted, (3, 3), 0)
                    
                    # Add JPEG Compression Artifacts (Simulate web transmission)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                    _, encimg = cv2.imencode('.jpg', distorted, encode_param)
                    distorted_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
                    
                    # Convert to bytes for API
                    lr_bytes = cv2.imencode('.png', distorted_img)[1].tobytes()
                    
                    for model in ["Nano Banana 2", "Imagen"]:
                        tasks.append(executor.submit(self.evaluate_task, label, tier, model, lr_bytes, ref_1k, distorted_img))
            
            print(f"🚀 Running {len(tasks)} upscaling tasks...")
            for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Experimental Comparison", unit="task"):
                pass
                
        if self.results:
            pd.DataFrame(self.results).to_csv("benchmark_results/comparison_matrix.csv", index=False)
