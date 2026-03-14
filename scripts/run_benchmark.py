# /// script
# dependencies = ["google-genai", "requests", "numpy", "opencv-python-headless", "opencv-contrib-python-headless", "scikit-image", "pandas", "torch", "torchvision", "lpips", "matplotlib", "tqdm"]
# ///
import sys
import os
import pandas as pd
import shutil

# Add src to path so we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from upscaler_benchmark.suite import Suite

def main():
    PROJECT_ID = "nemri-genai-bb" # Change this to your project ID
    
    # Dynamically load all valid images from the data/ directory
    DATASET = {}
    data_dir = "data"
    if os.path.exists(data_dir):
        # Sort files to ensure consistent ordering across runs
        valid_files = sorted([f for f in os.listdir(data_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        for f in valid_files:
            label = f.split('.')[0]
            DATASET[f"Image {label}"] = os.path.join(data_dir, f)
            
    if not DATASET:
        print("❌ Error: No images found in the 'data/' directory. Please run 'python scripts/expand_dataset.py' first.")
        return
    
    print("🌟 Starting Imagen Upscale vs. Nano Banana 2: Experimental Comparison")
    
    # Auto-cleanup previous results for a fresh start
    results_dir = "benchmark_results"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        print(f"    [CLEANUP] Deleted stale results directory: {results_dir}")
        
    # Dynamically use all available CPU cores for maximum parallelization
    max_workers = os.cpu_count() or 4
    suite = Suite(PROJECT_ID, DATASET, max_workers=max_workers)
    print(f"    [CONFIG] Parallelizing with {max_workers} workers.")
    suite.run_benchmark()

    # Integrated Automated Analysis
    csv_path = "benchmark_results/comparison_matrix.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        metrics_config = {
            "PSNR": ("Higher is Better", True),
            "SSIM": ("Higher is Better", True),
            "LPIPS": ("Lower is Better", False)
        }
        
        print("\n" + "="*75)
        print("🏆 FINAL BENCHMARK COMPARISON & AGGREGATED WINNERS")
        print("="*75)
        
        # Determine unique images (formerly labels)
        images = df['Label'].unique()
        for img_name in images:
            item_df = df[df['Label'] == img_name]
            gemini_rows = item_df[item_df['Model'] == 'Nano Banana 2']
            imagen_rows = item_df[item_df['Model'] == 'Imagen']
            
            if gemini_rows.empty or imagen_rows.empty:
                print(f"\n⚠️  {img_name}: Incomplete Data (Skipping comparison)")
                continue

            print(f"\n🖼️  {img_name}:")
            for metric, (desc, higher_is_better) in metrics_config.items():
                if metric not in item_df.columns: continue
                g_val = gemini_rows[metric].iloc[0]
                i_val = imagen_rows[metric].iloc[0]
                
                winner = "Nano Banana 2" if (g_val > i_val if higher_is_better else g_val < i_val) else "Imagen"
                print(f"  - {metric:6} ({desc}): {winner:13} (Nano Banana 2: {g_val:.4f} | Imagen: {i_val:.4f})")

        print("\n" + "="*70)
        print("📊 AGGREGATED OVERALL PERFORMANCE (Average of valid comparisons)")
        print("="*70)
        
        # Calculate overall wins based on images where BOTH succeeded
        valid_df = df[df['Label'].isin([img for img in images if len(df[df['Label'] == img]['Model'].unique()) == 2])]
        
        if valid_df.empty:
            print("❌ No valid pairwise comparisons available to aggregate.")
        else:
            total_wins = {"Nano Banana 2": 0, "Imagen": 0}
            for metric, (desc, higher_is_better) in metrics_config.items():
                g_avg = valid_df[valid_df['Model'] == 'Nano Banana 2'][metric].mean()
                i_avg = valid_df[valid_df['Model'] == 'Imagen'][metric].mean()
                winner = "Nano Banana 2" if (g_avg > i_avg if higher_is_better else g_avg < i_avg) else "Imagen"
                total_wins[winner] += 1
                print(f"🔹 {metric:6} ({desc}): {winner:13} (Nano Banana 2 Avg: {g_avg:.4f} | Imagen Avg: {i_avg:.4f})")

            overall_winner = max(total_wins, key=total_wins.get)
            print("\n" + "*"*70)
            print(f"🏅 OVERALL WINNING MODEL: {overall_winner} ({total_wins[overall_winner]}/{len(metrics_config)} Metrics Lead)")
            print("*"*70 + "\n")

if __name__ == "__main__":
    main()
