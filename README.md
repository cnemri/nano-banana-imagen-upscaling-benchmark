# 🖼️ Imagen Upscale vs. Nano Banana 2: An Upscaling Experiment

This repository contains a controlled experimental study designed to evaluate and compare the performance of **Nano Banana 2** (Gemini 3.1 Flash) and **Imagen 4.0** (Upscale) in high-fidelity resolution enhancement.

---

## 👨‍🔬 Author
**Chouaieb Nemri**  
*Generative AI BlackBelt Specialist @ Google*  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/nemri/)

---

## 🧪 The Experiment
This repository hosts a controlled experiment comparing two industry-leading models:
1.  **Nano Banana 2**: (Gemini 3.1 Flash Image Preview) Evaluated for its multi-modal reasoning and high-fidelity reconstruction capabilities.
2.  **Imagen 4.0 (Upscale Preview)**: Google's flagship diffusion-based upscaling model.

### 🔬 Experimental Methodology: Image Degradation & Restoration
To evaluate the models' true restoration capabilities, this experiment uses a multi-stage degradation pipeline rather than simple interpolation. This evaluates the model's ability to perform **Super-Resolution** and **Artifact Restoration** simultaneously.

1.  **Step 1: Downsampling**: The high-resolution original is downscaled to a **500px baseline** (height), physically removing ~75% of the original pixel data.
2.  **Step 2: Blurring**: A **3x3 Gaussian Blur** is applied to simulate lens softening and sensor noise.
3.  **Step 3: Compression**: The image is encoded with **JPEG Quality 40**, introducing blocking and ringing artifacts common in web transmission.
4.  **Step 4: AI Reconstruction**: The models (Nano Banana 2 / Imagen) attempt to upscale this "dirty" low-res input back to **1K resolution**.
5.  **Step 5: Evaluation**: The 1K output is compared against the **original image** (downsampled to a 1K ground truth) for precision.

---

## 🏆 Experimental Results (33-Image Landscape & Architecture Corpus)

The following metrics represent the **Aggregated Overall Performance** across our 33-image corpus. **Imagen** (v4.0 Upscale) has emerged as the dominant leader in high-fidelity restoration.

| Metric | Imagen (Avg) | Nano Banana 2 (Avg) | Winner |
| :--- | :--- | :--- | :--- |
| **PSNR** (Higher is Better) | **24.7207** | 22.2186 | **Imagen** |
| **SSIM** (Higher is Better) | **0.7040** | 0.6588 | **Imagen** |
| **LPIPS** (Lower is Better) | **0.1668** | 0.2287 | **Imagen** |

### 💡 The Quality-Price Tradeoff
In this comprehensive 33-image evaluation, **Imagen** proved to be the superior model for extreme restoration tasks involving blur and JPEG noise.

*   **Imagen (Winner)**: The definitive choice for applications requiring **maximum perceptual fidelity** and scientific accuracy in image reconstruction.
*   **Nano Banana 2**: While highly competitive in latency and cost (as a "Flash" model), it exhibits a delta in reconstruction accuracy compared to Imagen's specialized diffusion-based architecture.

---

## 📊 Metric Analysis & Educational Guide

A "non-AI sloppy" approach to quality assessment requires a multi-dimensional view of image data. We evaluate across three critical pillars:

### 1. PSNR (Peak Signal-to-Noise Ratio)
*   **Significance**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.
*   **Calculation**: Inversely proportional to the Logarithmic Mean Squared Error (MSE).
*   **Interpretation**: **Higher is Better**. Represents mathematical reconstruction accuracy.
*   **Classic Resource**: *Standard Signal Processing Theory.*

### 2. SSIM (Structural Similarity Index)
*   **Significance**: Unlike PSNR, SSIM is based on the human visual system's aptitude for extracting structural information. It evaluates luminance, contrast, and structural similarity.
*   **Calculation**: A weighted combination of three comparison functions: luminance, contrast, and structure.
*   **Interpretation**: **Higher is Better** (1.0 = Perfect match).
*   **Citation**: [Wang, Z., et al. (2004). "Image quality assessment: from error visibility to structural similarity."](https://ieeexplore.ieee.org/document/1284395)

### 3. LPIPS (Learned Perceptual Image Patch Similarity)
*   **Significance**: The modern "Gold Standard" for AI. It uses deep features (AlexNet/VGG) to measure how similar two images look to a human eye, capturing textures and artifacts that PSNR/SSIM often miss.
*   **Calculation**: Deep feature extraction and cosine distance calculation across intermediate layers.
*   **Interpretation**: **Lower is Better** (0.0 = Perceptually identical).
*   **Citation**: [Zhang, R., et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."](https://arxiv.org/abs/1801.03924)

---

## 📂 Project Structure
```text
upscaler_benchmark/
├── src/upscaler_benchmark/  # Professional package (flattened for navigation)
│   ├── models.py            # API Orchestration (Gemini/Imagen)
│   ├── metrics.py           # Evaluation Engine (PSNR/SSIM/LPIPS)
│   └── suite.py             # Internal Orchestrator
├── scripts/
│   └── run_benchmark.py     # Unified execution & analysis pipeline
├── data/                    # Benchmark assets (1.png... 4.jpg)
└── benchmark_results/       # Automated cleanup & CSV matrix (gitignored)
```

---

## 🚀 Getting Started

### 📦 Installation
Maintain environment parity using `uv`:
```bash
uv sync
```

### 📈 Execution
Run the unified pipeline to upscale your dataset, calculate metrics, and generate an automated comparative report:
```bash
uv run scripts/run_benchmark.py
```

---

## 📄 License
Apache 2.0
