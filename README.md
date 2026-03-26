# Lightweight-SISR-Review-Materials-Anonymous-Submission
# [Anonymous Submission] BDSRNet for Lightweight Image Super-Resolution

> **📢 Notice for Reviewers (Double-Blind Policy & Reproducibility):**
> To strictly comply with the double-blind peer-review policy and protect unpublished architectural innovations, the core PyTorch module source codes are temporarily omitted in this initial submission. 
> 
> However, we are committed to full transparency. This repository serves as a **Comprehensive Experimental Archive**, providing extensive visual proofs, authentic raw training/testing logs, and our strict evaluation scripts to fully support the authenticity of our reported results. 
> 
> **The complete executable codebase and pre-trained weights will be made publicly available immediately upon acceptance.**

---

## 1. Quantitative Results & Efficiency Analysis
To provide a quick overview of our model's superiority, we summarize the main quantitative results here. Detailed output parsing can be verified in the `Testing_Logs/` directory.

### Table 1: Benchmark Performance (PSNR / SSIM)
Comprehensive evaluation across 5 standard SR benchmarks and 2 zero-shot remote sensing datasets. 

![Benchmark Results](Images/performance_table.png)

### Table 2: Computational Efficiency & Perceptual Quality
We profile the hardware efficiency alongside perceptual metrics (LPIPS and NIQE) on the Urban100 dataset for $\times 4$ SR. BDSRNet achieves highly competitive perceptual quality while maintaining a strict lightweight footprint.
*(Note: The exact benchmarking code used for Params/FLOPs/Latency is provided in `unified_benchmark_params_flops_latency.py` for full transparency.)*

![Efficiency Results](Images/urban100_x4_efficiency_perceptual_comparison.png)

### Performance vs. Parameters Trade-off
Visualizing the accuracy-efficiency trade-off on Urban100 ($\times 4$).

![Performance vs Params](Images/performance_vs_params.svg)

---

## 2. Visual Highlights
We provide high-resolution visual materials that could not be fully displayed in the manuscript due to space constraints.

### Overall Architecture of BDSRNet
![Framework](Images/framework.svg)

---

## 3. Authentic Experimental Logs
To eliminate any concerns regarding result fabrication, we provide the raw execution logs directly exported from our server for all scales ($\times 2, \times 3, \times 4$):
* **`Training_Logs/`**: Real-time tracking of up to 1,000,000 iterations, demonstrating the stable convergence, `L1Loss` dropping curves, and validation PSNR climbing history.
* **`Testing_Logs/`**: Comprehensive parsing results on both standard benchmarks and zero-shot remote sensing datasets.

---

## 4. Rigorous Evaluation Protocol
Lightweight SISR heavily relies on fair computational benchmarking. We provide our profiling script (`unified_benchmark_params_flops_latency.py`) directly in the root directory. 

Reviewers are encouraged to inspect this script to verify our strict evaluation protocol, which utilizes `thop` for parameter/FLOPs counting, explicit GPU warm-ups, and precise `torch.cuda.synchronize()` operations to ensure unbiased and accurate latency profiling across all ablation variants.

---
**Thank you for your time, effort, and constructive feedback in reviewing our work!**
