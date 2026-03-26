# Lightweight-SISR-Review-Materials-Anonymous-Submission
# Anonymous Review Materials for BDSRNet

This anonymous repository provides supplementary review materials for the submission:
**"BDSRNet for Lightweight Image Super-Resolution"**

## Notice on Anonymity and Code Availability
To comply with the double-blind review policy, the full training and inference source code is not included in this repository during the review stage. 

This repository is strictly intended to support the inspection of the reported results by providing:
- quantitative result summaries,
- network architecture and visual comparisons,
- anonymized training and testing logs, and
- the profiling script used for hardware efficiency evaluation.

The complete executable codebase and pre-trained weights will be released after the review process, subject to the venue policy.

---

## 1. Quantitative Results and Efficiency Analysis
This section summarizes the main quantitative results reported in the manuscript.

### Table 1: Standard Benchmark Performance
Benchmark results (PSNR / SSIM) on five standard super-resolution datasets are provided in:
- `Images/performance_table.png`

![Benchmark Results](Images/performance_table.png)

### Table 2: Computational Efficiency and Perceptual Quality
Efficiency and perceptual-quality comparisons on the Urban100 dataset for $\times 4$ SR are provided in:
- `Images/urban100_x4_efficiency_perceptual_comparison.png`

This table reports:
- parameter count,
- FLOPs,
- inference latency,
- LPIPS, and
- NIQE.

![Efficiency Results](Images/urban100_x4_efficiency_perceptual_comparison.png)

### Table 3: Zero-shot Remote-Sensing Evaluation
Results demonstrating the zero-shot cross-domain performance on UCM and RSSCN7 datasets are provided in:
- `Images/zeroshot_remote_sensing.png`

![Zero-shot RS](Images/zeroshot_remote_sensing.png)

### Accuracy–Efficiency Trade-off
A visualization of the performance–parameter–FLOPs trade-off is provided in:
- `Images/performance_vs_params.svg`

![Trade-off](Images/performance_vs_params.svg)

---

## 2. Visual Materials
This section provides visual materials corresponding to the network design and reconstruction performance.

### Overall Architecture
The framework diagram of BDSRNet is provided in:
- `Images/framework.svg`

![Framework](Images/framework.svg)

### Visual Comparisons
Reconstructed image comparisons against baseline methods are provided in:
- Urban100: `Images/visual_comparison_urban100.svg`
- Manga109: `Images/visual_comparison_manga109.svg`

![Urban100 Comparison](Images/visual_comparison_urban100.svg)
![Manga109 Comparison](Images/visual_comparison_manga109.svg)

---

## 3. Anonymized Experimental Logs
To support the inspection of the reported training and testing results, this repository includes anonymized raw logs exported from our experimental runs.

### Training Logs
The directory `Training_Logs/` contains logs for the $\times 2$, $\times 3$, and $\times 4$ models, which record:
- iteration records,
- training loss (`L1Loss`) values, and
- validation PSNR tracking.

### Testing Logs
The directory `Testing_Logs/` contains the raw output logs parsing the evaluation on:
- five standard benchmark datasets, and
- two zero-shot remote-sensing datasets.

---

## 4. Evaluation Script for Efficiency Measurement
The script:
- `unified_benchmark_params_flops_latency.py`

is included to document the exact protocol used for the hardware efficiency evaluation in the manuscript.

This script details the profiling setup for:
- parameter counting,
- FLOPs estimation, and
- GPU latency measurement.

Latency measurement follows a fixed protocol with warm-up iterations and synchronized timing.
