# Lightweight-SISR-Review-Materials-Anonymous-Submission
# [Anonymous Submission] Review Materials for BDSRNet

This anonymous repository provides supplementary review materials for the submission  
**“BDSRNet for Lightweight Image Super-Resolution.”**

## Notice on Anonymity and Code Availability

To comply with the double-blind review policy, the full training and inference source code is not included in this anonymous repository during the review stage.

This repository is intended to support inspection of the reported results by providing:
- quantitative result summaries,
- visual materials,
- anonymized training and testing logs, and
- the profiling script used for efficiency evaluation.

The complete executable codebase and pre-trained weights will be released after the review process, subject to the venue policy.

---

## 1. Quantitative Results and Efficiency Analysis

This section summarizes the main quantitative results reported in the manuscript.

### Table 1: Benchmark Performance (PSNR / SSIM)

Benchmark results on five standard super-resolution datasets and two zero-shot remote-sensing datasets are provided in:

- `Images/performance_table.png`

### Table 2: Computational Efficiency and Perceptual Quality

Efficiency and perceptual-quality comparisons on Urban100 for $\times4$ SR are provided in:

- `Images/urban100_x4_efficiency_perceptual_comparison.png`

This figure reports:
- parameter count,
- FLOPs,
- inference latency,
- LPIPS, and
- NIQE.

### Accuracy–Efficiency Trade-off Visualization

A visualization of the performance–parameter–FLOPs trade-off is provided in:

- `Images/performance_vs_params.svg`

---

## 2. Visual Materials

This section provides visual materials corresponding to the manuscript figures.

### Overall Architecture

The overall architecture of BDSRNet is provided in:

- `Images/framework.svg`

---

## 3. Anonymized Experimental Logs

To support inspection of the reported training and testing results, this repository includes anonymized raw logs exported from our experimental runs.

### Training Logs
The directory `Training_Logs/` contains training logs for the $\times2$, $\times3$, and $\times4$ models, including:
- iteration records,
- training loss values, and
- validation PSNR records.

### Testing Logs
The directory `Testing_Logs/` contains testing logs for:
- standard benchmark datasets, and
- zero-shot remote-sensing evaluation.

---

## 4. Evaluation Script for Efficiency Measurement

The script

- `unified_benchmark_params_flops_latency.py`

is included to document the protocol used for efficiency evaluation in the manuscript.

The script includes the profiling setup for:
- parameter counting,
- FLOPs estimation, and
- GPU latency measurement.

Latency measurement follows a fixed protocol with warm-up iterations and synchronized timing.

---

## Repository Structure

```text
Images/
Training_Logs/
Testing_Logs/
unified_benchmark_params_flops_latency.py
README.md
