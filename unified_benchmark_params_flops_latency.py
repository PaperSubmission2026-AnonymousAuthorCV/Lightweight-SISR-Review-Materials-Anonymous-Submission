import math
import time
import torch
from thop import profile
import sys
import os

# Forcefully add the root directory to PYTHONPATH for module resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ==========================================
# 1. Import Macro Architecture Ablation Models
# ==========================================
from basicsr.archs.BDSRNet_arch import BDSRNet
from basicsr.archs.ablations.BDSRNet_PlainBackbone_arch import BDSRNet_PlainBackbone
from basicsr.archs.ablations.BDSRNet_WoBDM_arch import BDSRNet_WoBDM
from basicsr.archs.ablations.BDSRNet_WoCTRM_arch import BDSRNet_WoCTRM
from basicsr.archs.ablations.BDSRNet_WoSBNA_arch import BDSRNet_WoSBNA

# ==========================================
# 2. Import Micro-BDM Ablation Models (2x2)
# ==========================================
from basicsr.archs.ablations.BDSRNet_WoSBNA_WoDistill_BDM3x3x3_arch import BDSRNet_WoSBNA_WoDistill_BDM3x3x3
from basicsr.archs.ablations.BDSRNet_WoSBNA_WoDistill_arch import BDSRNet_WoSBNA_WoDistill
from basicsr.archs.ablations.BDSRNet_WoSBNA_BDM3x3x3_arch import BDSRNet_WoSBNA_BDM3x3x3

# ==========================================
# 3. Import Micro-CTRM Ablation Models (6-Row)
# ==========================================
from basicsr.archs.ablations.BDSRNet_WoSBNA_WoCCA_WoTFEB_arch import BDSRNet_WoSBNA_WoCCA_WoTFEB
from basicsr.archs.ablations.BDSRNet_WoSBNA_WoTFEB_arch import BDSRNet_WoSBNA_WoTFEB
from basicsr.archs.ablations.BDSRNet_WoSBNA_WoCCA_arch import BDSRNet_WoSBNA_WoCCA
from basicsr.archs.ablations.BDSRNet_WoSBNA_TFEB_Single_arch import BDSRNet_WoSBNA_TFEB_Single
from basicsr.archs.ablations.BDSRNet_WoSBNA_TFEB_NoDil_arch import BDSRNet_WoSBNA_TFEB_NoDil

# ==========================================
# 4. Import CTRM External Comparisons (Other Attentions) 
# ==========================================
from basicsr.archs.ablations.BDSRNet_WoSBNA_Attn_CBAM_arch import BDSRNet_WoSBNA_Attn_CBAM
from basicsr.archs.ablations.BDSRNet_WoSBNA_Attn_ESA_arch import BDSRNet_WoSBNA_Attn_ESA

# ==========================================
# 5. Import NLSA Comparison and SBNA Position Ablations
# ==========================================
from basicsr.archs.ablations.BDSRNet_Compare_NLSA_arch import BDSRNet_NLSA
from basicsr.archs.ablations.BDSRNet_SBNA_Pos_arch import BDSRNet_SBNA_Pos

# ==========================================
# 6. Import Blocks Depth Ablations
# ==========================================
from basicsr.archs.ablations.BDSRNet_WoSBNA_Blocks_arch import BDSRNet_WoSBNA_Blocks

def evaluate_unified_benchmark():
    # ==========================
    # Unified Benchmark Protocol
    # ==========================
    ref_hr_h, ref_hr_w = 720, 1280
    scale = 4 
    reps = 100    # Number of repetitions for stable latency profiling
    warmup = 30   # Number of GPU warm-up iterations
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 135)
    print("--- UNIFIED BENCHMARK: PARAMS, FLOPS, AND LATENCY (x4, 720P Reference) ---")
    print(f"- Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"- Protocol: Average of {reps} passes after {warmup} warm-ups")
    print("=" * 135)

    model_groups = [
        ("[Table 1: Macro Architecture Ablation]", {
            "Plain Backbone": BDSRNet_PlainBackbone(upscale=scale),
            "w/o BDM": BDSRNet_WoBDM(upscale=scale),
            "w/o CTRM": BDSRNet_WoCTRM(upscale=scale),
            "w/o SBNA": BDSRNet_WoSBNA(upscale=scale),
            "BDSRNet (Full)": BDSRNet(upscale=scale),
        }),
        ("[Table 2: Micro-BDM Internal Mechanism Ablation]", {
            "Direct Fusion (Fixed)": BDSRNet_WoSBNA_WoDistill_BDM3x3x3(upscale=scale),
            "Progressive Direct Fusion": BDSRNet_WoSBNA_WoDistill(upscale=scale),
            "Distilled Fusion": BDSRNet_WoSBNA_BDM3x3x3(upscale=scale),
            "Prog. Distilled Fusion (Ours)": BDSRNet_WoSBNA(upscale=scale),
        }),
        ("[Table 3: Micro-CTRM Internal Structure Ablation]", {
            "1. Plain Post-fusion Projection [WoCCA_WoTFEB]": BDSRNet_WoSBNA_WoCCA_WoTFEB(upscale=scale),
            "2. Channel-only Refinement      [WoTFEB]": BDSRNet_WoSBNA_WoTFEB(upscale=scale),
            "3. Spatial-only Refinement      [WoCCA]": BDSRNet_WoSBNA_WoCCA(upscale=scale),
            "4. CCA + Single-branch TFEB     [TFEB_Single]": BDSRNet_WoSBNA_TFEB_Single(upscale=scale),
            "5. CCA + Dual-branch (NoDil)    [TFEB_NoDil]": BDSRNet_WoSBNA_TFEB_NoDil(upscale=scale),
            "6. Full CTRM (Ours)             [WoSBNA]": BDSRNet_WoSBNA(upscale=scale),
        }),
        ("[Additional 1: CTRM External Attention Comparisons]", {
            "CTRM replaced with CBAM         [WoSBNA_CBAM]": BDSRNet_WoSBNA_Attn_CBAM(upscale=scale),
            "CTRM replaced with ESA          [WoSBNA_ESA]": BDSRNet_WoSBNA_Attn_ESA(upscale=scale),
        }),
        ("[Additional 2: NLSA Comparison & SBNA Position Ablation]", {
            "BDSRNet_NLSA (+NLSA standard conv)": BDSRNet_NLSA(upscale=scale),
            "BDSRNet_SBNA_Pos (sbna_pos='2_4')": BDSRNet_SBNA_Pos(upscale=scale, sbna_pos='2_4'),
            "BDSRNet_SBNA_Pos (sbna_pos='4_8')": BDSRNet_SBNA_Pos(upscale=scale, sbna_pos='4_8'),
        }),
        ("[Additional 3: Network Depth (EMFEM Blocks Number)]", {
            "BDSRNet_WoSBNA (Blocks = 4)": BDSRNet_WoSBNA_Blocks(upscale=scale, num_block=4),
            "BDSRNet_WoSBNA (Blocks = 6)": BDSRNet_WoSBNA_Blocks(upscale=scale, num_block=6),
            "BDSRNet_WoSBNA (Blocks = 8, Default)": BDSRNet_WoSBNA_Blocks(upscale=scale, num_block=8),
            "BDSRNet_WoSBNA (Blocks = 10)": BDSRNet_WoSBNA_Blocks(upscale=scale, num_block=10),
        }),
        ("[Additional 4: Network Width (Channel Capacity)]", {
            "BDSRNet (Channels = 32)": BDSRNet(upscale=scale, num_feat=32, num_atten=32),
            "BDSRNet (Channels = 48)": BDSRNet(upscale=scale, num_feat=48, num_atten=48),
            "BDSRNet (Channels = 56, Default)": BDSRNet(upscale=scale, num_feat=56, num_atten=56),
            "BDSRNet (Channels = 64)": BDSRNet(upscale=scale, num_feat=64, num_atten=64),
        })
    ]

    lr_h = ref_hr_h // scale
    lr_w = math.ceil(ref_hr_w / scale)
    inputs = torch.randn(1, 3, lr_h, lr_w, device=device)

    for group_name, models in model_groups:
        print(f"\n{group_name}")
        print("-" * 135)
        print(f"{'Model Variant':<55} | {'Params(K)':<12} | {'FLOPs(G)':<12} | {'Latency(ms)':<15} | {'Mem(MB)':<10}")
        print("-" * 135)

        for name, model_class in models.items():
            model = model_class.to(device)
            model.eval()

            # 1. Profile Parameters and FLOPs
            flops, params = profile(model, (inputs,), verbose=False)
            params_k = params / 1e3
            flops_g = flops / 1e9

            # 2. GPU Warm-up phase
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(inputs)

            # Reset peak memory statistics for accurate tracking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # 3. Benchmark Inference Latency
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps):
                    _ = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            avg_latency_ms = (end_time - start_time) * 1000.0 / reps

            # 4. Measure Peak Memory consumption
            peak_mem_mb = 0.0
            if torch.cuda.is_available():
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Print the aggregated metrics
            print(f"{name:<55} | {params_k:<12.2f} | {flops_g:<12.2f} | {avg_latency_ms:<15.2f} | {peak_mem_mb:<10.2f}")

            # Clear memory to prevent OOM across iterations
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    print("=" * 135)
    print("--- Evaluation completed ---")

if __name__ == '__main__':
    evaluate_unified_benchmark()