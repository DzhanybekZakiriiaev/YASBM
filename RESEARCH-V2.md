# YASBM v2 — Research Synthesis & Recommended Architecture

*Produced 2026-07-14 by a two-agent adversarial research round (a red-team critic and a systems architect researched independently, then attacked each other's findings). No code was written during this phase. This document is the deliverable.*

---

## 1. The question

The proposed v2 vision was a five-step reasoning pipeline: (1) precise 3D scene from the video → (2) detect moving/interacting objects → (3) map motion phases → (4) find regime-change frames → (5) run a reasoning model to compute physics per segment and audit it against the video.

Both agents agree the *premise* is sound — generative video measurably fails physics ([Physics-IQ](https://physics-iq.github.io/): best generator scores 29.5%, and visual realism is **uncorrelated** with physical correctness; [VideoPhy](https://arxiv.org/abs/2406.03520): best model 39.6% adherence). The disagreement was about the **measurement chain**, and it produced the central finding of this research.

## 2. The central finding: physics must be measured on the image plane

**Monocular depth cannot support dynamics.** All current metric depth models are single-image models with no temporal consistency term ([MoGe-2](https://arxiv.org/html/2507.02546v1): metric rel. error 15.7%, δ1 76.8%; [Depth Pro](https://arxiv.org/html/2410.02073v1): indoor δ1 as low as 46.6%). Constant scale bias cancels in acceleration — but frame-to-frame jitter does not: for a 15-frame acceleration fit at 30 fps, σ_accel ≈ 56·σ_z per s², so realistic 5–20 cm per-frame depth jitter produces **2.8–11 m/s² of acceleration noise — the same magnitude as gravity**. Every false positive we observed on real clips traces to this.

**Meanwhile CoTracker3 image-plane tracks are sub-pixel (~3 mm at 3 m) — roughly 100× more precise than the depth axis.** Ballistic motion under a near-static pinhole camera is still a parabola in pixel space. [Where Is The Ball (2025)](https://arxiv.org/abs/2506.05763) recovers 3D trajectories from 2D tracks, not from monocular depth, for exactly this reason.

**Independent validation:** [D3 (ICCV 2025)](https://arxiv.org/abs/2508.00701) — *training-free* second-order temporal differences on the image plane — achieves **98.46% mAP on GenVideo** (+10.4 over prior SOTA), with no depth, no 3D reconstruction, no object detection. [Code released](https://github.com/Zig-HS/D3). The physics intuition wins precisely when you stop trying to reconstruct the world first.

**Corollary (VLM check):** VLM-as-detector is dead on arrival — GPT-4o scores 49.5% on [PhysBench](https://arxiv.org/abs/2501.16411v2) (humans 95.9%) and ~55.6% 2AFC on Physics-IQ. The LLM's role is evidence-grounded *explanation*, never judgment over pixels.

## 3. The converged architecture: **"D3-Anchored Hybrid"**

Detection and explanation are decoupled: a calibrated low-variance image-plane signal decides; the 3D scene explains.

| # | Stage | Role | Status vs current code |
|---|---|---|---|
| 1 | **D3 second-order score** (training-free, per-clip) | **Primary calibrated detector** | new (~1 day) |
| 2 | CoTracker3 + YOLO objects + morph signals (flicker, rigidity CV, bbox jerk) | Secondary evidence; fires on every clip (zero-event fallback) | exists |
| 3 | PELT changepoints on image-plane velocity | **Timeline/UI segmentation only — not evidence** (underpowered at ≤30 samples) | new (small) |
| 4 | Per-segment **image-plane parabola fits** with *parameter-consistency* tests (fitted acceleration direction consistent across segments & objects; leave-one-out stability) — **not** goodness-of-fit (a 3-param parabola on 5–10 points fits any smooth curve; GOF has no power) | Physics evidence when free-flight segments exist | refactor of current fits (~1 day) |
| 5 | MoGe-2 keyframe depth → segment-median scale → **R3F ghost trajectory rendered from the fitted curve** (never forward-integrated from a release state — that amplifies initial-condition noise: σ_v ≈ 0.3–0.9 m/s → 12–36 cm drift by t = 0.4 s), labeled **"illustrative"** in the UI | Explainability only | refactor (~0.5 day) |
| 6 | SAM 2.1 mask-adjacency contact | Optional *soft* evidence — never a gate (no controlled evidence that mask contact beats boxes; occlusion corrupts both at contact) | optional |
| 7 | Claude Sonnet verdict over the structured evidence table {D3 score, morph signals, consistency table, events} | Explanation layer, template-constrained to computed numbers | exists (prompt update) |

**Total effort ≈ 2.5–3.5 days. Latency ~45 s/clip, ~$0.13/clip. All within existing Modal budget.**

### Explicit kill list (things researched and rejected)
- Metric-precision 3D as physics input (jitter math above)
- TSDF fusion for dynamics (reconstructs the static room while carving out the moving objects being audited)
- Forward-integration ghost from release state (measures initial-condition error, not physics)
- Savitzky-Golay pre-smoothing before the detector (circular — erases the second-order artifacts that ARE the discriminative signal; smoothing allowed for display only)
- Contact-from-2D-box-IoU as a physics gate; 100DOH hand-contact stage
- Goodness-of-fit residual tests on ≤10-sample segments
- VLM-native plausibility judging
- Depth Anything V2 Large (CC-BY-NC — license, and relative-only depth), Depth Pro (too slow on L4), MonST3R/MegaSaM (minutes-scale/OOM)
- ViewCrafter novel-view synthesis (already abandoned; 5 failed CUDA builds, wrong tool)

### Held for later (validated, not scheduled)
- **VGGT** (poses in seconds on L4, commercial checkpoint) — when camera motion support is needed
- **ProxyPose** ([arXiv 2607.06555](https://arxiv.org/abs/2607.06555), [code](https://github.com/ruihangzhang97/proxypose)) — 6-DoF pose trajectories would enable rotational-dynamics auditing (angular momentum) and real prop orientation
- **SpatialTrackerV2** — feed-forward world-space 3D tracking, potential CoTracker+lift replacement
- **Day-1 jitter gate** (from the critic): if 3D dynamics are ever reconsidered, first measure per-object MoGe-2 depth second-differences on real static clips; proceed only if jitter < ~1.5%

## 4. Calibration protocol (replaces the invented 3σ/10σ thresholds)

1. **Real side:** [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark) filmed clips + a second source (Pexels/DAVIS) to break dataset-specific confounds. **Fake side:** [VideoPhy-2](https://github.com/Hritikbansal/videophy) low-physics-score generations + [GenVideo](https://arxiv.org/abs/2405.19707) samples.
2. **Re-encode everything identically** (same fps, resolution, codec) — otherwise thresholds learn compression, not physics.
3. Run the pipeline sans LLM; log per-clip: D3 score, morph signals, consistency-test statistics.
4. ROC per signal + logistic combination; **bootstrap confidence intervals** (n = 40/40 alone gives AUC ± ~0.10 — insufficient to freeze), cross-dataset held-out check.
5. Freeze thresholds + percentiles into the verdict prompt ("this D3 score occurs in <5% of real clips"). Budget ≈ $10 of GPU.

## 5. Current operational state (accounting)

- All API keys revoked 2026-07-14. `/analyze` remains fully live (weights cached in Modal volumes, no external APIs). Claude verdict falls back to deterministic copy; `/hero` 503s; Poly Pizza returns null — all graceful paths.
- Repo clean at `622b209`. No v2 code exists.
- To restore full function: `uv run modal secret create kepler-anthropic ANTHROPIC_API_KEY=<new> [POLYPIZZA_API_KEY=..] [TRIPO_API_KEY=..]`
- D3 integration, image-plane physics, and calibration need **no external API keys** — only Modal deploy access.

## 6. Sources

Detection & benchmarks: [D3](https://arxiv.org/abs/2508.00701) · [D3 code](https://github.com/Zig-HS/D3) · [NSG physics-driven detection](https://arxiv.org/abs/2510.08073) · [VidGuard-R1](https://arxiv.org/abs/2510.02282) · [Physics-IQ](https://physics-iq.github.io/) · [VideoPhy](https://arxiv.org/abs/2406.03520) · [VideoPhy-2](https://arxiv.org/abs/2503.06800) · [PhyGenBench](https://arxiv.org/abs/2410.05363) · [PhysBench](https://arxiv.org/abs/2501.16411v2) · [GenVideo/DeMamba](https://arxiv.org/abs/2405.19707)

Geometry & tracking: [MoGe-2](https://github.com/microsoft/MoGe) · [Depth Pro](https://github.com/apple/ml-depth-pro) · [VGGT](https://github.com/facebookresearch/vggt) · [MegaSaM](https://arxiv.org/abs/2412.04463) · [Where Is The Ball](https://arxiv.org/abs/2506.05763) · [SpatialTrackerV2](https://spatialtracker.github.io/) · [SAM 2](https://arxiv.org/html/2408.00714v2) · [ProxyPose](https://arxiv.org/abs/2607.06555)

Physical reasoning lineage: [Galileo (NeurIPS 2015)](https://proceedings.neurips.cc/paper/2015/hash/d09bf41544a3365a46c9077ebb5e35c3-Abstract.html) · [ADEPT](http://physadept.csail.mit.edu/) · [PLATO](https://deepmind.google/discover/blog/intuitive-physics-learning-in-a-deep-learning-model-inspired-by-developmental-psychology/) · [Physion](https://arxiv.org/abs/2106.08261)
