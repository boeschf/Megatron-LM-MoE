# Native gfx942 FNUZ transport qualification

Run `tests/functional_tests/qualify_fnuz_dispatch.py` from the manifest-verified, signed Megatron snapshot, using published image `910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490`.
The parent runner owns Slurm authorization, frozen manifests, launch timeouts and per-rank logs.
The script makes no jobs or cache publications.

| Probe argv | Placement | Check |
| --- | --- | --- |
| `["codec"]` | 1 gfx942 rank | Finite/NaN/Inf inputs, zero and tiny blocks, positive and negative signs, saturation/RTNE edges, strided BF16, empty rows, widths 1792 and 2048, and rejected formats. |
| `["ep16"]` | 4 nodes, 4 ranks/node, block placement | Real `ManualGradNode` and `MoEFlexTokenDispatcher` over installed UCCL/CXI: three deterministic 8192-source-token/top-8 dispatches; original handles used for both BF16 cached dispatch and BF16 reverse combine, with FP32 probability gradients. |

Use `RANK`, `LOCAL_RANK`, `WORLD_SIZE` and, for EP16, `MASTER_ADDR`, `MASTER_PORT`.
Use the pinned image network environment, `UCCL_EP_TRANSPORT=cxi`, `UCCL_SOCKET_IFNAME=hsn`, `UCCL_CXI_THREADING=safe`, `NCCL_P2P_DISABLE=0`, and no UCCL config overrides.
Expected EP16 effective configs at 20 SMs are dispatch `(20,36,288,20,140)` and combine `(20,4,288,12,132)`, independently of the historical EP8 override.
Normal dispatch/combine use the actual `Buffer` size hints and report the width-1792 allocation.

## Fixed fixtures and acceptance

- The codec is BF16 → native FNUZ (one 128-feature block per FP32 scale) → BF16, with signed finite saturation at ±240 and RNE.
- The reference uses an independent E4M3FNUZ code table (bias 8, subnormals `k * 2^-10`) and nearest-even ties, not a cast to Torch FP8.
- For finite inputs, compute `round_fp32(amax / 240)` with a `1e-30` minimum for nonzero BF16 blocks and scale `1` for both signs of all-zero blocks; normalize using an FP32 rounded reciprocal and multiplication.
- A block containing any NaN or Inf is poisoned with NaN scale and FNUZ NaN bytes; require NaN BF16 output but do not constrain its NaN payload.
- Compare finite scales by FP32 bits, all encoded UZ payload bytes directly, and all finite decoded BF16 bits.
- Each distributed source rank has 8192 rows, 16 local experts out of 256 and eight expert slots per row.
- Width 1792 uses 14 FP32 scales (56B), and width 1536 uses 12 (48B), exercising the two installed receive-copy branches without padding width 1792 or reallocating the EP buffer.
- The balanced fixture has eight distinct experts over four destination ranks; the ragged fixture includes variable active slots, repeated destination ranks, skew toward rank 0, a rank with zero routes out (14), a separate rank with zero routes in (15), and empty experts.
- An exact dyadic router probability tag identifies the source rank/token and expert slot, independently of activations and receive ordering.
- Regenerate expected incoming source/token sets, receiver-local expert indices and counts; reject duplicate, missing or misassociated rows, then check all payload bytes and scale bits before accepting BF16 decode.
- BF16 token signals and FP32 gradients at the synthetic expert boundary are dyadic, with exact reductions even with BF16 intermediate reductions in native combine.
- Require exact BF16 forward combine, cached BF16 dispatch backward, BF16 reverse combine and FP32 probability gradients (including dropped and nonlocal slots).
- Forward and combine case A, forward and combine different-layout case B using the same dispatcher, then run A's and B's backward; require different, live handles and the same communicator buffer.
- The synthetic expert boundary isolates transport; it does not qualify fused expert permutation, expert arithmetic, full model overlap or performance.

Run the codec probe first and require its success before EP16.
Stop at the first identity, shape, native config, timeout, collective, lifecycle or numerical failure.
Acceptance needs successful process teardown and exactly one `FNUZ_TRANSPORT_RESULT` with `status: pass` for **every** expected rank; a missing, malformed or failing record is a failure, regardless of other ranks.
Review that the EP16 logs contain native CXI initialization on all ranks.
Retain failure logs; do not change the precision or tolerance and retry blindly.

## Pre-GPU source evidence

Image Triton `3.8.0+git669b31ac.rocm10.1.0` maps `torch.float8_e4m3fnuz` → `fp8e4b8` (`tl.float8e4b8`) and keeps this dtype available on gfx942.
Offline compilation with Triton extracted from image 910 for `GPUTarget("hip", "gfx942", 64)` lowered to `v_cvt_pk_fp8_f32` and `v_cvt_pk_f32_fp8_e32`; this is not device execution.
The image's installed `deep_ep/buffer.py` (`d30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768`) calls `uccl.ep`, allocates receive tensors of `x.dtype`/`x_scales.dtype`, and retains event tensors and allocator stream ownership.
The installed UCCL is pinned at `52d4184c89deb5631be6dac6a9b1c40896c9e26e` with the two image-owned CXI/ROCm and Torch resource-ownership corrections; neither is modified by this change.
