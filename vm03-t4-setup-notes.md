# vm-03 (GCP Tesla T4) — Inference Setup Notes
Date: 2026-07-07/08 · Project: `pp-pg-test-services-three` · Zone: `us-central1-b` · VM: `vm-03`

## Final working stack
- **GPU:** Tesla T4 (TU104, Turing, compute cap 7.5), 40 SMs × 64 cores = 2,560 CUDA cores, 320 Tensor cores, 15GB GDDR6, 1590 MHz SM / 5001 MHz mem
- **Driver:** 535.261.03 (DKMS-built kernel modules: nvidia, nvidia-uvm, nvidia-modeset, nvidia-drm, nvidia-peermem)
- **CUDA:** 12.2 (driver-side max). No standalone CUDA toolkit installed — PyTorch ships bundled CUDA runtime.
- **Python:** 3.11.2 (system) + venv at `/home/akandrapu/100-days-of-inference/.venv`
- **PyTorch:** 2.5.1+cu121, verified `torch.cuda.is_available() == True`, matmul on T4 OK
- **Kernel:** `inference100` (registered for user `akandrapu`)
- **Deps in venv:** numpy, matplotlib, ipykernel, ipywidgets, transformers, jupyter, nvtop (system)

## Repo
`/home/akandrapu/100-days-of-inference` — day01..day26 + "Inference Engineering.pdf".
day01 uses GPT-2 (124M) — small enough for CPU/T4. ~20 min.

## Pitfalls hit (so you don't repeat them)
1. **Debian-12 sources default to `main` only.** nvidia-driver needs `non-free-firmware non-free`; `nvidia-installer-cleanup` lives in `contrib`. Edit `/etc/apt/sources.list.d/debian.sources`:
   ```
   Components: main contrib non-free-firmware non-free
   ```
   (both stanzas — bookworm and bookworm-security). Then `apt-get update`.
2. **Always run apt installs non-interactive** or `dpkg-preconfigure` hangs forever on a pipe read with no TTY:
   ```
   DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::=--force-confold ...
   ```
3. **Never mix the Debian nvidia repo with the NVIDIA CUDA apt repo.** They conflict over `libcuda1` / `nvidia-opencl-icd` → "held broken packages". Pick ONE source for nvidia packages.
4. **CUDA repo dropped 12.2; driver 535 maxes at 12.2.** Resolution: skip the standalone toolkit. Install PyTorch with bundled CUDA instead:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   The toolkit (`nvcc`) is only needed for Day 7+ (writing/compiling `.cu` kernels) and Nsight profiling (Days 6/19). Days 1–6 + most serving frameworks (vLLM, SGLang prebuilt wheels) run fine on PyTorch's prebuilt CUDA.
5. **DKMS build is slow on 1 vCPU** (~5–10 min, compiles 5 modules). Don't interrupt it — orphaned child `make`/`cc1` processes keep the dpkg lock and look "stuck" but are actually compiling. Check with `pgrep -af 'dkms|cc1'` — many processes = working, not hung.
6. **Kernel registered as root went to `/root/.local/...`; VS Code runs as `akandrapu` and didn't see it.** Re-register as the user:
   ```
   sudo -u akandrapu bash -lc 'cd /home/akandrapu/100-days-of-inference && .venv/bin/python -m ipykernel install --user --name inference100 --display-name "Python (inference100)"'
   ```

## GPU visibility (no toolkit needed)
- `nvidia-smi` — aggregate VRAM / util % / temp / per-process
- `nvidia-smi dmon -s u` — SM util over time
- `nvtop` (installed) — htop-style live per-process GPU view
- `torch.profiler` — per-op GPU kernel timing (works with bundled CUDA)
- **Per-SM / warp-level metrics need Nsight Compute (`ncu`) = CUDA toolkit** (Day 7+)

## T4 limits (vs repo's DGX Spark / Hopper-Blackwell)
- No FP8 (Turing). Days 13 (FP8 quant) won't run on T4.
- 16GB VRAM caps model size.
- Day 7+ custom CUDA kernels: doable but no Hopper features (TMA, etc.).
- **Days 1–5 run fine on T4.** Upgrade hardware only when a specific day needs it.

## Run a notebook
VS Code Remote-SSH (already set up — `.vscode-server` on vm-03):
1. Open folder `/home/akandrapu/100-days-of-inference`
2. Open `day01/llm-inference-mechanics.ipynb`
3. Kernel → `Python (inference100)` (or `.venv/bin/python`)
4. Run All — first run downloads GPT-2 (~500MB, one-time)

Or headless:
```
cd /home/akandrapu/100-days-of-inference
.venv/bin/jupyter nbconvert --to notebook --execute day01/llm-inference-mechanics.ipynb
```
