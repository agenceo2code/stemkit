# o2code Intel Mac + AMD variant

This branch adds Intel macOS support while keeping `main` clean so the fork can continue to sync with `danielravina/stemkit`.

## Branch strategy

- `main`: mirror the upstream fork. Do not add o2code-specific commits here.
- `o2code/intel-amd`: Intel/x86_64 + AMD MPS compatibility layer.

## Updating from upstream

First update `main` with GitHub's **Sync fork** button, or locally:

```bash
git remote add upstream https://github.com/danielravina/stemkit.git
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main
```

Then bring the new upstream code into this branch:

```bash
git checkout o2code/intel-amd
git merge main
git push origin o2code/intel-amd
```

Resolve conflicts only in the small Intel-specific surface when upstream changes the same files.

## Intel runtime choices

- macOS x86_64 uses PyTorch / torchaudio 2.2.2, the final PyTorch line supporting macOS x64.
- MPS is auto-detected. On an Intel Mac with a compatible AMD GPU, separation attempts MPS first.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` lets unsupported Metal operators fall back to CPU.
- RoFormer runs MPS in fp32 on Intel for stability.
- Both Demucs and RoFormer retain their existing CPU fallback if MPS fails.
- FFmpeg + libsoxr are built natively as x86_64 on an Intel Mac.
- Release and smoke workflows use GitHub's `macos-26-intel` runner.

## Build locally on the Intel Mac

```bash
npm install
bash scripts/fetch-ffmpeg.sh
npm run dev
```

For the Intel DMG/ZIP:

```bash
npm run dist
```

## Verify AMD MPS after first engine setup

Run with the app's private Python environment or an equivalent Python 3.10/3.11 x86_64 environment:

```python
import platform
import torch

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    x = torch.randn(2048, 2048, device="mps")
    y = x @ x
    print("compute device:", y.device)
```

Expected on the target Mac is `x86_64`, PyTorch `2.2.2`, and ideally `mps available: True`.
