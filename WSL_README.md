# WSL + NVIDIA GPU Guide

This guide documents a working setup for running `vanity` on Windows 10 + WSL2 with NVIDIA GPU acceleration.

## 1) Prerequisites

- Windows 10 with WSL2 enabled
- NVIDIA Windows driver that supports CUDA in WSL2
- Ubuntu (or similar) WSL distro

Inside WSL, verify GPU visibility:

```bash
/usr/lib/wsl/lib/nvidia-smi || nvidia-smi
```

If this fails, update your Windows NVIDIA driver and run `wsl --update` from Windows.

## 2) Install Toolchain in WSL

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config curl git ca-certificates
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

Install CUDA toolkit in WSL:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit
```

Set environment:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc
```

## 3) Build Notes (Important)

- Cargo only uses `build.rs` as the build-script entrypoint.
- This repo keeps native GPU logic in `build_native.rs` and WSL-focused logic in `build_wsl.rs`.
- `build.rs` auto-detects WSL and picks the right builder.
- You can force selection with `VANITY_BUILD_IMPL`:
  - `VANITY_BUILD_IMPL=wsl`
  - `VANITY_BUILD_IMPL=native`
- Set `VANITY_CUDA_ARCH` for your GPU architecture.

Examples:

- RTX 30xx: `VANITY_CUDA_ARCH=86`
- RTX 40xx: `VANITY_CUDA_ARCH=89`

You can pass a comma-separated list if needed:

```bash
VANITY_CUDA_ARCH=86,89
```

Build:

```bash
VANITY_BUILD_IMPL=wsl VANITY_CUDA_ARCH=86 NVCC=/usr/local/cuda/bin/nvcc cargo build --release --features gpu
```

## 4) Usage

Main commands:

```bash
./target/release/vanity --help
./target/release/vanity grind --help
./target/release/vanity grind-keypair --help
./target/release/vanity verify --help
```

### Grind a mint-compatible keypair vanity prefix

Use `grind-keypair` (not `grind`) when you need an actual Solana keypair (for example a mint keypair):

```bash
./target/release/vanity grind-keypair --prefix DeFixxx --num-gpus 1 --num-cpus 1 --count 1
```

Output is printed to terminal and includes:

- `pubkey: ...`
- `keypair json (solana-compatible): [...]`

Save the JSON array to a file such as `mint-keypair.json` for Solana tooling.

## 5) Troubleshooting

- `nvcc: command not found`
  - Ensure CUDA toolkit is installed and `/usr/local/cuda/bin` is in `PATH`.

- `CUDA driver version is insufficient for CUDA runtime version`
  - Update Windows NVIDIA driver or install a CUDA toolkit version compatible with your current driver.

- Link errors mentioning `__cxa_guard_*` or `__gxx_personality_v0`
  - Ensure WSL builder is selected (`VANITY_BUILD_IMPL=wsl`) and `--features gpu` is enabled.

- `gpu_keypair_init(...): cudaGetDeviceProperties ...`
  - Verify `nvidia-smi` works inside WSL and GPU access is not blocked by policy.
