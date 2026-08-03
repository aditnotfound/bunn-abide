# AWS Compute Environment

This record identifies the verified compute environment for the ABIDE-I study.
It deliberately excludes IP addresses, SSH keys, credentials, and other
access details.

## Verified on 2026-08-03

- Instance hardware: one NVIDIA A10G (23,028 MiB reported by `nvidia-smi`).
- Operating system: Ubuntu 26.04 LTS.
- Booted kernel: `7.0.0-1009-aws`.
- NVIDIA driver: `595.84`, installed with the matching Ubuntu AWS kernel
  module package (`nvidia-driver-595-open` and
  `linux-modules-nvidia-595-open-aws`).
- Project environment: `~/bunn-abide/.venv`.
- Python: 3.14.
- PyTorch: `2.11.0+cu128`; CUDA runtime reported by PyTorch: 12.8.
- NumPy: 2.5.1.

## Health checks passed

- `nvidia-smi` identifies the A10G after reboot.
- `torch.cuda.is_available()` returns `True`.
- A 2048-by-2048 GPU matrix multiplication completed successfully.
- A GPU sum-of-squares test completed successfully with NumPy import enabled.

The public ABIDE-I AAL time-series files are stored below
`~/bunn-abide/data/`. They are not tracked in Git. Before any experiment,
activate the environment with:

```bash
source ~/bunn-abide/.venv/bin/activate
```
