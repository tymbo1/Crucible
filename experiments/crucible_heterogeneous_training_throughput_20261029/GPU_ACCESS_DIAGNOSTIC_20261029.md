# GPU access diagnostic — 2026-10-29

Status: **blocked; no GPU benchmark executed**.

The RTX-class PCI device is visible and the NVIDIA kernel modules are loaded
and bound, but this Codex process is running under a managed sandbox. The
`/dev/nvidia*` device nodes are absent from that restricted namespace. As a
result, `nvidia-smi` cannot communicate with the driver and PyTorch reports
CUDA unavailable here. This does not establish that the host driver is broken
outside the sandbox.

No Docker/Podman container markers were detected, but the managed sandbox is
confirmed by the process environment. No driver reinstall, module reload,
service restart or device-node creation was attempted.

The unchanged four-way benchmark can run after an independently approved
host execution or an approved sandbox GPU-device mapping restores CUDA device
access. See the machine-readable record in
`GPU_ACCESS_DIAGNOSTIC_20261029.json`.
