

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/JusperLee/Dolphin.git"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "vendor" / "dolphin"

_SHARED_PATCHES = [
    (
        "global_f = torch.zeros(\n"
        "            output[-1].shape, requires_grad=True, device=output1.device\n"
        "        )",
        "global_f = torch.zeros(\n"
        "            output[-1].shape,\n"
        "            requires_grad=True,\n"
        "            device=output1.device,\n"
        "            dtype=output1.dtype,\n"
        "        )",
    ),
    (
        "fusion_x = torch.zeros([x.shape[0], x.shape[1], min_len], "
        "requires_grad=True, device=x.device)",
        "fusion_x = torch.zeros(\n"
        "            [x.shape[0], x.shape[1], min_len],\n"
        "            requires_grad=True,\n"
        "            device=x.device,\n"
        "            dtype=x.dtype,\n"
        "        )",
    ),
]

_DOLPHIN_ONLY_PATCHES = [
    (
        'if (T == getattr(self, "__RES__", 0)) and '
        '(getattr(self, "__WEIGHT_COSN__", None).device == x.device):\n'
        '            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)\n'
        '            weight_exp = getattr(self, "__WEIGHT_EXP__", None)\n'
        "            assert weight_cosn is not None\n"
        "            assert weight_exp is not None\n"
        "        else:\n"
        "            weight_cosn = self.get_cos_map(T, device=x.device).detach_()\n"
        "            weight_exp = self.get_decay_map(T, device=x.device).detach_()",
        "if (\n"
        '            (T == getattr(self, "__RES__", 0))\n'
        '            and (getattr(self, "__WEIGHT_COSN__", None) is not None)\n'
        '            and (getattr(self, "__WEIGHT_COSN__").device == x.device)\n'
        '            and (getattr(self, "__WEIGHT_COSN__").dtype == x.dtype)\n'
        "        ):\n"
        '            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)\n'
        '            weight_exp = getattr(self, "__WEIGHT_EXP__", None)\n'
        "            assert weight_cosn is not None\n"
        "            assert weight_exp is not None\n"
        "        else:\n"
        "            weight_cosn = self.get_cos_map(T, device=x.device, "
        "dtype=x.dtype).detach_()\n"
        "            weight_exp = self.get_decay_map(T, device=x.device, "
        "dtype=x.dtype).detach_()",
    ),
]

_DOLPHIN_LIGHT_ONLY_PATCHES = [
    (
        'if (T == getattr(self, "__RES__", 0)) and '
        '(getattr(self, "__WEIGHT_EXP__", None).device == x.device):\n'
        '            weight_exp = getattr(self, "__WEIGHT_EXP__", None)\n'
        "            assert weight_exp is not None\n"
        "        else:\n"
        "            weight_exp = self.get_decay_map(T, device=x.device).detach_()",
        "if (\n"
        '            (T == getattr(self, "__RES__", 0))\n'
        '            and (getattr(self, "__WEIGHT_EXP__", None) is not None)\n'
        '            and (getattr(self, "__WEIGHT_EXP__").device == x.device)\n'
        '            and (getattr(self, "__WEIGHT_EXP__").dtype == x.dtype)\n'
        "        ):\n"
        '            weight_exp = getattr(self, "__WEIGHT_EXP__", None)\n'
        "            assert weight_exp is not None\n"
        "        else:\n"
        "            weight_exp = self.get_decay_map(T, device=x.device, "
        "dtype=x.dtype).detach_()",
    ),
]

_NAN_STABILITY_PATCHES = [
    (
        "    def get_decay_map(resolution=224, device=torch.device(\"cpu\"), dtype=torch.float):\n"
        "        # exp(- (n π / T)^2) for 1D\n"
        "        # returns: (Res_t,)\n"
        "        res_t = resolution\n"
        "        weight_n = torch.linspace(0, torch.pi, res_t + 1, device=device, dtype=dtype)[:res_t]\n"
        "        weight = torch.pow(weight_n, 2)\n"
        "        weight = torch.exp(-weight)\n"
        "        return weight",
        "    def get_decay_map(resolution=224, device=torch.device(\"cpu\"), dtype=torch.float):\n"
        "        # exp(- (n π / T)^2) for 1D\n"
        "        # returns: (Res_t,)\n"
        "        # NOTE: underflows fp16 precision; the downstream torch.pow(...) by a\n"
        "        # learned exponent produces inf/NaN if computed directly in float16 —\n"
        "        # always compute in float32, cast to the requested dtype at the end.\n"
        "        res_t = resolution\n"
        "        weight_n = torch.linspace(0, torch.pi, res_t + 1, device=device, dtype=torch.float32)[:res_t]\n"
        "        weight = torch.pow(weight_n, 2)\n"
        "        weight = torch.exp(-weight)\n"
        "        return weight.to(dtype=dtype)",
    ),
]

_DOLPHIN_ONLY_NAN_STABILITY_PATCHES = [
    (
        "    def get_cos_map(N=224, device=torch.device(\"cpu\"), dtype=torch.float):\n"
        "        # cos((x + 0.5) / N * n * π) which is also the form of DCT and IDCT\n"
        "        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * π) * f(x) )\n"
        "        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * π) * F(n) )\n"
        "        # returns: (Res_n, Res_x)\n"
        "        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N\n"
        "        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)\n"
        "        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)\n"
        "        weight[0, :] = weight[0, :] / math.sqrt(2)\n"
        "        return weight",
        "    def get_cos_map(N=224, device=torch.device(\"cpu\"), dtype=torch.float):\n"
        "        # cos((x + 0.5) / N * n * π) which is also the form of DCT and IDCT\n"
        "        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * π) * f(x) )\n"
        "        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * π) * F(n) )\n"
        "        # returns: (Res_n, Res_x)\n"
        "        # NOTE: n*x reaches tens of thousands for real sequence lengths,\n"
        "        # overflowing fp16 trig precision and producing NaN if computed\n"
        "        # directly in float16 — always compute in float32, cast at the end.\n"
        "        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=torch.float32).view(1, -1) + 0.5) / N\n"
        "        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=torch.float32).view(-1, 1)\n"
        "        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)\n"
        "        weight[0, :] = weight[0, :] / math.sqrt(2)\n"
        "        return weight.to(dtype=dtype)",
    ),
]

_POW_WEIGHT_EXP_PATCHES = [
    (
        "        weight_exp = torch.pow(weight_exp[:, None], self.k)\n"
        '        x = torch.einsum("bnc,nc->bnc", x, weight_exp)  # exp decay',
        "        weight_exp = torch.pow(weight_exp[:, None].float(), self.k.float()).to(dtype=x.dtype)\n"
        '        x = torch.einsum("bnc,nc->bnc", x, weight_exp)  # exp decay',
    ),
]

_DOLPHIN_LIGHT_ONLY_POW_WEIGHT_EXP_PATCHES = [
    (
        "        weight_exp = torch.pow(weight_exp[None, None, :], self.k[None, :, None])  # [1, hidden_dim, T]\n"
        "        x = x * weight_exp  # [B, hidden_dim, T]",
        "        weight_exp = torch.pow(weight_exp[None, None, :].float(), self.k[None, :, None].float()).to(dtype=x.dtype)  # [1, hidden_dim, T]\n"
        "        x = x * weight_exp  # [B, hidden_dim, T]",
    ),
]

_DISABLE_FLASH_SDP_PATCHES = [
    (
        "        sdp_kwargs: dict = dict(\n"
        "            enable_flash = True,\n"
        "            enable_math = True,\n"
        "            enable_mem_efficient = True\n"
        "        )",
        "        sdp_kwargs: dict = dict(\n"
        "            enable_flash = False,\n"
        "            enable_math = True,\n"
        "            enable_mem_efficient = True\n"
        "        )",
    ),
]

def _apply_patches(path: Path, patches: list[tuple[str, str]]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    patched = text
    for old, new in patches:
        patched = patched.replace(old, new)
    if patched != text:
        path.write_text(patched, encoding="utf-8")
        print(f"Patched fp16 dtype bug in {path}")

def _patch_fp16_dtype_bug(dest: Path) -> None:
    _apply_patches(
        dest / "look2hear/models/dolphin.py",
        _SHARED_PATCHES
        + _DOLPHIN_ONLY_PATCHES
        + _NAN_STABILITY_PATCHES
        + _DOLPHIN_ONLY_NAN_STABILITY_PATCHES
        + _POW_WEIGHT_EXP_PATCHES,
    )
    _apply_patches(
        dest / "look2hear/models/dolphin_light.py",
        _SHARED_PATCHES
        + _DOLPHIN_LIGHT_ONLY_PATCHES
        + _NAN_STABILITY_PATCHES
        + _DOLPHIN_LIGHT_ONLY_POW_WEIGHT_EXP_PATCHES,
    )

def _patch_disable_flash_sdp(dest: Path) -> None:
    _apply_patches(
        dest / "look2hear/models/video_compoent.py",
        _DISABLE_FLASH_SDP_PATCHES,
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vendor Dolphin into vendor/dolphin")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--depth", type=int, default=1, help="shallow clone depth")
    args = parser.parse_args(argv)

    dest: Path = args.dest
    if (dest / "look2hear").is_dir():
        print(f"Dolphin already present at {dest}")
        _patch_fp16_dtype_bug(dest)
        _patch_disable_flash_sdp(dest)
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth",
        str(args.depth),
        REPO_URL,
        str(dest),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Cloned Dolphin to {dest}")
    _patch_fp16_dtype_bug(dest)
    _patch_disable_flash_sdp(dest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
