from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import NamedTuple


class SetupStep(NamedTuple):
    title: str
    command: str


class SetupConfig(NamedTuple):
    project_root: Path
    conda_executable: str
    env_name: str
    env_prefix: Path
    pkgs_dir: Path
    python_version: str
    pip_bootstrap_packages: list[str]
    pytorch_pip_packages: list[str]
    pytorch_index_url: str
    pip_packages: list[str]
    kernel_name: str
    kernel_display_name: str


def build_commands(config: SetupConfig) -> list[SetupStep]:
    pip_bootstrap_packages = " ".join(config.pip_bootstrap_packages)
    pytorch_pip_packages = " ".join(config.pytorch_pip_packages)
    pip_packages = " ".join(config.pip_packages)
    env_prefix = shlex.quote(str(config.env_prefix))
    pkgs_dir = shlex.quote(str(config.pkgs_dir))
    conda_prefix = f"CONDA_PKGS_DIRS={pkgs_dir} {config.conda_executable}"
    conda_run = f"{conda_prefix} run --prefix {env_prefix}"
    pytorch_command = (
        f"{conda_run} python -m pip install {pytorch_pip_packages} "
        f"--extra-index-url {config.pytorch_index_url}"
    )
    return [
        SetupStep("Check NVIDIA driver", "nvidia-smi"),
        SetupStep(
            "Create local package cache directory",
            f"mkdir -p {pkgs_dir}",
        ),
        SetupStep(
            "Create local conda env directory",
            f"mkdir -p {shlex.quote(str(config.project_root / '.conda-envs'))}",
        ),
        SetupStep(
            "Create project-local conda environment",
            f"{conda_prefix} create --prefix {env_prefix} python={config.python_version} -y",
        ),
        SetupStep(
            "Upgrade pip tooling",
            f"{conda_run} python -m pip install --upgrade {pip_bootstrap_packages}",
        ),
        SetupStep("Install PyTorch", pytorch_command),
        SetupStep(
            "Install Python packages",
            f"{conda_run} python -m pip install {pip_packages}",
        ),
        SetupStep(
            "Register Jupyter kernel",
            (
                f'{conda_run} python -m ipykernel install --user '
                f'--name {config.kernel_name} '
                f'--display-name "{config.kernel_display_name}"'
            ),
        ),
        SetupStep(
            "Validate PyTorch CUDA",
            (
                f"{conda_run} python -c "
                + shlex.quote(
                    "import torch; "
                    "print('torch:', torch.__version__); "
                    "print('cuda in torch:', torch.version.cuda); "
                    "print('cuda available:', torch.cuda.is_available()); "
                    "print('device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); "
                    "print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
                )
            ),
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Python 3.10 + CUDA 11.3 compatible EEG-Conformer environment from Jupyter or shell"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="EEG-Conformer project root where the local conda env and package cache will be created",
    )
    parser.add_argument("--conda-executable", default=os.environ.get("CONDA_EXE", "conda"))
    parser.add_argument("--env-name", default="eegconformer310")
    parser.add_argument("--python-version", default="3.10")
    parser.add_argument("--kernel-name", default="eegconformer310")
    parser.add_argument("--kernel-display-name", default="Python (eegconformer310)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SetupConfig:
    env_name = args.env_name
    conda_executable = args.conda_executable
    project_root = Path(args.project_root).expanduser().resolve()
    pip_packages = [
        "numpy<2",
        "pandas",
        "scikit-learn",
        "einops",
        "torchsummary",
        "matplotlib",
        "pillow",
        "ipykernel",
    ]
    pip_bootstrap_packages = [
        "pip",
        "setuptools",
        "wheel",
    ]
    pytorch_pip_packages = [
        "torch==1.12.1+cu113",
        "torchvision==0.13.1+cu113",
        "torchaudio==0.12.1",
    ]
    return SetupConfig(
        project_root=project_root,
        conda_executable=conda_executable,
        env_name=env_name,
        env_prefix=project_root / ".conda-envs" / env_name,
        pkgs_dir=project_root / ".conda-pkgs",
        python_version=args.python_version,
        pip_bootstrap_packages=pip_bootstrap_packages,
        pytorch_pip_packages=pytorch_pip_packages,
        pytorch_index_url="https://download.pytorch.org/whl/cu113",
        pip_packages=pip_packages,
        kernel_name=args.kernel_name,
        kernel_display_name=args.kernel_display_name,
    )


def run_steps(steps: list[SetupStep], dry_run: bool) -> None:
    for index, step in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {step.title}")
        print(step.command)
        if dry_run:
            continue
        subprocess.run(step.command, shell=True, check=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    steps = build_commands(config)
    run_steps(steps, dry_run=args.dry_run)

    print("\nDone.")
    print(f"Project root: {config.project_root}")
    print(f"Conda env prefix: {config.env_prefix}")
    print(f"Conda package cache: {config.pkgs_dir}")
    print(f"Jupyter kernel: {config.kernel_display_name}")
    print("\nIf validation prints 'cuda available: True', switch your notebook kernel to the new kernel before training.")


if __name__ == "__main__":
    main()
