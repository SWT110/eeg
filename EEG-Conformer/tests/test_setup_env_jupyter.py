from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "setup_env_jupyter.py"


def load_module():
    spec = importlib.util.spec_from_file_location("setup_env_jupyter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SetupEnvJupyterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="setup-env-jupyter-")).resolve()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_build_commands_includes_core_setup_steps(self) -> None:
        config = self.module.SetupConfig(
            project_root=self.temp_dir,
            conda_executable="conda",
            env_name="eegconformer310",
            env_prefix=self.temp_dir / ".conda-envs" / "eegconformer310",
            pkgs_dir=self.temp_dir / ".conda-pkgs",
            python_version="3.10",
            pip_bootstrap_packages=["pip", "setuptools", "wheel"],
            pytorch_pip_packages=[
                "torch==1.12.1+cu113",
                "torchvision==0.13.1+cu113",
                "torchaudio==0.12.1",
            ],
            pytorch_index_url="https://download.pytorch.org/whl/cu113",
            pip_packages=[
                "numpy<2",
                "pandas",
                "scikit-learn",
                "einops",
                "torchsummary",
                "matplotlib",
                "pillow",
                "ipykernel",
            ],
            kernel_name="eegconformer310",
            kernel_display_name="Python (eegconformer310)",
        )

        commands = self.module.build_commands(config)

        self.assertEqual(commands[0].title, "Check NVIDIA driver")
        self.assertEqual(commands[0].command, "nvidia-smi")
        command_text = "\n".join(item.command for item in commands)
        self.assertIn("CONDA_PKGS_DIRS=", command_text)
        self.assertIn(str(self.temp_dir / ".conda-pkgs"), command_text)
        self.assertIn(str(self.temp_dir / ".conda-envs" / "eegconformer310"), command_text)
        self.assertIn("conda create --prefix", command_text)
        self.assertIn("python=3.10 -y", command_text)
        self.assertIn(
            "conda run --prefix",
            command_text,
        )
        self.assertIn("python -m pip install --upgrade pip setuptools wheel", command_text)
        self.assertIn("torch==1.12.1+cu113", command_text)
        self.assertIn("https://download.pytorch.org/whl/cu113", command_text)
        self.assertIn("numpy<2", command_text)
        self.assertIn(
            '--display-name "Python (eegconformer310)"',
            command_text,
        )

    def test_parse_args_keeps_defaults(self) -> None:
        args = self.module.parse_args([])

        self.assertEqual(args.env_name, "eegconformer310")
        self.assertEqual(args.python_version, "3.10")
        self.assertEqual(args.kernel_name, "eegconformer310")
        self.assertEqual(args.kernel_display_name, "Python (eegconformer310)")

    def test_build_config_uses_project_local_prefix_paths(self) -> None:
        args = self.module.parse_args(
            [
                "--project-root",
                str(self.temp_dir),
                "--env-name",
                "gpu310",
            ]
        )

        config = self.module.build_config(args)

        self.assertEqual(config.project_root, self.temp_dir)
        self.assertEqual(config.env_prefix, self.temp_dir / ".conda-envs" / "gpu310")
        self.assertEqual(config.pkgs_dir, self.temp_dir / ".conda-pkgs")
        self.assertIn("numpy<2", config.pip_packages)


if __name__ == "__main__":
    unittest.main()
