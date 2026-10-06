"""Build a wheel from the UniFFI library and its generated Python binding."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class BuildPythonBindings(build_py):
    def run(self) -> None:
        crate_dir = Path(__file__).resolve().parent
        workspace = crate_dir.parents[1]
        profile = os.environ.get("TESSERA_WHEEL_PROFILE", "release")
        if profile not in {"debug", "release"}:
            raise ValueError("TESSERA_WHEEL_PROFILE must be debug or release")

        metadata = subprocess.check_output(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=workspace,
        )
        target = Path(json.loads(metadata)["target_directory"]) / profile
        command = ["cargo", "build", "--locked", "-p", "tessera-python3d"]
        if profile == "release":
            command.append("--release")
        subprocess.run(command, cwd=workspace, check=True)

        if sys.platform == "win32":
            library = target / "tessera3d.dll"
            bindgen = target / "tessera-uniffi-bindgen.exe"
        elif sys.platform == "darwin":
            library = target / "libtessera3d.dylib"
            bindgen = target / "tessera-uniffi-bindgen"
        else:
            library = target / "libtessera3d.so"
            bindgen = target / "tessera-uniffi-bindgen"

        super().run()
        package = Path(self.build_lib) / "tessera3d"
        package.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tessera-uniffi-") as temporary:
            subprocess.run(
                [
                    str(bindgen),
                    "generate",
                    "--library",
                    str(library),
                    "--language",
                    "python",
                    "--out-dir",
                    temporary,
                ],
                cwd=workspace,
                check=True,
            )
            shutil.copy2(Path(temporary) / "tessera3d.py", package / "tessera3d.py")
        shutil.copy2(library, package / library.name)


class PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"build_py": BuildPythonBindings, "bdist_wheel": PlatformWheel})
