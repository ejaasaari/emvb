import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = Path().absolute()
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DFAISS_ENABLE_GPU=OFF",
            "-DFAISS_ENABLE_PYTHON=OFF",
        ]

        build_args = ["--config", "Release", "--", "-j"]

        os.chdir(str(build_temp))
        subprocess.check_call(["cmake", str(cwd)] + cmake_args)
        subprocess.check_call(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))


setup(
    name="emvb",
    version="0.1.0",
    author="EMVB Team",
    description="Efficient Multi-Vector Retrieval with Bit Vectors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("emvb")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    packages=["emvb"],
    package_dir={"emvb": "python/emvb"},
)
