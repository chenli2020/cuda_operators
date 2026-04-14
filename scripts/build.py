#!/usr/bin/env python3
"""Build script for CUDA operators."""
import os
import subprocess
import sys
import argparse


def run_command(cmd, cwd=None):
    """Run shell command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build CUDA operators")
    parser.add_argument(
        "--cuda-arch",
        default="80",
        help="CUDA architecture (e.g., 80 for Ampere, 86 for RTX 30xx)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building",
    )
    parser.add_argument(
        "--pip",
        action="store_true",
        help="Build and install with pip",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, "build")

    # Clean if requested
    if args.clean and os.path.exists(build_dir):
        print("Cleaning build directory...")
        import shutil
        shutil.rmtree(build_dir)

    if args.pip:
        # Build with pip
        run_command("pip install -e .", cwd=project_root)
    else:
        # Create build directory
        os.makedirs(build_dir, exist_ok=True)

        # Configure with CMake
        cmake_cmd = (
            f'cmake .. -DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch} '
            f'-DCMAKE_BUILD_TYPE=Release'
        )
        run_command(cmake_cmd, cwd=build_dir)

        # Build
        if sys.platform == "win32":
            run_command("cmake --build . --config Release", cwd=build_dir)
        else:
            run_command("make -j", cwd=build_dir)

    print("Build completed successfully!")


if __name__ == "__main__":
    main()
