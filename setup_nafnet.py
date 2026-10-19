import subprocess
import sys
from pathlib import Path


repo_path = Path("/content/NAFNet")
if not repo_path.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/megvii-research/NAFNet.git",
            str(repo_path),
        ],
        check=True,
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "numpy<2", "gdown"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    cwd=repo_path,
    check=True,
)

weights_path = repo_path / "experiments/pretrained_models/NAFNet-GoPro-width64.pth"
weights_path.parent.mkdir(parents=True, exist_ok=True)
if not weights_path.exists():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "https://drive.google.com/uc?id=1S0PVRbyTakYY9a82kujgZLbMihfNBLfC",
            "-O",
            str(weights_path),
        ],
        check=True,
    )

if weights_path.stat().st_size < 1_000_000:
    raise RuntimeError(
        "The pretrained checkpoint download looks incomplete. Rerun this setup script."
    )

print("NAFNet motion-deblurring model is ready.")
