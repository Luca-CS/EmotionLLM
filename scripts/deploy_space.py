"""Assemble the Hugging Face Space from this repo and (optionally) upload it.

    python scripts/deploy_space.py --stage build/space              # only assemble, nothing leaves the machine
    python scripts/deploy_space.py --push Luca-CS/EmotionLLM        # after `hf auth login` (or with HF_TOKEN set)

The write token is the one saved by `hf auth login` or the `HF_TOKEN` variable; it is never stored in the repo.
The Space gets the code, the weights and `space/README.md` as its README (it carries the Space metadata).
"""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "requirements.txt", "LICENSE"]
DIRS = ["emotionllm", "weights"]


def stage(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(ROOT / name, dest / name)
    for name in DIRS:
        shutil.copytree(ROOT / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "original"))
    shutil.copy2(ROOT / "space" / "README.md", dest / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path, default=ROOT / "build" / "space", help="where to assemble the Space")
    parser.add_argument("--push", metavar="USER/SPACE", help="create the Space if needed and upload the folder")
    args = parser.parse_args()

    stage(args.stage)
    print(f"Space assembled in {args.stage}")
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))  # None: the token saved by `hf auth login`
        api.create_repo(args.push, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=args.stage, repo_id=args.push, repo_type="space", commit_message="Update Space")
        print(f"Uploaded to https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
