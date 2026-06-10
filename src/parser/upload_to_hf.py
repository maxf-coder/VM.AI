"""
VM-AI - HuggingFace Model Uploader
Uploads trained model folders to Hugging Face Hub.
Usage: python upload_to_hf.py [--message "commit msg"]

Written by: Vanea
"""

import argparse
import os
from getpass import getpass

from huggingface_hub import CommitOperationAdd, HfApi, login

HF_USERNAME = "vaneaa"
REPO_NAME = "vmai-parser"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "finetuned_parser")
REGRESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "regressors")

IGNORE_PREFIXES = ["checkpoint-", ".cache"]


def collect_files(folder_path, prefix_in_repo):
    """Walk folder_path and return CommitOperationAdd list, skipping IGNORE_PREFIXES."""
    ops = []
    for root, dirs, filenames in os.walk(folder_path):
        dirs[:] = [d for d in dirs if not any(d.startswith(p) for p in IGNORE_PREFIXES)]
        for fn in filenames:
            if any(fn.startswith(p) for p in IGNORE_PREFIXES):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, folder_path)
            path_in_repo = f"{prefix_in_repo}/{rel}".replace("\\", "/")
            ops.append(
                CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=full)
            )
    return ops


def check_folder(path, label):
    if not os.path.isdir(path):
        print(f"  Error: {label} folder not found at {path}")
        return False
    files = collect_files(path, "")
    if not files:
        print(f"  Error: {label} folder is empty (or all files filtered)")
        return False
    return True


def main():
    argp = argparse.ArgumentParser(description="Upload VM.AI parser to HuggingFace Hub")
    argp.add_argument(
        "--message", default="Upload VM.AI parser model", help="Commit message"
    )
    args = argp.parse_args()

    print("=" * 60)
    print("VM.AI Parser - Hugging Face Upload")
    print("=" * 60)
    print()

    token = getpass("Enter HuggingFace token (hidden): ").strip()
    if not token:
        print("Error: No token provided")
        print("Get one at: https://huggingface.co/settings/tokens")
        return

    print()
    print("Checking folders...")

    all_ops = []
    if check_folder(MODEL_PATH, "finetuned_parser"):
        parser_files = collect_files(MODEL_PATH, "finetuned_parser")
        all_ops.extend(parser_files)
        print(f"  finetuned_parser {len(parser_files)} files")
    else:
        print("  finetuned_parser (skipped - folder not found)")

    if check_folder(REGRESSOR_PATH, "regressors"):
        regressor_files = collect_files(REGRESSOR_PATH, "regressors")
        all_ops.extend(regressor_files)
        print(f"  regressors       {len(regressor_files)} files")
    else:
        print("  regressors (skipped - folder not found)")

    if not all_ops:
        print("\nNothing to upload.")
        return

    print()
    print(f"  Repo: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
    print(f"  Commit: {args.message}")
    print()

    print("Logging in...")
    try:
        login(token=token)
        print("  OK")
    except Exception as e:
        print(f"  Failed: {e}")
        return

    api = HfApi()
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    try:
        api.model_info(repo_id=repo_id)
        print("Repository exists")
    except Exception:
        print("Creating repository...")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)
        print(f"  Created: {repo_id}")

    print()
    print(f"Uploading {len(all_ops)} files in one commit...")
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=all_ops,
            commit_message=args.message,
            token=token,
        )
        print()
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print()
        print(f"https://huggingface.co/{repo_id}")
    except Exception as e:
        print()
        print(f"Upload failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check token has WRITE permission")
        print("  2. Make sure repo exists on huggingface.co")
        print("  3. Check internet connection")


if __name__ == "__main__":
    main()
