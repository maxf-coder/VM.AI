"""
VM-AI - HuggingFace Model Downloader
Downloads both T5 parser and regressor models from Hugging Face.
Usage: python pull_from_hf.py [token]
ALWAYS backs up existing models before downloading

Written by: Vanea
"""

import os
import shutil
import sys
import tempfile

from huggingface_hub import snapshot_download

HF_USERNAME = "vaneaa"
REPO_NAME = "vmai-parser"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

PARSER_PATH = os.path.join(PROJECT_ROOT, "models", "finetuned_parser")
PARSER_BACKUP_PATH = os.path.join(PROJECT_ROOT, "models", "finetuned_parser_backup")

REGRESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "regressors")
REGRESSOR_BACKUP_PATH = os.path.join(PROJECT_ROOT, "models", "regressors_backup")


def backup_existing(path, backup_path, label):
    """Backup a model directory if it exists."""
    if not os.path.exists(path):
        print(f"  No existing {label} to backup")
        return False

    print(f"  Backing up {label}...")
    if os.path.exists(backup_path):
        print(f"    Removing old backup...")
        shutil.rmtree(backup_path)
    shutil.move(path, backup_path)
    print(f"    Backed up to: {os.path.basename(backup_path)}")
    return True


def main():
    print("=" * 60)
    print("VM.AI Parser - Download from Hugging Face")
    print("=" * 60)
    print()

    token = sys.argv[1].strip() if len(sys.argv) > 1 else None

    if token:
        print("Using provided token")
    else:
        print("No token provided - will attempt public repo download")
    print()

    # Backup both existing models
    print("Backing up existing models...")
    backup_existing(PARSER_PATH, PARSER_BACKUP_PATH, "T5 parser")
    backup_existing(REGRESSOR_PATH, REGRESSOR_BACKUP_PATH, "regressors")
    print()

    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    print(f"Repository: https://huggingface.co/{repo_id}")
    print()

    # Download to a temp directory to flatten the structure
    tmp_dir = tempfile.mkdtemp(prefix="vmai_hf_")
    print(f"Downloading to temporary directory...")

    try:
        download_kwargs = {"repo_id": repo_id, "local_dir": tmp_dir}
        if token:
            download_kwargs["token"] = token

        snapshot_download(**download_kwargs)

        # Check what subdirs we got
        items = os.listdir(tmp_dir)
        print(f"  Downloaded items: {items}")

        # Move T5 parser model files
        src_parser = os.path.join(tmp_dir, "finetuned_parser")
        if os.path.isdir(src_parser):
            os.makedirs(PARSER_PATH, exist_ok=True)
            for item in os.listdir(src_parser):
                shutil.move(os.path.join(src_parser, item), os.path.join(PARSER_PATH, item))
            parser_files = os.listdir(PARSER_PATH)
            print(f"  T5 parser: {len(parser_files)} files → {PARSER_PATH}")
        else:
            print("  Warning: finetuned_parser/ not found in repo")

        # Move regressor model files
        src_regressor = os.path.join(tmp_dir, "regressors")
        if os.path.isdir(src_regressor):
            os.makedirs(REGRESSOR_PATH, exist_ok=True)
            for item in os.listdir(src_regressor):
                shutil.move(os.path.join(src_regressor, item), os.path.join(REGRESSOR_PATH, item))
            reg_files = os.listdir(REGRESSOR_PATH)
            print(f"  Regressors: {len(reg_files)} files → {REGRESSOR_PATH}")
        else:
            print("  Warning: regressors/ not found in repo")

        # Verify everything worked
        if not os.listdir(PARSER_PATH):
            raise Exception("Parser model directory is empty after download")

        if not os.listdir(REGRESSOR_PATH):
            raise Exception("Regressor directory is empty after download")

        print()
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print()
        print(f"  T5 parser: {PARSER_PATH}")
        print(f"  Regressors: {REGRESSOR_PATH}")
        print()
        print("Backups saved at:")
        print(f"  {PARSER_BACKUP_PATH}")
        print(f"  {REGRESSOR_BACKUP_PATH}")
        print()
        print("To restore:")
        print(f"  Replace {os.path.basename(PARSER_PATH)} with {os.path.basename(PARSER_BACKUP_PATH)}")
        print(f"  Replace {os.path.basename(REGRESSOR_PATH)} with {os.path.basename(REGRESSOR_BACKUP_PATH)}")

    except Exception as e:
        print()
        print(f"Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check token is valid")
        print("  2. Check repo exists: https://huggingface.co/vaneaa/vmai-parser")
        print("  3. Check internet connection")
        print()
        print("Restoring backups...")
        if os.path.exists(PARSER_BACKUP_PATH) and not os.path.exists(PARSER_PATH):
            shutil.move(PARSER_BACKUP_PATH, PARSER_PATH)
            print(f"  Restored {os.path.basename(PARSER_PATH)}")
        if os.path.exists(REGRESSOR_BACKUP_PATH) and not os.path.exists(REGRESSOR_PATH):
            shutil.move(REGRESSOR_BACKUP_PATH, REGRESSOR_PATH)
            print(f"  Restored {os.path.basename(REGRESSOR_PATH)}")

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            print()
            print("Cleaned up temporary files")


if __name__ == "__main__":
    main()
