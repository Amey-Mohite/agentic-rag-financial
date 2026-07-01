"""Upload this whole project to a Hugging Face Space (folders and all), safely.

Usage:
    pip install huggingface_hub
    python scripts/deploy_hf.py --repo <your-username>/<space-name> --token hf_xxx

- Preserves the folder structure (src/, web/, k8s/, ...).
- EXCLUDES secrets and local junk (.env, .git, venv, caches, vector storage, sample data) via
  ignore_patterns — so your keys never leave your machine.
- Re-run it any time to push updates; the Space rebuilds automatically.

Get a WRITE token at: https://huggingface.co/settings/tokens
Create the Space first (New → Space → SDK: Docker) so <username>/<space-name> exists.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. amey/agentic-rag")
    ap.add_argument("--token", required=True, help="HF write token (hf_...)")
    ap.add_argument("--path", default=".", help="folder to upload (default: project root)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    api.upload_folder(
        folder_path=args.path,
        repo_id=args.repo,
        repo_type="space",                 # it's a Space, not a model/dataset
        commit_message="deploy app",
        # NEVER upload these — secrets + local/build artifacts:
        ignore_patterns=[
            ".env", ".env.*", ".git/*", ".git", ".venv/*", "venv/*",
            "**/__pycache__/*", "*.pyc", ".pytest_cache/*", ".ruff_cache/*",
            "qdrant_storage/*", "qdrant_data/*",
            "data/*.pdf", "data/*.htm", "data/*.html",
            "*.ipynb_checkpoints/*",
        ],
    )
    print(f"Done → https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
