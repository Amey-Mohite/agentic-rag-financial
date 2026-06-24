"""CLI entry point: build the search index from documents.

Usage:
    python scripts/ingest.py --config config.yaml --path 'data/*.pdf'

Loads the config, runs the ingest pipeline (extract → chunk → embed → store) over the files
matched by --path, and prints how many chunks were stored.
"""
import argparse
import sys, pathlib

# Put <project>/src on the import path so `import agentic_rag...` works when run directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from agentic_rag.config import AppConfig
from agentic_rag.ingest import ingest_paths


def main():
    """Parse args, load config, ingest the matched files, report the chunk count."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")  # path to YAML config
    # --path accepts a single file or a glob (e.g. 'data/*.pdf'); required.
    ap.add_argument("--path", required=True, help="file path or glob, e.g. 'data/*.pdf'")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)   # load + env-expand the config
    n = ingest_paths(cfg, [args.path])       # run the full ingest pipeline
    print(f"ingested {n} chunks")            # report the result


if __name__ == "__main__":
    main()
