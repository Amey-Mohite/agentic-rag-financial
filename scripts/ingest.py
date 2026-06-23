"""CLI: build the index from documents. Usage: python scripts/ingest.py --config config.yaml --path 'data/*.pdf'"""
import argparse
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from agentic_rag.config import AppConfig
from agentic_rag.ingest import ingest_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--path", required=True, help="file path or glob, e.g. 'data/*.pdf'")
    args = ap.parse_args()
    cfg = AppConfig.from_yaml(args.config)
    n = ingest_paths(cfg, [args.path])
    print(f"ingested {n} chunks")

if __name__ == "__main__":
    main()
