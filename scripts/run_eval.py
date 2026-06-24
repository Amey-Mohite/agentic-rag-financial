"""CLI entry point: run an evaluation pass over a labeled question set.

Usage:
    python scripts/run_eval.py --config config.yaml --eval-set data/eval_set.jsonl

The eval set is JSONL — one JSON object per line, each like:
    {"question": "...", "ground_truth": "..."}

CONCEPT: WHY EVALUATE A RAG SYSTEM?
-----------------------------------
"It looks right" doesn't scale. To improve a RAG system you measure it on a fixed question
set so you can compare configs (chunking strategy, hybrid on/off, reranker, etc.). This is a
thin harness: it runs every question through the agent and collects
(question, answer, retrieved contexts, ground_truth). The commented-out block shows where you
plug in Ragas metrics (faithfulness, answer relevancy, context precision/recall) to score them.
"""
import argparse, json
import sys, pathlib

# Put <project>/src on the import path so `import agentic_rag...` works when run directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from agentic_rag.agent import AgenticRAG


def main():
    """Parse args, run every eval question through the agent, collect + report results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")  # path to YAML config
    ap.add_argument("--eval-set", required=True)        # path to the JSONL eval set
    args = ap.parse_args()
    rag = AgenticRAG.from_config(args.config)           # build the pipeline once

    # Read the JSONL file: parse each non-blank line into a dict. `if l.strip()` skips blanks.
    rows = [json.loads(l) for l in open(args.eval_set) if l.strip()]
    records = []
    for r in rows:
        ans = rag.answer(r["question"])  # run the agent on this question
        # Collect everything an eval framework needs to score this row.
        records.append({
            "question": r["question"],
            "answer": ans.text,
            "ground_truth": r.get("ground_truth", ""),  # may be absent → default ""
            "n_citations": len(ans.citations),
        })
        # Progress line: first 60 chars of the question + how many citations it produced.
        print(f"[done] {r['question'][:60]}... ({len(ans.citations)} citations)")

    # --- Ragas scoring (uncomment once you have ragas installed + an eval set) ---
    # from ragas import evaluate
    # from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    # ... build a ragas Dataset from `records` (including the retrieved contexts) and call evaluate(...)
    print(f"\nprocessed {len(records)} questions. Add Ragas scoring to publish the metrics table.")


if __name__ == "__main__":
    main()
