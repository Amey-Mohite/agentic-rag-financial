"""CLI: run Ragas eval over a labeled JSONL set (one {"question","ground_truth"} per line).

Usage: python scripts/run_eval.py --config config.yaml --eval-set data/eval_set.jsonl

This is a thin harness: it runs each question through the agent, collects (question, answer,
retrieved contexts, ground_truth), then scores with Ragas. Wire your Langfuse keys via env to trace.
"""
import argparse, json
from agentic_rag.agent import AgenticRAG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--eval-set", required=True)
    args = ap.parse_args()
    rag = AgenticRAG.from_config(args.config)

    rows = [json.loads(l) for l in open(args.eval_set) if l.strip()]
    records = []
    for r in rows:
        ans = rag.answer(r["question"])
        records.append({
            "question": r["question"],
            "answer": ans.text,
            "ground_truth": r.get("ground_truth", ""),
            "n_citations": len(ans.citations),
        })
        print(f"[done] {r['question'][:60]}... ({len(ans.citations)} citations)")

    # --- Ragas scoring (uncomment once you have ragas installed + an eval set) ---
    # from ragas import evaluate
    # from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    # ... build a ragas Dataset from records (incl. retrieved contexts) and call evaluate(...)
    print(f"\nprocessed {len(records)} questions. Add Ragas scoring to publish the metrics table.")

if __name__ == "__main__":
    main()
