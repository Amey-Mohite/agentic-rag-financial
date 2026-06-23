"""CLI: ask a question. Usage: python scripts/ask.py --config config.yaml "What was FY revenue?" """
import argparse
from agentic_rag.agent import AgenticRAG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("question", nargs="+")
    args = ap.parse_args()
    rag = AgenticRAG.from_config(args.config)
    result = rag.answer(" ".join(args.question))
    print("\n=== ANSWER ===\n" + result.text)
    print(f"\n=== CITATIONS ({len(result.citations)}) ===")
    for c in result.citations:
        print(f"  [{c['source']} p{c['page']}] chunk {c['chunk_id']} score {c['score']}")
    print(f"\nsteps={result.steps} usage={result.usage}")

if __name__ == "__main__":
    main()
