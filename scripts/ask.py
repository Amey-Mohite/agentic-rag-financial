"""CLI entry point: ask the agent a question from the terminal.

Usage:
    python scripts/ask.py --config config.yaml "What was FY revenue?"

It builds the AgenticRAG pipeline from a config file, runs one question through it, and
pretty-prints the answer, the citations, and the step/usage stats.
"""
import argparse              # parses command-line flags/arguments
import sys, pathlib          # used to put the project's src/ directory on the import path

# The package lives under <project>/src. This line inserts that folder at the front of
# sys.path so `import agentic_rag...` works even when running the script directly (no install).
#   __file__ -> this script -> .parent (scripts/) -> .parent (project root) -> /src
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from agentic_rag.agent import AgenticRAG


def main():
    """Parse args, run one question, print the answer + citations + stats."""
    # --- Define the CLI: an optional --config and one-or-more positional words for the question.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")   # path to the YAML config
    ap.add_argument("question", nargs="+")               # question words (joined with spaces below)
    args = ap.parse_args()

    # --- Build the pipeline and answer the question.
    rag = AgenticRAG.from_config(args.config)
    result = rag.answer(" ".join(args.question))  # rejoin the words into one query string

    # --- Print the answer.
    print("\n=== ANSWER ===\n" + result.text)
    # --- Print every citation: [source pPage] chunk <id> score <score>.
    print(f"\n=== CITATIONS ({len(result.citations)}) ===")
    for c in result.citations:
        print(f"  [{c['source']} p{c['page']}] chunk {c['chunk_id']} score {c['score']}")
    # --- Print how many search steps ran and the token usage.
    print(f"\nsteps={result.steps} usage={result.usage}")


# Standard Python idiom: only run main() when this file is executed directly
# (python scripts/ask.py ...), not when it's imported by another module.
if __name__ == "__main__":
    main()
