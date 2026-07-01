# Guide — use it, run it locally, deploy it

Everything you need to actually run this project. Three parts:
1. **Using the web app** (the bring-your-own-keys UI)
2. **Running it locally** (your machine)
3. **Deploying to Hugging Face Spaces** (free, public URL)

> You need two API keys regardless: **OpenAI** (embeddings) and **Anthropic** (the answer model).
> Hosting is free; the API calls are billed to your keys (cents per question for a demo).

---

# Part 1 — Using the web app

Open the app (locally `http://localhost:8000`, or your deployed URL). It has four tabs:

| Tab | What you do |
|---|---|
| **⚙️ Settings** | Paste your keys, choose where vectors are stored, tune the pipeline, click **Save & connect**. |
| **Upload** | Drag in PDFs/HTML/TXT/MD → they're vectorized into your index. |
| **Ask** | Ask questions → streamed answers with **source citations**; follow-ups remember context. |
| **How it's built** | A short architecture overview. |

The header shows **● connected / not connected** and how many **docs / chunks** you've indexed.

### Step-by-step
1. **Settings → API keys.** Paste your OpenAI and Anthropic keys. They're held in server memory for
   your session only and are never logged.
2. **Choose where to store vectors** (see §1.2). Easiest: *In-memory* (just needs the two keys).
3. *(Optional)* tune the pipeline — each option has a one-line explanation (and §1.1 below).
4. **Save & connect** — the server builds your pipeline and validates the keys. You'll see **✓ Connected**.
5. **Upload tab** → add documents → **Vectorize** (first upload is slow while models warm up).
6. **Ask tab** → ask a question; ask a follow-up to see memory work.

## 1.1 Every setting, and why it matters
**Models** — *Embedding model* (large = richest, small = cheaper) · *Generator (Claude) model* ·
*Max answer tokens* · *Temperature* (0 = factual) · *Prompt caching* (cheaper multi-step).
**Chunking** — *Strategy* (recursive = best default) · *Chunk size* (bigger = more context, less precise) ·
*Overlap* (guards facts on boundaries).
**Retrieval** — *Top-K* (chunks the model sees) · *Candidate pool* (fetched before reranking) ·
*Hybrid search* (dense + sparse) · *Cross-encoder rerank* (accuracy boost; needs ~2 GB RAM — turn OFF
on tiny hosts) · *Sparse backend* (keyword, or splade/BM42 — Qdrant only) · *RRF k* (fusion constant, 60).
**Agent / memory / ingestion** — *Max agent steps* (multi-part answers) · *Memory turns* (follow-ups) ·
*Conversation memory* on/off · *Table-aware PDF extraction* (keeps numbers structured).

## 1.2 Where is the database stored?
Choose on Settings → **Vector store**:
- **In-memory (default, zero setup):** vectors live inside the API server's RAM. ✅ Nothing to set up.
  ⚠️ Cleared on restart. Each browser session has its own isolated index.
- **Qdrant Cloud (persistent):** your managed cluster (free tier at cloud.qdrant.io, 1 GB). Enter URL + key.
- **Supabase / Postgres — pgvector (persistent):** a free Supabase Postgres DB (pgvector is included
  free). Paste the connection string from Supabase → Project Settings → Database (use the **Session
  pooler** or **Direct** URI). ⚠️ Free Supabase pauses after 7 days idle (resume ~60 s) and is 500 MB;
  learned-sparse (SPLADE) is Qdrant-only, so Postgres uses dense + keyword retrieval.

**Where do the uploaded files go?** To the server's disk only for the few seconds it takes to
extract → chunk → embed, then **deleted right after ingestion**. The durable, searchable data is the
**vectors** in your chosen store — never the raw files.

> Rule of thumb: **quick test → In-memory.** **Stable free link → Qdrant Cloud or Supabase/pgvector.**

## 1.2a Supabase setup (step-by-step, ~3 minutes)
Use this if you picked **Supabase / Postgres (pgvector)** for a free, persistent index. You do **not**
run any SQL — the app creates the extension, table, and indexes for you on first connect.

1. Go to **[supabase.com](https://supabase.com)** → sign in → **New project**. Pick a name, set a
   strong **database password** (save it), choose a region, and create. Wait ~1 minute for it to provision.
2. Open the project → **Project Settings** (gear icon) → **Database**.
3. Under **Connection string**, choose the **Session pooler** tab (most compatible with hosts; use
   **Direct connection** only if your host has IPv6). Copy the URI — it looks like:
   ```
   postgresql://postgres.<ref>:[YOUR-PASSWORD]@aws-0-<region>.pooler.supabase.com:5432/postgres
   ```
4. Replace `[YOUR-PASSWORD]` with the database password from step 1.
5. In the app → **⚙️ Settings** → **Vector store** → choose **Supabase / Postgres — pgvector** →
   paste the connection string into the **Postgres connection string** field.
6. Click **Save & connect**. On success the app has already created the `vector` extension + the
   `filings` table + indexes in your Supabase database. Now **Upload** documents and **Ask**.

**Verify it worked (optional):** in Supabase → **Table Editor**, you'll see a `filings` table fill up
with rows as you upload; or **Database → Extensions** shows `vector` enabled.

**Good to know:**
- Free Supabase **pauses after 7 days of inactivity** — if connect fails later, open the dashboard and
  **Restore/Resume** the project (~60 s), then reconnect.
- Free tier is **500 MB** — plenty for thousands of chunks.
- **SPLADE/BM42 is Qdrant-only**, so leave **Sparse backend = keyword** when using Postgres (the app
  uses dense + keyword retrieval, which works well).
- Needs **pgvector ≥ 0.7** for the `halfvec` type — current Supabase has it.

## 1.3 Troubleshooting the UI
| Message | Fix |
|---|---|
| "Please enter both API keys" | Fill the OpenAI **and** Anthropic fields. |
| "Could not connect with these settings" | A key is wrong, or (Qdrant/Postgres) the URL/connection string is wrong or the DB is paused. |
| "No pipeline for this session…" | Click **Save & connect** first (or reconnect after a server restart). |
| "I don't know" to everything | Upload documents first. |
| First upload/question slow | One-time model warm-up; later calls are fast. |
| Crash with reranker on | Needs ~2 GB RAM → turn **Cross-encoder rerank** OFF on small hosts. |

---

# Part 2 — Run it locally

### Prerequisites
- Python 3.12+, and your `OPENAI_API_KEY` + `ANTHROPIC_API_KEY`.
- (Optional) Docker, if you want the one-command path.

### Option A — One command (Docker, recommended)
Brings up the app + a local Qdrant and serves the UI:
```bash
cp .env.example .env        # put OPENAI_API_KEY and ANTHROPIC_API_KEY in it
docker compose up --build
# then open http://localhost:8000
```

### Option B — Python directly
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[tables]"        # add ".[hybrid]" too if you want SPLADE/BM42
cp .env.example .env               # fill in your keys
uvicorn agentic_rag.api:app --reload --port 8000
# open http://localhost:8000
```
With the web UI you don't even need keys in `.env` — you can paste them on the Settings tab and use
the **In-memory** store (zero setup).

### Optional — command-line scripts (no UI)
```bash
python scripts/download_filings.py                       # grab a few sample SEC filings into ./data
python scripts/ingest.py --config config.yaml --path "data/*"   # build the index
python scripts/ask.py --config config.yaml "What was total revenue last fiscal year?"
python scripts/run_eval.py --config config.yaml --eval-set data/eval_set.jsonl   # Ragas eval
```

### Run the tests
```bash
pip install -e ".[dev]" && pytest
```

---

# Part 3 — Deploy to Hugging Face Spaces (free, public URL)

Hugging Face gives Docker Spaces **16 GB RAM** free — enough for the reranker — and a public link
like `https://huggingface.co/spaces/<you>/agentic-rag`. Great for sharing on Upwork.

### 3.1 Create the Space
Create a free account at huggingface.co → **New → Space** → **SDK: Docker**, **CPU basic (free)**,
**Public**. Note the id `‹username›/‹space-name›` (e.g. `Amey91/agentic-rag`). Get a **write** token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3.2 Upload the project (folders and all)
**Option A — the uploader script (recommended; preserves folders, skips secrets):**
```bash
pip install huggingface_hub
python scripts/deploy_hf.py --repo <username>/<space-name> --token hf_xxxxx
# example:  python scripts/deploy_hf.py --repo Amey91/agentic-rag --token hf_xxxxx
```
Re-run the same command any time you change a file — the Space rebuilds automatically. The script
excludes `.env` and local junk, so your keys never leave your machine.

**Option B — git:**
```bash
git clone https://huggingface.co/spaces/<username>/<space-name> hf-space
cp -r ./* hf-space/ && cd hf-space   # do NOT copy .env
git add . && git commit -m "deploy" && git push
```
(When git asks for a password, paste your HF **write token**.) Or use the Space's **Files → Upload**
button for a file or two.

### 3.3 The Space card metadata (required — this is what causes "config error")
A Docker Space is configured by a **YAML block at the top of `README.md`** (the "Space card"). This
repo's README already includes it:
```
---
title: Agentic RAG Document QA
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---
```
If your Space shows *"config error that prevents it from running"*, it's because this block is missing
or malformed at the very top of the README in the Space — make sure it's there, exactly, as the first
thing in the file. `app_port: 8000` routes HF to our container (which listens on 8000), so you do
**not** need to set any `PORT` variable.

### 3.4 (optionally) pre-set keys
- Easiest model: leave it at that and let each visitor enter **their own** keys on the Settings tab
  with the **In-memory** store. Nothing else to configure, and it won't spend your credits.
- Or pre-configure it yourself by adding **Secrets** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and (for a
  persistent index) `QDRANT_URL`/`QDRANT_API_KEY` + Variable `RAG_CONFIG=config.k8s.yaml`.

### 3.5 Build & open
The Space builds the image (it pre-downloads the reranker — first build takes a few minutes). When
the status is **Running**, open the URL → **Settings** (enter keys if not pre-set) → **Upload** →
**Ask**.

### 3.6 Make persistence stick
HF disk is **ephemeral** (cleared on restart/sleep). The **In-memory** store is cleared too. For a
stable index that survives restarts, choose **Qdrant Cloud** or **Supabase/pgvector** on the Settings
tab (both free, see §1.2).

### 3.7 Notes for sharing the link
- Visitors use **their own** keys → your credits are safe.
- Free Spaces **sleep when idle** and wake on visit — pre-warm it (ask one question) before demoing.
- For a stable demo you control: pre-configure Qdrant Cloud/Supabase and pre-upload a couple of files.

---

## Quick reference

| I want to… | Do this |
|---|---|
| Try it fastest | Run locally (Part 2A), open the UI, In-memory store, paste keys, upload, ask. |
| Share a public link | Deploy to HF Spaces (Part 3); visitors bring their own keys. |
| Persist my index | Pick Qdrant Cloud or Supabase/pgvector in Settings. |
| Understand the code | Open `agentic_rag_walkthrough.ipynb`; read `README.md`. |
