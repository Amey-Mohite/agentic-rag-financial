# Running Agentic RAG on Local Minikube (Windows / PowerShell)

A step-by-step guide to deploy this project on a local Kubernetes cluster with Minikube, written for
**Windows PowerShell**. Every command here is PowerShell — copy/paste directly.

> If you just want it running locally with less ceremony, use `docker compose up` instead (see the
> main README). Use Minikube when you want to practice the real Kubernetes deployment flow.

> **Why the extra steps?** Minikube runs its own container runtime *separate* from your host Docker,
> so your locally built image is invisible to the cluster until you load it. The two things people
> get wrong: (1) not loading the image into Minikube, and (2) the `:latest` image-pull trap. Both are
> handled below.

---

## Prerequisites

- [Minikube](https://minikube.sigs.k8s.io/docs/start/) installed (Docker Desktop is the easiest driver on Windows)
- `kubectl` installed (Docker Desktop bundles it; otherwise `winget install Kubernetes.kubectl`)
- Docker Desktop running
- API keys ready: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- A Qdrant API key you generate yourself (Step 3)

> Run all commands in **PowerShell** (not Command Prompt). These are plain commands, not scripts, so
> no execution-policy change is needed.

---

## Step 1 — Start the cluster

```powershell
minikube start

# sanity check — the node should show Ready:
kubectl get nodes
```

---

## Step 2 — Build the image *into* Minikube

Point your Docker CLI at Minikube's internal daemon and build there — no registry needed.

```powershell
# this PowerShell session now talks to Minikube's Docker daemon:
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# build with a REAL tag (never :latest — see the note below):
docker build -t agentic-rag:0.1.0 .

# confirm the image is inside Minikube:
minikube image ls | Select-String agentic-rag
```

> **Alternative (build on host, then load):**
> ```powershell
> docker build -t agentic-rag:0.1.0 .
> minikube image load agentic-rag:0.1.0
> ```

> ⚠️ **The `:latest` trap.** Kubernetes defaults `imagePullPolicy` to `Always` for images tagged
> `:latest` (or untagged), forcing a registry pull that fails for a local-only image →
> `ImagePullBackOff`. Always use a specific tag like `0.1.0` **and** set `imagePullPolicy: Never`
> (Step 5).

> **Note:** `docker-env` only applies to the *current* PowerShell window. Open a new window and you're
> back on Docker Desktop's daemon — re-run the `docker-env` line if you need Minikube's daemon again.

---

## Step 3 — Deploy Qdrant inside the cluster

The app pods reach Qdrant by its in-cluster service name (`qdrant`), so run Qdrant in the cluster too.

```powershell
# generate a Qdrant API key (this IS your credential — there is no signup for self-hosted Qdrant):
$env:QDRANT_API_KEY = -join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Max 256) })
echo $env:QDRANT_API_KEY        # save this value; you reuse it in Step 4

kubectl create deployment qdrant --image=qdrant/qdrant:latest
kubectl set env deployment/qdrant QDRANT__SERVICE__API_KEY=$env:QDRANT_API_KEY
kubectl expose deployment qdrant --port=6333 --target-port=6333

# wait for it to come up:
kubectl get pods -l app=qdrant
```

> If you have Git for Windows, `openssl` is usually on PATH and this also works:
> `$env:QDRANT_API_KEY = (openssl rand -hex 32)`

---

## Step 4 — Create the secret your app reads

The deployment loads secrets via `envFrom: secretRef: agentic-rag-secrets`. Use your real keys and the
**same** `QDRANT_API_KEY` from Step 3. In PowerShell, the backtick `` ` `` is the line-continuation
character:

```powershell
kubectl create secret generic agentic-rag-secrets `
  --from-literal=OPENAI_API_KEY=sk-... `
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... `
  --from-literal=QDRANT_API_KEY=$env:QDRANT_API_KEY `
  --from-literal=QDRANT_URL=http://qdrant:6333
```

> Or as a single line (no backticks) if you prefer:
> ```powershell
> kubectl create secret generic agentic-rag-secrets --from-literal=OPENAI_API_KEY=sk-... --from-literal=ANTHROPIC_API_KEY=sk-ant-... --from-literal=QDRANT_API_KEY=$env:QDRANT_API_KEY --from-literal=QDRANT_URL=http://qdrant:6333
> ```

> `QDRANT_URL=http://qdrant:6333` is the in-cluster DNS name of the Qdrant service from Step 3 — that's
> how the app pod finds Qdrant.

---

## Step 5 — Patch the manifest for local images

The repo's `k8s/deployment.yaml` points at `REGISTRY/agentic-rag:latest` (a production placeholder).
For Minikube, edit the container spec so it uses your local image and never tries to pull it:

```yaml
# k8s/deployment.yaml — inside spec.template.spec.containers[0]
        - name: api
          image: agentic-rag:0.1.0      # your locally built tag (NOT :latest)
          imagePullPolicy: Never        # CRITICAL: use the local image, do not pull from a registry
```

Open the file in your editor (`notepad k8s\deployment.yaml` or VS Code) and make those two edits.
Everything else (the `/healthz` readiness probe, the Service) stays as-is.

---

## Step 6 — Deploy the app

```powershell
kubectl apply -f k8s/deployment.yaml

# wait until the agentic-rag pod is Running:
kubectl get pods
```

If a pod is stuck, inspect it:

```powershell
kubectl describe pod -l app=agentic-rag    # read the Events section at the bottom
kubectl logs -l app=agentic-rag
```

---

## Step 7 — Ingest documents

Ingestion runs from your machine against the in-cluster Qdrant. Port-forward Qdrant in one window, run
the scripts in another.

**Window A — forward Qdrant to your machine:**
```powershell
kubectl port-forward svc/qdrant 6333:6333
```

**Window B — download filings + ingest:**
```powershell
$env:SEC_USER_AGENT = "Your Name your@email.com"
python scripts/download_filings.py --tickers AAPL MSFT JPM --forms 10-K 10-Q

# point the local config at the forwarded Qdrant + provide the key:
#   in config.yaml set:  vector_store.url: http://localhost:6333
$env:QDRANT_API_KEY = "<the key you generated in Step 3>"
python scripts/ingest.py --config config.yaml --path "data/*"
```

> The app pod uses `http://qdrant:6333` (in-cluster); your machine uses `http://localhost:6333` (the
> port-forward). Same Qdrant, two addresses depending on where you call it from.

---

## Step 8 — Reach the API

**Window C — forward the API:**
```powershell
kubectl port-forward svc/agentic-rag 8000:80
```

**Another window — call it.** PowerShell's `curl` is an alias for `Invoke-WebRequest` with different
syntax, so use `Invoke-RestMethod`:

```powershell
# health check:
Invoke-RestMethod http://localhost:8000/healthz

# ask a question:
$body = @{ question = "What was total revenue in the most recent fiscal year?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -ContentType "application/json" -Body $body
```

> If you prefer real `curl.exe` (Windows 10+ ships it), call it explicitly so PowerShell doesn't use
> its alias:
> ```powershell
> curl.exe -s http://localhost:8000/ask -H "content-type: application/json" -d '{\"question\":\"What was total revenue in the most recent fiscal year?\"}'
> ```

---

## Rebuilding after a code change

You used a fixed tag, so bump it each rebuild or Kubernetes won't notice the change:

```powershell
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t agentic-rag:0.1.1 .
# update image: agentic-rag:0.1.1 in k8s/deployment.yaml, then:
kubectl apply -f k8s/deployment.yaml
kubectl rollout restart deployment/agentic-rag
```

> Reusing the *same* tag won't trigger a rollout (K8s sees the tag unchanged). Bump the tag (best) or
> `kubectl rollout restart` after re-loading the image.

---

## Cleanup

```powershell
kubectl delete -f k8s/deployment.yaml
kubectl delete deployment qdrant
kubectl delete service qdrant
kubectl delete secret agentic-rag-secrets
minikube stop                 # or: minikube delete  (removes the whole cluster + image cache)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImagePullBackOff` | K8s trying to pull a local-only image | Use a real tag + `imagePullPolicy: Never` (Steps 2, 5) |
| `ErrImageNeverPull` | Image not actually in Minikube | `minikube image ls`; rebuild after the `docker-env` line |
| App can't reach Qdrant | Wrong URL or Qdrant not up | App must use `http://qdrant:6333`; check `kubectl get pods -l app=qdrant` |
| Qdrant rejects requests (401) | API key mismatch | Secret's `QDRANT_API_KEY` must equal the one on the Qdrant deployment |
| `Invoke-WebRequest` errors on `curl` | PowerShell aliases `curl` | Use `Invoke-RestMethod`, or call `curl.exe` explicitly |
| `docker build` not hitting Minikube | New window lost the docker-env | Re-run the `minikube ... docker-env \| Invoke-Expression` line |
| Image stale after rebuild | Same tag reused | Bump the tag or `kubectl rollout restart deployment/agentic-rag` |
| Backtick line-continuation fails | Trailing space after `` ` `` | Ensure nothing follows the backtick, or use the single-line form |

---

## When to use this vs. Docker Compose

- **Docker Compose** (`docker compose up`) — fastest local demo; one command brings up Qdrant + the API
  together. Use for day-to-day development.
- **Minikube** (this guide) — practice the real Kubernetes flow (images, secrets, services, probes,
  rollouts). Use for the K8s story in interviews or to mirror production.
