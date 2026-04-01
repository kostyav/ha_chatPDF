# Part 5.3 — Kubernetes Deployment on Minikube

Deploys the full RAG system (Parts 2–4) on a local Kubernetes cluster using minikube. All Docker Compose services are translated to K8s manifests with the same service names, so internal DNS resolution is identical to the Compose setup.

---

## Directory Layout

```
src/part5_3/
├── k8s/
│   ├── kustomization.yaml          # kubectl apply -k entry point
│   ├── namespace.yaml              # rag-system namespace
│   ├── configmap.yaml              # shared env vars (Redis, Qdrant, Ollama URLs)
│   ├── secret.yaml                 # HuggingFace token template
│   ├── pvc/                        # PersistentVolumeClaims (minikube hostPath)
│   │   ├── redis-pvc.yaml
│   │   ├── qdrant-pvc.yaml
│   │   ├── pdf-data-pvc.yaml
│   │   ├── parsed-docs-pvc.yaml
│   │   ├── byaldi-index-pvc.yaml
│   │   └── query-logs-pvc.yaml
│   ├── redis/                      # Redis Deployment + ClusterIP Service
│   ├── qdrant/                     # Qdrant Deployment + ClusterIP Service
│   ├── ollama/                     # Ollama Deployment + ClusterIP Service
│   │                               #   initContainer pulls gemma3:4b on start
│   ├── parser/                     # Docling parser Deployment (queue consumer)
│   ├── text-indexer/               # sentence-transformers + Qdrant indexer
│   ├── visual-indexer/             # ColQwen2 / Byaldi indexer (GPU optional)
│   ├── orchestrator/
│   │   ├── indexer-job.yaml        # one-shot Job: index all PDFs
│   │   └── query-job.yaml          # one-shot Job: run a single query
│   └── api/                        # FastAPI + SSE service (NodePort 30808)
└── config/
    └── config.yaml                 # updated config pointing to K8s services
```

---

## Service → K8s Mapping

| Docker Compose service | K8s workload type | K8s Service type | Port |
|------------------------|-------------------|------------------|------|
| redis                  | Deployment        | ClusterIP        | 6379 |
| qdrant                 | Deployment        | ClusterIP        | 6333 |
| ollama                 | Deployment        | ClusterIP        | 11434 |
| parser                 | Deployment        | —                | — |
| text_indexer           | Deployment        | —                | — |
| visual_indexer         | Deployment        | —                | — |
| orchestrator (main.py) | **Job**           | —                | — |
| orchestrator (query.py)| **Job**           | —                | — |
| api (Part 4)           | Deployment        | NodePort         | 30808 |

Parser, text_indexer, and visual_indexer communicate only via Redis queues — no K8s Service is needed for them.

---

## Prerequisites

### kubectl

kubectl is the CLI for every Kubernetes operation in this guide.

**Linux (x86_64)**
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

**macOS**
```bash
brew install kubectl
kubectl version --client
```

**Windows (PowerShell, run as Administrator)**
```powershell
winget install Kubernetes.kubectl
kubectl version --client
```

Verify it works:
```bash
kubectl version --client
# Client Version: v1.xx.x
```

---

### minikube

minikube runs a single-node Kubernetes cluster locally. It requires a container or VM driver — Docker (already available here) is used by default.

**Linux (x86_64)**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube version
```

**macOS**
```bash
brew install minikube
minikube version
```

**Windows (PowerShell, run as Administrator)**
```powershell
winget install Kubernetes.minikube
minikube version
```

**Docker driver prerequisite** — minikube's default driver uses Docker. Confirm it is running:
```bash
docker info   # must succeed; if not, start Docker Desktop or the Docker daemon
```

---

## Step 1 — Start Minikube

```bash
# CPU-only (no GPU):
minikube start --memory=16g --cpus=6 --disk-size=40g

# With GPU (requires nvidia-container-toolkit on host):
minikube start --memory=16g --cpus=6 --disk-size=40g \
  --driver=docker --gpus all
```

Enable GPU device plugin (only if using GPU):
```bash
minikube addons enable nvidia-gpu-device-plugin
```

---

## Step 2 — Build Images into Minikube

Point your shell's Docker client to minikube's Docker daemon so images are available inside the cluster without a registry:

```bash
eval $(minikube docker-env)
```

Build all custom images:

```bash
# From repo root
docker build -t rag/parser:latest        -f src/part2/services/parser/Dockerfile        src/part2/
docker build -t rag/text-indexer:latest  -f src/part2/services/text_indexer/Dockerfile  src/part2/
docker build -t rag/visual-indexer:latest -f src/part2/services/visual_indexer/Dockerfile src/part2/
docker build -t rag/orchestrator:latest  -f src/part2/services/orchestrator/Dockerfile  src/part2/
docker build -t rag/api:latest           -f src/part4/Dockerfile                         .
```

All K8s manifests use `imagePullPolicy: Never` so Kubernetes uses these local images.

---

## Step 3 — HuggingFace Token (gated models)

ColQwen2-v0.1 is a gated HuggingFace model. Create the Secret before applying:

```bash
kubectl create secret generic hf-token \
  --from-literal=HUGGING_FACE_HUB_TOKEN=<YOUR_HF_TOKEN> \
  -n rag-system
```

> **Do not** commit `secret.yaml` with a real token. The file in this repo is a placeholder template.

---

## Step 4 — Copy PDFs into the Cluster

The `pdf-data` PVC must contain the PDFs before indexing.

```bash
# One-time helper pod to copy files into the PVC
kubectl run pdf-loader --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"pdf","persistentVolumeClaim":{"claimName":"pdf-data"}}],"containers":[{"name":"pdf-loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"pdf","mountPath":"/data/pdfs"}]}]}}' \
  -n rag-system

# Copy PDFs
kubectl cp src/part2/example_data/23870758.pdf rag-system/pdf-loader:/data/pdfs/
kubectl cp src/part2/example_data/24069913.pdf rag-system/pdf-loader:/data/pdfs/

# Cleanup
kubectl delete pod pdf-loader -n rag-system
```

---

## Step 5 — Deploy

```bash
# Apply everything except the orchestrator Jobs (they run after services are up)
kubectl apply -k src/part5_3/k8s/
```

Wait for infrastructure to be healthy:

```bash
kubectl rollout status deployment/redis        -n rag-system
kubectl rollout status deployment/qdrant       -n rag-system
kubectl rollout status deployment/ollama       -n rag-system
kubectl rollout status deployment/parser       -n rag-system
kubectl rollout status deployment/text-indexer -n rag-system
kubectl rollout status deployment/visual-indexer -n rag-system
```

---

## Step 6 — Index Documents

```bash
kubectl apply -f src/part5_3/k8s/orchestrator/indexer-job.yaml -n rag-system
kubectl logs -f job/orchestrator-indexer -n rag-system
```

Wait until the Job completes (`kubectl get job orchestrator-indexer -n rag-system` shows `1/1`).

---

## Step 7 — Query

**One-shot query (edit `args` in `query-job.yaml` first):**

```bash
kubectl apply -f src/part5_3/k8s/orchestrator/query-job.yaml -n rag-system
kubectl logs -f job/orchestrator-query -n rag-system
# Cleanup before next run:
kubectl delete job orchestrator-query -n rag-system
```

**Interactive REPL:**

```bash
kubectl run query-repl --rm -it --restart=Never \
  --image=rag/orchestrator:latest \
  --image-pull-policy=Never \
  --overrides='{"spec":{"volumes":[{"name":"logs","persistentVolumeClaim":{"claimName":"query-logs"}}],"containers":[{"name":"query-repl","image":"rag/orchestrator:latest","imagePullPolicy":"Never","command":["python","query.py"],"stdin":true,"tty":true,"envFrom":[{"configMapRef":{"name":"rag-config"}}],"volumeMounts":[{"name":"logs","mountPath":"/data/logs"}]}]}}' \
  -n rag-system
```

---

## Step 8 — Access the API

```bash
# Print the URL
minikube service api -n rag-system --url
# Opens in browser:
minikube service api -n rag-system
```

Or access directly on the NodePort:
```bash
curl http://$(minikube ip):30808/
```

Stream a chat response:
```bash
curl -N -X POST http://$(minikube ip):30808/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Fig. 4 show?"}'
```

---

## GPU Notes

The following services use GPU in production:

| Service        | Why                            | Default in this manifest |
|----------------|--------------------------------|--------------------------|
| parser         | Docling layout model           | CPU (GPU commented out)  |
| visual-indexer | ColQwen2 / Byaldi              | CPU (GPU commented out)  |
| ollama         | LLM inference                  | CPU (GPU commented out)  |

To enable GPU for a service, uncomment the `nvidia.com/gpu: "1"` block in its Deployment manifest under `resources.limits`.

---

## Teardown

```bash
kubectl delete namespace rag-system
minikube stop
```

---

## Configuration

[config/config.yaml](config/config.yaml) mirrors `src/part2/config/config.yaml` with all hostnames updated to K8s service names (`redis`, `qdrant`, `ollama`). These names are identical to what Docker Compose used, so no application code changes were needed — only the ConfigMap and this config file reflect the K8s deployment topology.
