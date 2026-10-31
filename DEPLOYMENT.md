# Cloud Deployment Architecture & Zero-Idle-Cost Runbook (`NexusRec AI`)

This document outlines the production deployment topologies for **NexusRec AI** across **Google Cloud Platform (GCP)** and **Amazon Web Services (AWS)**, including scale-to-zero FinOps cost engineering, hybrid vector indexing strategies, and exact CLI deployment sequences.

---

## 1. Architectural Topology & Trade-Offs

```mermaid
flowchart LR
    Client["Operator / Shopper UI"] --> API["FastAPI Lifespan Service<br/>(Cloud Run / ECS Fargate)"]
    API --> NLP["Node 1: Conversational Intent<br/>& Visual Descriptor (32-D)"]
    NLP --> TT["Node 2: Two-Tower 64-D<br/>User & Item Encoders"]
    TT --> Qdrant["Node 3: Hybrid Retriever<br/>Dense Cosine + BM25 RRF (k=60)"]
    Qdrant --> Rerank["Node 4: GBDT Cross-Encoder<br/>+ Thompson Bandit + MMR"]
    Rerank --> Redis[("Redis / Memorystore<br/>Session & Beta(α,β) Arms")]
```

| Architectural Dimension | Google Cloud Run (Recommended Default) | AWS ECS Fargate + SageMaker Serverless |
| :--- | :--- | :--- |
| **Compute Scaling** | `minScale: 0` to `maxScale: 12` in < 1.8s with `--startup-cpu-boost` | ECS Fargate Spot (API) + SageMaker Serverless Inference (Two-Tower) |
| **GPU Acceleration** | On-demand `nvidia-l4` attachment for large batch catalog re-encoding | SageMaker Serverless 4GB–6GB RAM (CPU/ONNX) or `ml.g5.xlarge` async |
| **Idle Baseline Cost** | **$0.00 / month** when `minScale: 0` | **~$8–$14 / month** (ALB + ECR/S3 storage; $0 compute for SageMaker Serverless) |
| **P95 Funnel Latency** | `3.5 ms – 6.2 ms` (in-memory hybrid index) | `8.0 ms – 14.5 ms` (across VPC endpoint to SageMaker Serverless) |

---

## 2. Google Cloud Platform (GCP) Deployment Guide

### Prerequisites
- Google Cloud SDK (`gcloud`) authenticated to your target project
- Artifact Registry API and Cloud Run Admin API enabled

### Step-by-Step CLI Commands

```bash
# 1. Set environment variables
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO_NAME="nexusrec-repo"

gcloud config set project "${PROJECT_ID}"
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com

# 2. Create Docker Artifact Registry repository (one-time)
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="NexusRec AI Multimodal Discovery Containers"

# 3. Trigger Google Cloud Build (runs pytest gate -> builds image -> deploys to Cloud Run)
gcloud builds submit --config=deploy/gcp/cloudbuild.yaml \
  --substitutions=_REGION="${REGION}",_ARTIFACT_REPO="${REPO_NAME}"

# 4. Alternative: Apply the declarative Knative service manifest directly
sed "s/PROJECT_ID/${PROJECT_ID}/g" deploy/gcp/service.yaml | \
  gcloud run services replace - --region="${REGION}"
```

### Enabling NVIDIA L4 GPU on Cloud Run (Optional Batch Indexing Mode)
When encoding >100,000 high-resolution product images with PyTorch + OpenCV, uncomment the `run.googleapis.com/accelerator: "nvidia-l4"` annotation and `nvidia.com/gpu: "1"` resource limit in `deploy/gcp/service.yaml`. Cloud Run scales GPU instances down to **zero** after 15 minutes of inactivity, eliminating the $500+/month idle cost of dedicated GPU VMs.

---

## 3. Amazon Web Services (AWS) Deployment Guide

### Prerequisites
- AWS CLI v2 (`aws configure`) and Terraform `>= 1.6.0`
- Docker Desktop or Buildx

### Step-by-Step CLI Commands

```bash
# 1. Initialize and apply Terraform infrastructure
cd deploy/aws
terraform init
terraform plan -var="aws_region=us-east-1" -var="environment=production"
terraform apply -auto-approve

# 2. Authenticate Docker to the provisioned Amazon ECR repository
export ECR_URL=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin "${ECR_URL}"

# 3. Build and push the multi-stage production image from repo root
cd ../..
docker build -t "${ECR_URL}:latest" .
docker push "${ECR_URL}:latest"
```

---

## 4. Local Single-Command Verification

```bash
# Install dependencies and start the full-stack FastAPI + Operator UI server
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run the complete unit and RecSys evaluation test suite
pytest -v
```
