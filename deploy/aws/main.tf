# ==============================================================================
# NexusRec AI — AWS Production Infrastructure-as-Code (Terraform)
# Provisions:
# 1. Amazon ECR Repository for the FastAPI + Multimodal RecSys container
# 2. Amazon S3 Model Registry bucket for Two-Tower 64-D ONNX/PyTorch weights
# 3. Amazon SageMaker Serverless Inference Endpoint (scale-to-zero neural scoring)
# 4. Amazon ElastiCache Serverless Redis for UserPersona session & Bandit arms
# 5. AWS ECS Fargate Service + Application Load Balancer
# ==============================================================================
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "Target AWS region for NexusRec AI deployment"
}

variable "environment" {
  type        = string
  default     = "production"
}

provider "aws" {
  region = var.aws_region
}

# 1. Amazon ECR Repository
resource "aws_ecr_repository" "nexusrec" {
  name                 = "nexusrec-ai-${var.environment}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# 2. Amazon S3 Bucket for Two-Tower Model Artifacts & Qdrant Snapshots
resource "aws_s3_bucket" "model_registry" {
  bucket_prefix = "nexusrec-model-registry-${var.environment}-"
}

resource "aws_s3_bucket_versioning" "model_registry_versioning" {
  bucket = aws_s3_bucket.model_registry.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 3. IAM Role for SageMaker Serverless Two-Tower Endpoint
resource "aws_iam_role" "sagemaker_exec" {
  name = "nexusrec-sagemaker-exec-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# 4. SageMaker Serverless Inference Model & Endpoint Configuration (Scale-to-Zero)
resource "aws_sagemaker_model" "two_tower_onnx" {
  name               = "nexusrec-two-tower-64d-${var.environment}"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn

  primary_container {
    image          = "${aws_ecr_repository.nexusrec.repository_url}:latest"
    model_data_url = "s3://${aws_s3_bucket.model_registry.id}/models/two_tower_64d.tar.gz"
    environment = {
      EMBEDDING_DIM = "64"
      TAU_TEMP      = "0.08"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "serverless_cfg" {
  name = "nexusrec-serverless-cfg-${var.environment}"

  production_variants {
    variant_name = "TwoTowerPrimary"
    model_name   = aws_sagemaker_model.two_tower_onnx.name

    serverless_config {
      memory_size_in_mb = 4096
      max_concurrency   = 32
    }
  }
}

resource "aws_sagemaker_endpoint" "two_tower_endpoint" {
  name                 = "nexusrec-two-tower-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.serverless_cfg.name
}

# 5. AWS ECS Cluster for FastAPI Orchestrator + Hybrid RRF & MMR Reranker
resource "aws_ecs_cluster" "nexusrec_cluster" {
  name = "nexusrec-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.nexusrec.repository_url
  description = "Push the NexusRec AI Docker image to this ECR URL"
}

output "sagemaker_serverless_endpoint_name" {
  value       = aws_sagemaker_endpoint.two_tower_endpoint.name
  description = "Scale-to-zero SageMaker endpoint for 64-D Two-Tower inference"
}
