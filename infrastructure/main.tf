# Terraform Configuration for Secure AI/ML Infrastructure

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "secure-aiml-ops"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}