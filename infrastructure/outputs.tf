# VPC ID
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

# Subnet IDs
output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "private_subnet_id" {
  description = "ID of the private subnet"
  value       = aws_subnet.private.id
}

# Security Group IDs
output "web_security_group_id" {
  description = "ID of the web security group"
  value       = aws_security_group.web.id
}

output "internal_security_group_id" {
  description = "ID of the internal security group"
  value       = aws_security_group.internal.id
}

# ECR Repository
output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.main.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.main.arn
}

# IAM Role ARNs
output "airflow_role_arn" {
  description = "ARN of the Airflow execution role"
  value       = aws_iam_role.airflow_execution_role.arn
}

output "bedrock_role_arn" {
  description = "ARN of the Bedrock access role"
  value       = aws_iam_role.bedrock_access_role.arn
}

output "ecr_role_arn" {
  description = "ARN of the ECR access role"
  value       = aws_iam_role.ecr_access_role.arn
}

# CloudWatch Log Group
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.main.name
}