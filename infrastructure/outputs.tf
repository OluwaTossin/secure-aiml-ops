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

output "database_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database.id
}

output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
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

output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_task_execution_role.arn
}

# CloudWatch Log Group
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.main.name
}

# S3 Bucket
output "s3_data_bucket_name" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.data.bucket
}

output "s3_data_bucket_arn" {
  description = "ARN of the S3 data bucket"
  value       = aws_s3_bucket.data.arn
}

# Network Configuration
output "nat_gateway_ip" {
  description = "Elastic IP of the NAT Gateway"
  value       = aws_eip.nat.public_ip
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}