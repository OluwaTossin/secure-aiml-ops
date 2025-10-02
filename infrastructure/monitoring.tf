# CloudWatch Log Group for application logs
resource "aws_cloudwatch_log_group" "main" {
  name              = "/aws/${var.project_name}"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = {
    Name = "${var.project_name}-log-group"
  }
}

# CloudWatch Log Group for Airflow
resource "aws_cloudwatch_log_group" "airflow" {
  name              = "/aws/${var.project_name}/airflow"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = {
    Name = "${var.project_name}-airflow-log-group"
  }
}

# CloudWatch Log Group for Streamlit
resource "aws_cloudwatch_log_group" "streamlit" {
  name              = "/aws/${var.project_name}/streamlit"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = {
    Name = "${var.project_name}-streamlit-log-group"
  }
}

# CloudWatch Log Group for Bedrock
resource "aws_cloudwatch_log_group" "bedrock" {
  name              = "/aws/${var.project_name}/bedrock"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = {
    Name = "${var.project_name}-bedrock-log-group"
  }
}

# CloudWatch Metric Filter for Error Logs
resource "aws_cloudwatch_log_metric_filter" "error_logs" {
  name           = "${var.project_name}-error-logs"
  log_group_name = aws_cloudwatch_log_group.main.name
  pattern        = "[timestamp, request_id, ERROR]"

  metric_transformation {
    name      = "ErrorCount"
    namespace = "${var.project_name}/Application"
    value     = "1"
  }
}

# CloudWatch Alarm for High Error Rate
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ErrorCount"
  namespace           = "${var.project_name}/Application"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors application error rate"

  tags = {
    Name = "${var.project_name}-error-alarm"
  }
}

# SNS Topic for Alerts (optional)
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"

  tags = {
    Name = "${var.project_name}-alerts-topic"
  }
}

# GuardDuty Detector (if enabled)
resource "aws_guardduty_detector" "main" {
  count  = var.enable_guardduty ? 1 : 0
  enable = true

  tags = {
    Name = "${var.project_name}-guardduty"
  }
}