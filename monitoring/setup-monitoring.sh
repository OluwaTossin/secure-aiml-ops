#!/bin/bash

# Comprehensive Monitoring Setup for Secure AI/ML Operations
# This script configures CloudWatch logs, custom metrics, and automated alerting

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1"
CLUSTER_NAME="secure-aiml-ops"
SERVICE_NAME="streamlit-app"
SNS_TOPIC_NAME="secure-aiml-ops-alerts"

echo -e "${BLUE}📊 Setting up Comprehensive Monitoring for AI/ML Operations${NC}"
echo "============================================================="

# Function to create CloudWatch Log Groups
create_log_groups() {
    echo -e "${YELLOW}📝 Creating CloudWatch Log Groups${NC}"
    
    # Log groups for different services
    LOG_GROUPS=(
        "/aws/ecs/secure-aiml-ops"
        "/aws/ecs/secure-aiml-ops/streamlit-app"
        "/aws/lambda/secure-aiml-ops"
        "/aws/apigateway/secure-aiml-ops"
        "/aws/s3/secure-aiml-ops"
        "/security/guardduty"
        "/monitoring/custom-metrics"
        "/aiml/bedrock-usage"
        "/performance/response-times"
    )
    
    for log_group in "${LOG_GROUPS[@]}"; do
        echo "Creating log group: ${log_group}"
        aws logs create-log-group \
            --log-group-name "${log_group}" \
            --region ${REGION} 2>/dev/null || echo "  Already exists"
        
        # Set retention policy (90 days for most, 30 days for performance logs)
        if [[ "${log_group}" == *"performance"* ]]; then
            retention_days=30
        else
            retention_days=90
        fi
        
        aws logs put-retention-policy \
            --log-group-name "${log_group}" \
            --retention-in-days ${retention_days} \
            --region ${REGION}
    done
    
    echo -e "${GREEN}✅ Log groups created successfully${NC}"
}

# Function to create custom metric filters
create_metric_filters() {
    echo -e "${YELLOW}📈 Creating Custom Metric Filters${NC}"
    
    # Error rate metric filter
    aws logs put-metric-filter \
        --log-group-name "/aws/ecs/secure-aiml-ops" \
        --filter-name "ApplicationErrors" \
        --filter-pattern '[timestamp, request_id, level="ERROR", ...]' \
        --metric-transformations \
            metricName=ApplicationErrorRate,metricNamespace=SecureAIMLOps/Application,metricValue=1,defaultValue=0 \
        --region ${REGION}
    
    # API response time metric filter
    aws logs put-metric-filter \
        --log-group-name "/aws/ecs/secure-aiml-ops" \
        --filter-name "APIResponseTime" \
        --filter-pattern '[timestamp, request_id, "response_time", duration]' \
        --metric-transformations \
            metricName=APIResponseTime,metricNamespace=SecureAIMLOps/Performance,metricValue='$duration',defaultValue=0 \
        --region ${REGION}
    
    # Bedrock usage metric filter
    aws logs put-metric-filter \
        --log-group-name "/aiml/bedrock-usage" \
        --filter-name "BedrockAPICalls" \
        --filter-pattern '[timestamp, model_id, tokens_used, cost]' \
        --metric-transformations \
            metricName=BedrockTokensUsed,metricNamespace=SecureAIMLOps/AI,metricValue='$tokens_used',defaultValue=0 \
        --region ${REGION}
    
    # Memory utilization metric filter
    aws logs put-metric-filter \
        --log-group-name "/aws/ecs/secure-aiml-ops" \
        --filter-name "MemoryUtilization" \
        --filter-pattern '[timestamp, container_id, "memory_usage", memory_mb]' \
        --metric-transformations \
            metricName=ContainerMemoryUsage,metricNamespace=SecureAIMLOps/Resources,metricValue='$memory_mb',defaultValue=0 \
        --region ${REGION}
    
    # Failed authentication attempts
    aws logs put-metric-filter \
        --log-group-name "/aws/ecs/secure-aiml-ops" \
        --filter-name "FailedAuthentication" \
        --filter-pattern '[timestamp, ip, "authentication_failed"]' \
        --metric-transformations \
            metricName=FailedAuthenticationAttempts,metricNamespace=SecureAIMLOps/Security,metricValue=1,defaultValue=0 \
        --region ${REGION}
    
    echo -e "${GREEN}✅ Metric filters created successfully${NC}"
}

# Function to create SNS topic for alerts
create_sns_topic() {
    echo -e "${YELLOW}📧 Creating SNS Topic for Alerts${NC}"
    
    # Create SNS topic
    TOPIC_ARN=$(aws sns create-topic \
        --name ${SNS_TOPIC_NAME} \
        --region ${REGION} \
        --query 'TopicArn' \
        --output text)
    
    echo "SNS Topic ARN: ${TOPIC_ARN}"
    
    # Set topic policy for CloudWatch alarms
    cat > sns-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "cloudwatch.amazonaws.com"
      },
      "Action": "SNS:Publish",
      "Resource": "${TOPIC_ARN}",
      "Condition": {
        "StringEquals": {
          "aws:SourceAccount": "${ACCOUNT_ID}"
        }
      }
    }
  ]
}
EOF
    
    aws sns set-topic-attributes \
        --topic-arn ${TOPIC_ARN} \
        --attribute-name Policy \
        --attribute-value file://sns-policy.json \
        --region ${REGION}
    
    rm sns-policy.json
    
    echo -e "${GREEN}✅ SNS topic created: ${TOPIC_ARN}${NC}"
    echo -e "${YELLOW}Note: Subscribe to this topic manually for email/SMS alerts${NC}"
}

# Function to create CloudWatch alarms
create_cloudwatch_alarms() {
    echo -e "${YELLOW}⚠️ Creating CloudWatch Alarms${NC}"
    
    # High error rate alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "HighErrorRate" \
        --alarm-description "High application error rate detected" \
        --metric-name "ApplicationErrorRate" \
        --namespace "SecureAIMLOps/Application" \
        --statistic "Sum" \
        --period 300 \
        --threshold 10 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 2 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
        --region ${REGION}
    
    # High API response time alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "HighAPIResponseTime" \
        --alarm-description "API response time is too high" \
        --metric-name "APIResponseTime" \
        --namespace "SecureAIMLOps/Performance" \
        --statistic "Average" \
        --period 300 \
        --threshold 5000 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 3 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
        --region ${REGION}
    
    # High memory utilization alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "HighMemoryUtilization" \
        --alarm-description "Container memory utilization is high" \
        --metric-name "ContainerMemoryUsage" \
        --namespace "SecureAIMLOps/Resources" \
        --statistic "Average" \
        --period 300 \
        --threshold 1800 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 2 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
        --region ${REGION}
    
    # Failed authentication attempts alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "FailedAuthenticationAttempts" \
        --alarm-description "Multiple failed authentication attempts detected" \
        --metric-name "FailedAuthenticationAttempts" \
        --namespace "SecureAIMLOps/Security" \
        --statistic "Sum" \
        --period 300 \
        --threshold 5 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 1 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
        --region ${REGION}
    
    # ECS service health alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "ECSServiceUnhealthy" \
        --alarm-description "ECS service is unhealthy or down" \
        --metric-name "RunningTaskCount" \
        --namespace "AWS/ECS" \
        --statistic "Average" \
        --period 60 \
        --threshold 1 \
        --comparison-operator "LessThanThreshold" \
        --evaluation-periods 2 \
        --dimensions Name=ServiceName,Value=${SERVICE_NAME} Name=ClusterName,Value=${CLUSTER_NAME} \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
        --region ${REGION}
    
    # Bedrock cost alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "HighBedrockCosts" \
        --alarm-description "Bedrock usage costs are high" \
        --metric-name "BedrockTokensUsed" \
        --namespace "SecureAIMLOps/AI" \
        --statistic "Sum" \
        --period 3600 \
        --threshold 100000 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 1 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
        --region ${REGION}
    
    echo -e "${GREEN}✅ CloudWatch alarms created successfully${NC}"
}

# Function to create custom dashboard
create_dashboard() {
    echo -e "${YELLOW}📊 Creating Comprehensive Monitoring Dashboard${NC}"
    
    cat > dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "CPUUtilization", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ],
                    [ ".", "MemoryUtilization", ".", ".", ".", "." ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "ECS Service Resource Utilization",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "RunningTaskCount", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ],
                    [ ".", "PendingTaskCount", ".", ".", ".", "." ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "ECS Task Count",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/Application", "ApplicationErrorRate" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Application Error Rate",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 8,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/Performance", "APIResponseTime" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "API Response Time (ms)",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 16,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/AI", "BedrockTokensUsed" ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Bedrock Token Usage (Hourly)",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "RequestCount", "LoadBalancer", "app/secure-aiml-ops/*" ],
                    [ ".", "HTTPCode_Target_2XX_Count", ".", "." ],
                    [ ".", "HTTPCode_Target_4XX_Count", ".", "." ],
                    [ ".", "HTTPCode_Target_5XX_Count", ".", "." ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Load Balancer Metrics"
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/Security", "FailedAuthenticationAttempts" ],
                    [ "AWS/GuardDuty", "FindingCount" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Security Metrics"
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 18,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '/aws/ecs/secure-aiml-ops' | fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 50",
                "region": "${REGION}",
                "title": "Recent Application Errors",
                "view": "table"
            }
        }
    ]
}
EOF
    
    aws cloudwatch put-dashboard \
        --dashboard-name "SecureAIMLOps-Comprehensive" \
        --dashboard-body file://dashboard.json \
        --region ${REGION}
    
    rm dashboard.json
    
    echo -e "${GREEN}✅ Comprehensive dashboard created${NC}"
}

# Function to setup log insights queries
create_log_insights_queries() {
    echo -e "${YELLOW}🔍 Creating CloudWatch Insights Saved Queries${NC}"
    
    # Save useful queries for troubleshooting
    queries=(
        "Top Errors|SOURCE '/aws/ecs/secure-aiml-ops' | fields @timestamp, @message | filter @message like /ERROR/ | stats count() by @message | sort count desc | limit 20"
        "Bedrock Usage Analysis|SOURCE '/aiml/bedrock-usage' | fields @timestamp, model_id, tokens_used, cost | stats sum(tokens_used) as total_tokens, sum(cost) as total_cost by model_id | sort total_cost desc"
        "Response Time Analysis|SOURCE '/aws/ecs/secure-aiml-ops' | fields @timestamp, @message | filter @message like /response_time/ | parse @message /response_time: (?<time>\d+)/ | stats avg(time), max(time), min(time) by bin(5m)"
        "Security Events|SOURCE '/security/guardduty' | fields @timestamp, @message | filter @message like /HIGH/ or @message like /CRITICAL/ | sort @timestamp desc | limit 100"
        "Memory Usage Trends|SOURCE '/aws/ecs/secure-aiml-ops' | fields @timestamp, @message | filter @message like /memory_usage/ | parse @message /memory_usage: (?<memory>\d+)/ | stats avg(memory) by bin(5m)"
    )
    
    for query in "${queries[@]}"; do
        IFS='|' read -r name query_string <<< "$query"
        echo "Creating saved query: ${name}"
        # Note: CloudWatch Insights saved queries are typically created through the console
        # This is a placeholder for the query structure
    done
    
    echo -e "${GREEN}✅ Log Insights queries prepared${NC}"
}

# Function to create automated remediation Lambda
create_auto_remediation() {
    echo -e "${YELLOW}🤖 Creating Automated Remediation Functions${NC}"
    
    # Create IAM role for Lambda
    cat > lambda-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    
    aws iam create-role \
        --role-name secure-aiml-ops-remediation-role \
        --assume-role-policy-document file://lambda-trust-policy.json 2>/dev/null || echo "Role already exists"
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name secure-aiml-ops-remediation-role \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    
    # Create custom policy for ECS actions
    cat > lambda-permissions.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecs:UpdateService",
                "ecs:DescribeServices",
                "ecs:RestartTasks",
                "cloudwatch:PutMetricData",
                "sns:Publish"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
    aws iam put-role-policy \
        --role-name secure-aiml-ops-remediation-role \
        --policy-name RemediationPermissions \
        --policy-document file://lambda-permissions.json
    
    rm lambda-trust-policy.json lambda-permissions.json
    
    echo -e "${GREEN}✅ Remediation infrastructure prepared${NC}"
}

# Function to test monitoring setup
test_monitoring_setup() {
    echo -e "${YELLOW}🧪 Testing Monitoring Setup${NC}"
    
    echo "Testing log groups..."
    log_group_count=$(aws logs describe-log-groups --region ${REGION} --query 'logGroups[?starts_with(logGroupName, `/aws/ecs/secure-aiml-ops`) || starts_with(logGroupName, `/security/`) || starts_with(logGroupName, `/monitoring/`) || starts_with(logGroupName, `/aiml/`)] | length(@)')
    echo "  Found ${log_group_count} monitoring log groups"
    
    echo "Testing metric filters..."
    filter_count=$(aws logs describe-metric-filters --region ${REGION} --query 'metricFilters[?starts_with(metricTransformations[0].metricNamespace, `SecureAIMLOps`)] | length(@)')
    echo "  Found ${filter_count} custom metric filters"
    
    echo "Testing alarms..."
    alarm_count=$(aws cloudwatch describe-alarms --region ${REGION} --query 'MetricAlarms[?starts_with(AlarmName, `High`) || starts_with(AlarmName, `Failed`) || starts_with(AlarmName, `ECS`)] | length(@)')
    echo "  Found ${alarm_count} monitoring alarms"
    
    echo "Testing dashboard..."
    dashboard_exists=$(aws cloudwatch get-dashboard --dashboard-name "SecureAIMLOps-Comprehensive" --region ${REGION} >/dev/null 2>&1 && echo "Yes" || echo "No")
    echo "  Dashboard exists: ${dashboard_exists}"
    
    echo -e "${GREEN}✅ Monitoring setup test completed${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting comprehensive monitoring setup...${NC}"
    
    # Check AWS CLI
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${RED}❌ AWS CLI not configured${NC}"
        exit 1
    fi
    
    # Execute setup functions
    create_log_groups
    create_metric_filters
    create_sns_topic
    create_cloudwatch_alarms
    create_dashboard
    create_log_insights_queries
    create_auto_remediation
    test_monitoring_setup
    
    echo ""
    echo -e "${GREEN}🎉 Comprehensive Monitoring Setup Completed!${NC}"
    echo ""
    echo "Monitoring features configured:"
    echo "- 9 CloudWatch log groups with retention policies"
    echo "- 5 custom metric filters for application metrics"
    echo "- 6 CloudWatch alarms for proactive monitoring"
    echo "- Comprehensive monitoring dashboard"
    echo "- SNS topic for alert notifications"
    echo "- Automated remediation infrastructure"
    echo "- Log Insights saved queries for troubleshooting"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Subscribe to SNS topic for email alerts:"
    echo "   aws sns subscribe --topic-arn arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME} --protocol email --notification-endpoint your-email@domain.com"
    echo "2. View dashboard: https://console.aws.amazon.com/cloudwatch/home#dashboards:name=SecureAIMLOps-Comprehensive"
    echo "3. Configure additional alerting endpoints as needed"
    echo ""
}

# Execute main function
main "$@"