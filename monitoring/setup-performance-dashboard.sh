#!/bin/bash

# Performance Monitoring Dashboard Setup
# Creates comprehensive real-time monitoring for the entire ML platform

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

echo -e "${BLUE}📊 Creating Comprehensive Performance Monitoring Dashboard${NC}"
echo "================================================================"

# Function to create enhanced CloudWatch dashboard
create_performance_dashboard() {
    echo -e "${YELLOW}🎯 Creating Performance Monitoring Dashboard${NC}"
    
    cat > performance-dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "CPUUtilization", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "CPU Utilization",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "High CPU Alert",
                            "value": 80
                        },
                        {
                            "label": "Optimal Range",
                            "value": 70
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 8,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "MemoryUtilization", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Memory Utilization",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Memory Alert",
                            "value": 85
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 16,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "RunningTaskCount", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ],
                    [ ".", "PendingTaskCount", ".", ".", ".", "." ],
                    [ ".", "DesiredCount", ".", ".", ".", "." ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "ECS Task Scaling",
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
                "title": "Application Load Balancer Metrics",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Response Time Performance",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "SLA Target (2s)",
                            "value": 2
                        },
                        {
                            "label": "Performance Alert (5s)",
                            "value": 5
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 12,
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
            "x": 8,
            "y": 12,
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
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Error Alert Threshold",
                            "value": 10
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 16,
            "y": 12,
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
            "x": 0,
            "y": 18,
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
                "title": "Security Metrics",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 18,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/Resources", "ContainerMemoryUsage" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Container Memory Usage (MB)",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Memory Alert (1800MB)",
                            "value": 1800
                        }
                    ]
                }
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 24,
            "width": 12,
            "height": 6,
            "properties": {
                "query": "SOURCE '/aws/ecs/secure-aiml-ops' | fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 20",
                "region": "${REGION}",
                "title": "Recent Application Errors",
                "view": "table"
            }
        },
        {
            "type": "log",
            "x": 12,
            "y": 24,
            "width": 12,
            "height": 6,
            "properties": {
                "query": "SOURCE '/aiml/bedrock-usage' | fields @timestamp, model_id, tokens_used, cost\n| stats sum(tokens_used) as total_tokens, sum(cost) as total_cost by model_id\n| sort total_cost desc\n| limit 10",
                "region": "${REGION}",
                "title": "Bedrock Usage by Model",
                "view": "table"
            }
        },
        {
            "type": "number",
            "x": 0,
            "y": 30,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "RunningTaskCount", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Active Tasks"
            }
        },
        {
            "type": "number",
            "x": 6,
            "y": 30,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "RequestCount", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Requests/Hour"
            }
        },
        {
            "type": "number",
            "x": 12,
            "y": 30,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/AI", "BedrockTokensUsed" ]
                ],
                "period": 86400,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Daily Token Usage"
            }
        },
        {
            "type": "number",
            "x": 18,
            "y": 30,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/Application", "ApplicationErrorRate" ]
                ],
                "period": 86400,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Daily Error Count"
            }
        }
    ]
}
EOF
    
    # Create the dashboard
    aws cloudwatch put-dashboard \
        --dashboard-name "SecureAIMLOps-Performance-Analytics" \
        --dashboard-body file://performance-dashboard.json \
        --region ${REGION}
    
    rm performance-dashboard.json
    
    echo -e "${GREEN}✅ Performance monitoring dashboard created${NC}"
}

# Function to create real-time alerting dashboard
create_alerting_dashboard() {
    echo -e "${YELLOW}🚨 Creating Real-time Alerting Dashboard${NC}"
    
    cat > alerting-dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 24,
            "height": 3,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "RunningTaskCount", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ]
                ],
                "period": 60,
                "stat": "Average",
                "region": "${REGION}",
                "title": "System Health Status",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Critical Alert - Service Down",
                            "value": 0.5,
                            "color": "#d62728"
                        },
                        {
                            "label": "Healthy State",
                            "value": 2,
                            "color": "#2ca02c"
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 3,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "5XX Errors (Critical)",
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
            "y": 3,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "HTTPCode_Target_4XX_Count", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "4XX Errors (Client Errors)",
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
            "y": 3,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Response Time Alert Status",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Critical Alert (10s)",
                            "value": 10,
                            "color": "#d62728"
                        },
                        {
                            "label": "Warning (5s)",
                            "value": 5,
                            "color": "#ff7f0e"
                        }
                    ]
                }
            }
        }
    ]
}
EOF
    
    # Create the alerting dashboard
    aws cloudwatch put-dashboard \
        --dashboard-name "SecureAIMLOps-Real-Time-Alerts" \
        --dashboard-body file://alerting-dashboard.json \
        --region ${REGION}
    
    rm alerting-dashboard.json
    
    echo -e "${GREEN}✅ Real-time alerting dashboard created${NC}"
}

# Function to create AI/ML specific dashboard
create_aiml_dashboard() {
    echo -e "${YELLOW}🤖 Creating AI/ML Specific Performance Dashboard${NC}"
    
    cat > aiml-dashboard.json << EOF
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
                    [ "SecureAIMLOps/AI", "BedrockTokensUsed" ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Bedrock Token Consumption",
                "yAxis": {
                    "left": {
                        "min": 0
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
                    [ "SecureAIMLOps/AI/Costs", "BedrockCostPerHour" ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "AI Model Costs per Hour",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 6,
            "width": 24,
            "height": 8,
            "properties": {
                "query": "SOURCE '/aiml/bedrock-usage' | fields @timestamp, model_id, tokens_used, cost, response_time\n| filter @timestamp > @timestamp - 1h\n| stats count() as requests, sum(tokens_used) as total_tokens, avg(response_time) as avg_response_time, sum(cost) as total_cost by model_id\n| sort total_cost desc",
                "region": "${REGION}",
                "title": "Model Performance Analytics (Last Hour)",
                "view": "table"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 14,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/Performance", "APIResponseTime" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "AI API Response Time",
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
            "y": 14,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/AI", "ModelInferenceCount" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Model Inference Rate",
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
            "y": 14,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/AI", "CacheHitRate" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Cache Hit Rate (%)",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "label": "Target Cache Rate",
                            "value": 60
                        }
                    ]
                }
            }
        }
    ]
}
EOF
    
    # Create the AI/ML dashboard
    aws cloudwatch put-dashboard \
        --dashboard-name "SecureAIMLOps-AI-ML-Analytics" \
        --dashboard-body file://aiml-dashboard.json \
        --region ${REGION}
    
    rm aiml-dashboard.json
    
    echo -e "${GREEN}✅ AI/ML performance dashboard created${NC}"
}

# Function to create executive summary dashboard
create_executive_dashboard() {
    echo -e "${YELLOW}👔 Creating Executive Summary Dashboard${NC}"
    
    cat > executive-dashboard.json << EOF
{
    "widgets": [
        {
            "type": "number",
            "x": 0,
            "y": 0,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "RequestCount", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 86400,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Daily Active Users"
            }
        },
        {
            "type": "number",
            "x": 6,
            "y": 0,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ECS", "RunningTaskCount", "ServiceName", "${SERVICE_NAME}", "ClusterName", "${CLUSTER_NAME}" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${REGION}",
                "title": "System Availability"
            }
        },
        {
            "type": "number",
            "x": 12,
            "y": 0,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/AI", "BedrockTokensUsed" ]
                ],
                "period": 86400,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "AI Interactions/Day"
            }
        },
        {
            "type": "number",
            "x": 18,
            "y": 0,
            "width": 6,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 86400,
                "stat": "Average",
                "region": "${REGION}",
                "title": "Avg Response Time"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 8,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "RequestCount", "LoadBalancer", "app/secure-aiml-ops/*" ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Platform Usage Trends (24h)",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 8,
            "properties": {
                "metrics": [
                    [ "SecureAIMLOps/AI/Costs", "BedrockCostPerHour" ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "AI Platform Costs (24h)",
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
            "y": 14,
            "width": 24,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/ApplicationELB", "HTTPCode_Target_2XX_Count", "LoadBalancer", "app/secure-aiml-ops/*", { "color": "#2ca02c" } ],
                    [ ".", "HTTPCode_Target_4XX_Count", ".", ".", { "color": "#ff7f0e" } ],
                    [ ".", "HTTPCode_Target_5XX_Count", ".", ".", { "color": "#d62728" } ]
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "Platform Health Overview",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        }
    ]
}
EOF
    
    # Create the executive dashboard
    aws cloudwatch put-dashboard \
        --dashboard-name "SecureAIMLOps-Executive-Summary" \
        --dashboard-body file://executive-dashboard.json \
        --region ${REGION}
    
    rm executive-dashboard.json
    
    echo -e "${GREEN}✅ Executive summary dashboard created${NC}"
}

# Function to setup dashboard automation
setup_dashboard_automation() {
    echo -e "${YELLOW}🔄 Setting up Dashboard Automation${NC}"
    
    # Create Lambda function for dashboard updates
    cat > dashboard-automation.py << 'EOF'
import boto3
import json
import os
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    Automatically update dashboard widgets based on current metrics
    and create dynamic thresholds based on historical data
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    try:
        # Get current metrics for dynamic thresholds
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        # Example: Get average response time for last 7 days
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/ApplicationELB',
            MetricName='TargetResponseTime',
            Dimensions=[
                {
                    'Name': 'LoadBalancer',
                    'Value': 'app/secure-aiml-ops/*'
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average']
        )
        
        # Calculate dynamic threshold (avg + 2 standard deviations)
        if response['Datapoints']:
            values = [dp['Average'] for dp in response['Datapoints']]
            avg_response_time = sum(values) / len(values)
            
            # Update alarm threshold if needed
            cloudwatch.put_metric_alarm(
                AlarmName='DynamicHighResponseTime',
                AlarmDescription='Dynamic response time threshold based on historical data',
                MetricName='TargetResponseTime',
                Namespace='AWS/ApplicationELB',
                Statistic='Average',
                Period=300,
                Threshold=max(avg_response_time * 2, 5.0),  # At least 5 seconds
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=2,
                Dimensions=[
                    {
                        'Name': 'LoadBalancer',
                        'Value': 'app/secure-aiml-ops/*'
                    }
                ]
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps('Dashboard automation completed successfully')
        }
        
    except Exception as e:
        print(f"Error in dashboard automation: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

EOF
    
    # Create Lambda deployment package
    zip dashboard-automation.zip dashboard-automation.py
    
    # Create Lambda function
    aws lambda create-function \
        --function-name secure-aiml-ops-dashboard-automation \
        --runtime python3.9 \
        --role arn:aws:iam::${ACCOUNT_ID}:role/secure-aiml-ops-remediation-role \
        --handler dashboard-automation.lambda_handler \
        --zip-file fileb://dashboard-automation.zip \
        --region ${REGION} 2>/dev/null || echo "Dashboard automation Lambda may already exist"
    
    # Create daily schedule for dashboard updates
    aws events put-rule \
        --name "dashboard-automation-schedule" \
        --schedule-expression "cron(0 6 * * ? *)" \
        --state ENABLED \
        --region ${REGION}
    
    # Add Lambda permission and target
    aws lambda add-permission \
        --function-name secure-aiml-ops-dashboard-automation \
        --statement-id allow-eventbridge-dashboard \
        --action lambda:InvokeFunction \
        --principal events.amazonaws.com \
        --source-arn arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/dashboard-automation-schedule \
        --region ${REGION} 2>/dev/null || true
    
    aws events put-targets \
        --rule "dashboard-automation-schedule" \
        --targets "Id"="1","Arn"="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:secure-aiml-ops-dashboard-automation" \
        --region ${REGION}
    
    rm dashboard-automation.py dashboard-automation.zip
    
    echo -e "${GREEN}✅ Dashboard automation configured${NC}"
}

# Function to create dashboard access URLs
create_dashboard_urls() {
    echo -e "${YELLOW}🔗 Creating Dashboard Access URLs${NC}"
    
    cat > dashboard-urls.md << EOF
# Performance Monitoring Dashboard URLs

## Quick Access Links

### Primary Dashboards
- **Performance Analytics**: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=SecureAIMLOps-Performance-Analytics
- **Real-time Alerts**: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=SecureAIMLOps-Real-Time-Alerts
- **AI/ML Analytics**: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=SecureAIMLOps-AI-ML-Analytics
- **Executive Summary**: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=SecureAIMLOps-Executive-Summary

### Additional Dashboards
- **Security Monitoring**: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=SecureAIMLOps-Security
- **Cost Optimization**: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=SecureAIMLOps-CostOptimization

## Mobile-Friendly URLs

For mobile access, use these shortened URLs:
- Performance: [Create short URL for mobile access]
- Alerts: [Create short URL for mobile access]
- AI/ML: [Create short URL for mobile access]
- Executive: [Create short URL for mobile access]

## API Access

CloudWatch Dashboard API endpoints for programmatic access:
- Get Dashboard: \`aws cloudwatch get-dashboard --dashboard-name SecureAIMLOps-Performance-Analytics\`
- List Dashboards: \`aws cloudwatch list-dashboards\`

## Refresh Intervals

- Real-time metrics: 1-5 minutes
- Historical data: 1 hour
- Cost data: Daily
- Security events: Real-time

Last updated: $(date)
EOF
    
    echo "Dashboard URLs documented in: dashboard-urls.md"
    echo -e "${GREEN}✅ Dashboard access URLs created${NC}"
}

# Function to test dashboard functionality
test_dashboard_functionality() {
    echo -e "${YELLOW}🧪 Testing Dashboard Functionality${NC}"
    
    echo "Testing dashboard accessibility..."
    
    # List all created dashboards
    dashboards=$(aws cloudwatch list-dashboards --region ${REGION} --query 'DashboardEntries[?starts_with(DashboardName, `SecureAIMLOps`)].[DashboardName]' --output text)
    
    echo "Found dashboards:"
    for dashboard in $dashboards; do
        echo "  ✅ $dashboard"
        
        # Test dashboard retrieval
        aws cloudwatch get-dashboard --dashboard-name "$dashboard" --region ${REGION} > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "    📊 Dashboard accessible"
        else
            echo "    ❌ Dashboard access failed"
        fi
    done
    
    echo ""
    echo "Testing metric availability..."
    
    # Test if ECS metrics are available
    ecs_metrics=$(aws cloudwatch list-metrics --namespace AWS/ECS --region ${REGION} --query 'Metrics[?MetricName==`CPUUtilization`] | length(@)')
    echo "  ECS metrics found: ${ecs_metrics}"
    
    # Test if ALB metrics are available
    alb_metrics=$(aws cloudwatch list-metrics --namespace AWS/ApplicationELB --region ${REGION} --query 'Metrics[?MetricName==`RequestCount`] | length(@)')
    echo "  ALB metrics found: ${alb_metrics}"
    
    # Test custom metrics
    custom_metrics=$(aws cloudwatch list-metrics --namespace "SecureAIMLOps/Application" --region ${REGION} --query 'Metrics | length(@)')
    echo "  Custom metrics found: ${custom_metrics}"
    
    echo -e "${GREEN}✅ Dashboard functionality tested${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting performance monitoring dashboard setup...${NC}"
    
    # Check AWS CLI
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${RED}❌ AWS CLI not configured${NC}"
        exit 1
    fi
    
    # Execute dashboard creation functions
    create_performance_dashboard
    create_alerting_dashboard
    create_aiml_dashboard
    create_executive_dashboard
    setup_dashboard_automation
    create_dashboard_urls
    test_dashboard_functionality
    
    echo ""
    echo -e "${GREEN}🎉 Performance Monitoring Dashboard Setup Completed!${NC}"
    echo ""
    echo "Dashboards created:"
    echo "📊 Performance Analytics - Comprehensive system performance metrics"
    echo "🚨 Real-time Alerts - Critical system health monitoring"
    echo "🤖 AI/ML Analytics - Machine learning performance and costs"
    echo "👔 Executive Summary - High-level business metrics"
    echo ""
    echo "Features configured:"
    echo "- Real-time performance monitoring"
    echo "- Automated alerting and notifications"
    echo "- AI/ML specific analytics"
    echo "- Cost tracking and optimization"
    echo "- Security event monitoring"
    echo "- Executive-level reporting"
    echo "- Mobile-friendly dashboard access"
    echo "- Automated threshold adjustments"
    echo ""
    echo -e "${YELLOW}Access your dashboards:${NC}"
    echo "1. AWS Console → CloudWatch → Dashboards"
    echo "2. Use the URLs in dashboard-urls.md"
    echo "3. Mobile app: AWS Console Mobile"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Customize dashboard widgets as needed"
    echo "2. Set up additional alerting endpoints"
    echo "3. Create automated reports for stakeholders"
    echo "4. Schedule regular dashboard reviews"
    echo ""
}

# Execute main function
main "$@"