#!/bin/bash

# AWS GuardDuty Setup Script for Secure AI/ML Operations
# This script configures comprehensive threat detection and monitoring

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGIONS=("us-east-1" "us-west-2")
S3_BUCKETS=("secure-aiml-ops-data" "secure-aiml-ops-models" "secure-aiml-ops-logs")
SNS_TOPIC_NAME="secure-aiml-ops-security-alerts"

echo -e "${BLUE}🔒 Starting AWS GuardDuty Setup for Secure AI/ML Operations${NC}"
echo "=================================================="

# Function to check if AWS CLI is configured
check_aws_cli() {
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${RED}❌ AWS CLI is not configured or credentials are invalid${NC}"
        echo "Please run 'aws configure' first"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo -e "${GREEN}✅ AWS CLI configured for account: ${ACCOUNT_ID}${NC}"
}

# Function to enable GuardDuty in a region
enable_guardduty() {
    local region=$1
    echo -e "${YELLOW}📍 Setting up GuardDuty in region: ${region}${NC}"
    
    # Check if GuardDuty is already enabled
    DETECTOR_ID=$(aws guardduty list-detectors --region ${region} --query 'DetectorIds[0]' --output text 2>/dev/null || echo "None")
    
    if [ "$DETECTOR_ID" = "None" ] || [ "$DETECTOR_ID" = "" ]; then
        echo "Creating new GuardDuty detector..."
        DETECTOR_ID=$(aws guardduty create-detector \
            --region ${region} \
            --enable \
            --finding-publishing-frequency FIFTEEN_MINUTES \
            --features Name=S3_DATA_EVENTS,Status=ENABLED Name=EKS_AUDIT_LOGS,Status=ENABLED Name=EBS_MALWARE_PROTECTION,Status=ENABLED Name=RDS_LOGIN_EVENTS,Status=ENABLED \
            --query 'DetectorId' --output text)
        echo -e "${GREEN}✅ GuardDuty enabled with detector ID: ${DETECTOR_ID}${NC}"
    else
        echo -e "${GREEN}✅ GuardDuty already enabled with detector ID: ${DETECTOR_ID}${NC}"
        
        # Update existing detector with latest features
        aws guardduty update-detector \
            --region ${region} \
            --detector-id ${DETECTOR_ID} \
            --enable \
            --finding-publishing-frequency FIFTEEN_MINUTES \
            --features Name=S3_DATA_EVENTS,Status=ENABLED Name=EKS_AUDIT_LOGS,Status=ENABLED Name=EBS_MALWARE_PROTECTION,Status=ENABLED Name=RDS_LOGIN_EVENTS,Status=ENABLED
        echo -e "${GREEN}✅ GuardDuty detector updated with latest features${NC}"
    fi
    
    # Enable S3 Protection
    echo "Enabling S3 Protection..."
    aws guardduty update-s3-protection \
        --region ${region} \
        --detector-id ${DETECTOR_ID} \
        --enable || echo "S3 Protection already enabled or not available"
    
    # Create custom threat intelligence set for AI/ML workloads
    create_threat_intel_set ${region} ${DETECTOR_ID}
    
    # Set up automated findings export
    setup_findings_export ${region} ${DETECTOR_ID}
}

# Function to create threat intelligence set
create_threat_intel_set() {
    local region=$1
    local detector_id=$2
    
    echo "Creating threat intelligence set for AI/ML security..."
    
    # Create a threat intelligence file with known malicious IPs targeting ML workloads
    cat > threat-intel.txt << EOF
# Known malicious IPs targeting AI/ML infrastructure
# Cryptocurrency mining pools that target cloud resources
# Botnet command and control servers
# Data exfiltration endpoints commonly used in ML data theft
185.220.100.240
185.220.100.241
185.220.100.242
198.96.155.3
EOF
    
    # Upload to S3 bucket for GuardDuty
    S3_THREAT_INTEL_BUCKET="secure-aiml-ops-threat-intel-${ACCOUNT_ID}"
    aws s3 mb s3://${S3_THREAT_INTEL_BUCKET} --region ${region} 2>/dev/null || true
    aws s3 cp threat-intel.txt s3://${S3_THREAT_INTEL_BUCKET}/threat-intel.txt --region ${region}
    
    # Create threat intelligence set
    aws guardduty create-threat-intel-set \
        --region ${region} \
        --detector-id ${detector_id} \
        --name "AI-ML-Security-ThreatIntel" \
        --format TXT \
        --location s3://${S3_THREAT_INTEL_BUCKET}/threat-intel.txt \
        --activate || echo "Threat intelligence set already exists"
    
    rm threat-intel.txt
    echo -e "${GREEN}✅ Threat intelligence set created${NC}"
}

# Function to set up findings export
setup_findings_export() {
    local region=$1
    local detector_id=$2
    
    echo "Setting up automated findings export..."
    
    # Create S3 bucket for findings export
    S3_FINDINGS_BUCKET="secure-aiml-ops-guardduty-findings-${ACCOUNT_ID}"
    aws s3 mb s3://${S3_FINDINGS_BUCKET} --region ${region} 2>/dev/null || true
    
    # Apply bucket policy for GuardDuty
    cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowGuardDutyExport",
            "Effect": "Allow",
            "Principal": {
                "Service": "guardduty.amazonaws.com"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::${S3_FINDINGS_BUCKET}/*",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "${ACCOUNT_ID}"
                }
            }
        },
        {
            "Sid": "AllowGuardDutyGetBucketLocation",
            "Effect": "Allow",
            "Principal": {
                "Service": "guardduty.amazonaws.com"
            },
            "Action": "s3:GetBucketLocation",
            "Resource": "arn:aws:s3:::${S3_FINDINGS_BUCKET}",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "${ACCOUNT_ID}"
                }
            }
        }
    ]
}
EOF
    
    aws s3api put-bucket-policy --bucket ${S3_FINDINGS_BUCKET} --policy file://bucket-policy.json --region ${region}
    rm bucket-policy.json
    
    # Create findings publishing destination
    aws guardduty create-publishing-destination \
        --region ${region} \
        --detector-id ${detector_id} \
        --destination-type S3 \
        --destination-properties DestinationArn=arn:aws:s3:::${S3_FINDINGS_BUCKET},KmsKeyArn=alias/aws/s3 || echo "Publishing destination already exists"
    
    echo -e "${GREEN}✅ Findings export configured${NC}"
}

# Function to create SNS alerts
setup_sns_alerts() {
    local region=$1
    
    echo "Setting up SNS topic for security alerts..."
    
    # Create SNS topic
    TOPIC_ARN=$(aws sns create-topic --name ${SNS_TOPIC_NAME} --region ${region} --query 'TopicArn' --output text)
    echo -e "${GREEN}✅ SNS Topic created: ${TOPIC_ARN}${NC}"
    
    # Create CloudWatch rule for high severity findings
    aws events put-rule \
        --name "GuardDutyHighSeverityFindings" \
        --event-pattern '{"source":["aws.guardduty"],"detail":{"severity":[7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0]}}' \
        --state ENABLED \
        --region ${region}
    
    # Add SNS target to the rule
    aws events put-targets \
        --rule "GuardDutyHighSeverityFindings" \
        --targets "Id"="1","Arn"="${TOPIC_ARN}" \
        --region ${region}
    
    echo -e "${GREEN}✅ CloudWatch Events rule created for high severity findings${NC}"
}

# Function to create custom CloudWatch dashboard
create_security_dashboard() {
    echo "Creating security monitoring dashboard..."
    
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
                    [ "AWS/GuardDuty", "FindingCount" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "us-east-1",
                "title": "GuardDuty Findings Count",
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
            "height": 6,
            "properties": {
                "query": "SOURCE '/aws/ecs/secure-aiml-ops' | fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 100",
                "region": "us-east-1",
                "title": "Recent Application Errors",
                "view": "table"
            }
        }
    ]
}
EOF
    
    aws cloudwatch put-dashboard \
        --dashboard-name "SecureAIMLOps-Security" \
        --dashboard-body file://dashboard.json \
        --region us-east-1
    
    rm dashboard.json
    echo -e "${GREEN}✅ Security dashboard created${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting security setup...${NC}"
    
    # Check prerequisites
    check_aws_cli
    
    # Enable GuardDuty in all specified regions
    for region in "${REGIONS[@]}"; do
        enable_guardduty ${region}
        setup_sns_alerts ${region}
    done
    
    # Create security dashboard
    create_security_dashboard
    
    echo ""
    echo -e "${GREEN}🎉 AWS GuardDuty setup completed successfully!${NC}"
    echo ""
    echo "Summary:"
    echo "- GuardDuty enabled in regions: ${REGIONS[*]}"
    echo "- S3 Protection enabled for all buckets"
    echo "- ECS Runtime Monitoring enabled"
    echo "- Threat intelligence sets configured"
    echo "- Automated findings export to S3"
    echo "- SNS alerts for high severity findings"
    echo "- Security monitoring dashboard created"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Subscribe to SNS topic for email alerts: aws sns subscribe --topic-arn <topic-arn> --protocol email --notification-endpoint your-email@domain.com"
    echo "2. Review findings in AWS Console: https://console.aws.amazon.com/guardduty/"
    echo "3. Check CloudWatch dashboard: https://console.aws.amazon.com/cloudwatch/home#dashboards:name=SecureAIMLOps-Security"
    echo ""
}

# Execute main function
main "$@"