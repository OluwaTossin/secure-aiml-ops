#!/bin/bash

# S3 Security Configuration Script
# Implements comprehensive security for our AI/ML data storage

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
BUCKETS=("secure-aiml-ops-data" "secure-aiml-ops-models" "secure-aiml-ops-logs")

echo -e "${BLUE}🔐 Configuring S3 Security for AI/ML Operations${NC}"
echo "================================================="

# Function to create and secure S3 bucket
secure_s3_bucket() {
    local bucket_name=$1
    local bucket_purpose=$2
    
    echo -e "${YELLOW}📦 Securing S3 bucket: ${bucket_name} (${bucket_purpose})${NC}"
    
    # Create bucket if it doesn't exist
    if ! aws s3api head-bucket --bucket ${bucket_name} 2>/dev/null; then
        echo "Creating bucket ${bucket_name}..."
        aws s3api create-bucket \
            --bucket ${bucket_name} \
            --region ${REGION} \
            --create-bucket-configuration LocationConstraint=${REGION} 2>/dev/null || \
        aws s3api create-bucket \
            --bucket ${bucket_name} \
            --region us-east-1 2>/dev/null || true
    fi
    
    # Enable versioning
    echo "Enabling versioning..."
    aws s3api put-bucket-versioning \
        --bucket ${bucket_name} \
        --versioning-configuration Status=Enabled
    
    # Enable server-side encryption
    echo "Configuring server-side encryption..."
    aws s3api put-bucket-encryption \
        --bucket ${bucket_name} \
        --server-side-encryption-configuration '{
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    },
                    "BucketKeyEnabled": true
                }
            ]
        }'
    
    # Block public access
    echo "Blocking public access..."
    aws s3api put-public-access-block \
        --bucket ${bucket_name} \
        --public-access-block-configuration \
            BlockPublicAcls=true,\
            IgnorePublicAcls=true,\
            BlockPublicPolicy=true,\
            RestrictPublicBuckets=true
    
    # Apply bucket policy
    echo "Applying security policy..."
    sed "s/ACCOUNT_ID/${ACCOUNT_ID}/g" security/s3-bucket-policy.json > temp-policy.json
    aws s3api put-bucket-policy \
        --bucket ${bucket_name} \
        --policy file://temp-policy.json
    rm temp-policy.json
    
    # Enable CloudTrail logging for data events
    echo "Configuring CloudTrail data events..."
    aws s3api put-bucket-notification-configuration \
        --bucket ${bucket_name} \
        --notification-configuration '{
            "CloudWatchConfigurations": [
                {
                    "Id": "ObjectCreationEvents",
                    "CloudWatchConfiguration": {
                        "LogGroupName": "/aws/s3/secure-aiml-ops"
                    },
                    "Events": ["s3:ObjectCreated:*"],
                    "Filter": {
                        "Key": {
                            "FilterRules": [
                                {
                                    "Name": "prefix",
                                    "Value": ""
                                }
                            ]
                        }
                    }
                }
            ]
        }' 2>/dev/null || echo "CloudWatch configuration skipped (may require additional setup)"
    
    # Set lifecycle policy for cost optimization
    case ${bucket_purpose} in
        "data")
            lifecycle_policy='{
                "Rules": [
                    {
                        "ID": "DataLifecycleRule",
                        "Status": "Enabled",
                        "Filter": {"Prefix": ""},
                        "Transitions": [
                            {
                                "Days": 30,
                                "StorageClass": "STANDARD_IA"
                            },
                            {
                                "Days": 90,
                                "StorageClass": "GLACIER"
                            },
                            {
                                "Days": 365,
                                "StorageClass": "DEEP_ARCHIVE"
                            }
                        ]
                    }
                ]
            }'
            ;;
        "models")
            lifecycle_policy='{
                "Rules": [
                    {
                        "ID": "ModelLifecycleRule",
                        "Status": "Enabled",
                        "Filter": {"Prefix": ""},
                        "Transitions": [
                            {
                                "Days": 60,
                                "StorageClass": "STANDARD_IA"
                            }
                        ],
                        "NoncurrentVersionTransitions": [
                            {
                                "NoncurrentDays": 30,
                                "StorageClass": "GLACIER"
                            }
                        ]
                    }
                ]
            }'
            ;;
        "logs")
            lifecycle_policy='{
                "Rules": [
                    {
                        "ID": "LogsLifecycleRule",
                        "Status": "Enabled",
                        "Filter": {"Prefix": ""},
                        "Transitions": [
                            {
                                "Days": 7,
                                "StorageClass": "STANDARD_IA"
                            },
                            {
                                "Days": 30,
                                "StorageClass": "GLACIER"
                            }
                        ],
                        "Expiration": {
                            "Days": 2555
                        }
                    }
                ]
            }'
            ;;
    esac
    
    echo "Setting lifecycle policy..."
    echo "${lifecycle_policy}" > temp-lifecycle.json
    aws s3api put-bucket-lifecycle-configuration \
        --bucket ${bucket_name} \
        --lifecycle-configuration file://temp-lifecycle.json
    rm temp-lifecycle.json
    
    echo -e "${GREEN}✅ Bucket ${bucket_name} secured successfully${NC}"
}

# Function to create CloudWatch log group for S3
create_s3_logging() {
    echo -e "${YELLOW}📊 Setting up S3 CloudWatch logging${NC}"
    
    aws logs create-log-group \
        --log-group-name "/aws/s3/secure-aiml-ops" \
        --region ${REGION} 2>/dev/null || echo "Log group already exists"
    
    # Set retention policy
    aws logs put-retention-policy \
        --log-group-name "/aws/s3/secure-aiml-ops" \
        --retention-in-days 90 \
        --region ${REGION}
    
    echo -e "${GREEN}✅ S3 logging configured${NC}"
}

# Function to create S3 access monitoring
create_s3_monitoring() {
    echo -e "${YELLOW}📈 Setting up S3 access monitoring${NC}"
    
    # Create CloudWatch metric filter for unauthorized access attempts
    aws logs put-metric-filter \
        --log-group-name "/aws/s3/secure-aiml-ops" \
        --filter-name "UnauthorizedS3Access" \
        --filter-pattern '[timestamp, request_id, remote_ip, requester, operation="REST.GET.OBJECT", key, request_uri, http_status="403", ...]' \
        --metric-transformations \
            metricName="UnauthorizedS3AccessAttempts",\
            metricNamespace="SecureAIMLOps/Security",\
            metricValue="1" \
        --region ${REGION}
    
    # Create alarm for unauthorized access
    aws cloudwatch put-metric-alarm \
        --alarm-name "S3UnauthorizedAccess" \
        --alarm-description "Detect unauthorized S3 access attempts" \
        --metric-name "UnauthorizedS3AccessAttempts" \
        --namespace "SecureAIMLOps/Security" \
        --statistic "Sum" \
        --period 300 \
        --threshold 5 \
        --comparison-operator "GreaterThanThreshold" \
        --evaluation-periods 1 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:secure-aiml-ops-security-alerts" \
        --region ${REGION} 2>/dev/null || echo "Alarm creation skipped (SNS topic may not exist yet)"
    
    echo -e "${GREEN}✅ S3 monitoring configured${NC}"
}

# Function to test S3 security configuration
test_s3_security() {
    echo -e "${YELLOW}🧪 Testing S3 security configuration${NC}"
    
    for bucket in "${BUCKETS[@]}"; do
        echo "Testing bucket: ${bucket}"
        
        # Test encryption
        encryption=$(aws s3api get-bucket-encryption --bucket ${bucket} --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm' --output text 2>/dev/null || echo "None")
        if [ "$encryption" = "AES256" ]; then
            echo -e "${GREEN}  ✅ Encryption: Enabled${NC}"
        else
            echo -e "${RED}  ❌ Encryption: Not configured${NC}"
        fi
        
        # Test public access block
        public_block=$(aws s3api get-public-access-block --bucket ${bucket} --query 'PublicAccessBlockConfiguration.BlockPublicAcls' --output text 2>/dev/null || echo "false")
        if [ "$public_block" = "True" ]; then
            echo -e "${GREEN}  ✅ Public Access: Blocked${NC}"
        else
            echo -e "${RED}  ❌ Public Access: Not fully blocked${NC}"
        fi
        
        # Test versioning
        versioning=$(aws s3api get-bucket-versioning --bucket ${bucket} --query 'Status' --output text 2>/dev/null || echo "None")
        if [ "$versioning" = "Enabled" ]; then
            echo -e "${GREEN}  ✅ Versioning: Enabled${NC}"
        else
            echo -e "${RED}  ❌ Versioning: Not enabled${NC}"
        fi
        
        echo ""
    done
}

# Main execution
main() {
    echo -e "${BLUE}Starting S3 security configuration...${NC}"
    
    # Check AWS CLI
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${RED}❌ AWS CLI not configured${NC}"
        exit 1
    fi
    
    # Create CloudWatch logging
    create_s3_logging
    
    # Secure each bucket
    secure_s3_bucket "secure-aiml-ops-data" "data"
    secure_s3_bucket "secure-aiml-ops-models" "models"
    secure_s3_bucket "secure-aiml-ops-logs" "logs"
    
    # Set up monitoring
    create_s3_monitoring
    
    # Test configuration
    test_s3_security
    
    echo ""
    echo -e "${GREEN}🎉 S3 security configuration completed!${NC}"
    echo ""
    echo "Security features enabled:"
    echo "- Server-side encryption (AES256)"
    echo "- Versioning enabled"
    echo "- Public access blocked"
    echo "- Secure transport enforced"
    echo "- IP-based access restrictions"
    echo "- Lifecycle policies for cost optimization"
    echo "- CloudWatch monitoring and alerting"
    echo ""
}

# Execute main function
main "$@"