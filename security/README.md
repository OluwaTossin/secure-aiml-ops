# AWS GuardDuty Setup and Configuration

This directory contains the configuration files and scripts for setting up AWS GuardDuty for our secure AI/ML operations platform.

## Overview

AWS GuardDuty is a threat detection service that continuously monitors for malicious activity and unauthorized behavior to protect AWS accounts, workloads, and data stored in Amazon S3.

## Features Enabled

### 1. **Threat Detection**
- Malware detection for ECS workloads
- Cryptocurrency mining detection
- Suspicious network activity monitoring
- Anomalous API call detection

### 2. **S3 Protection**
- S3 bucket policy violations
- Unusual data access patterns
- Credential compromise detection
- Data exfiltration monitoring

### 3. **ECS Runtime Monitoring**
- Container runtime security
- Process and file system monitoring
- Network traffic analysis
- Malicious container detection

### 4. **Custom Rules**
- AI/ML specific threat patterns
- Financial services compliance monitoring
- Custom IP whitelist/blacklist
- Behavioral anomaly detection

## Quick Setup

1. **Enable GuardDuty**
   ```bash
   aws guardduty create-detector --enable --finding-publishing-frequency FIFTEEN_MINUTES
   ```

2. **Configure S3 Protection**
   ```bash
   aws guardduty create-s3-protection --detector-id <detector-id> --enable
   ```

3. **Enable ECS Runtime Monitoring**
   ```bash
   aws guardduty update-detector --detector-id <detector-id> --features Name=ECS_RUNTIME_MONITORING,Status=ENABLED
   ```

## Automated Setup Script

Use the provided script for complete setup:

```bash
chmod +x setup-guardduty.sh
./setup-guardduty.sh
```

## Monitoring and Alerts

- **CloudWatch Integration**: Automatic log forwarding to CloudWatch
- **SNS Notifications**: Real-time alerts for critical findings
- **Custom Dashboards**: Security metrics visualization
- **Automated Response**: Lambda-based incident response

## Cost Optimization

- **Intelligent Sampling**: Reduces costs while maintaining security coverage
- **Regional Optimization**: Enabled only in active regions (us-east-1, us-west-2)
- **Finding Frequency**: Optimized to 15-minute intervals for cost efficiency
- **Data Retention**: 90-day retention policy for compliance and cost balance

## Compliance

This setup ensures compliance with:
- SOC 2 Type II requirements
- PCI DSS standards
- Financial services regulations
- Enterprise security frameworks

## Integration with Existing Infrastructure

- **ECS Integration**: Seamless monitoring of our Streamlit application
- **S3 Protection**: Monitors our data and model storage buckets
- **VPC Monitoring**: Network traffic analysis within our secure VPC
- **IAM Integration**: Works with our least-privilege access policies