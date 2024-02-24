########## Cloud ######################################################################################################
# Example using AWS SDK (boto3)
import boto3

# Initialize AWS client
ec2 = boto3.client('ec2')

# Example usage: List all EC2 instances
response = ec2.describe_instances()
for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(f"Instance ID: {instance['InstanceId']}, State: {instance['State']['Name']}")

########## AWS ######################################################################################################
# Example using AWS SDK (boto3)
import boto3

# Initialize AWS client for S3
s3 = boto3.client('s3')

# Example usage: List all buckets in S3
response = s3.list_buckets()
buckets = [bucket['Name'] for bucket in response['Buckets']]
print("S3 Buckets:", buckets)


########## Google ######################################################################################################
# Example using Google Cloud SDK (google-cloud-storage)
from google.cloud import storage

# Initialize Google Cloud Storage client
client = storage.Client()

# Example usage: List all buckets in Google Cloud Storage
buckets = list(client.list_buckets())
bucket_names = [bucket.name for bucket in buckets]
print("Google Cloud Storage Buckets:", bucket_names)


########## Deployment ######################################################################################################
# Example using AWS SDK (boto3) for deploying an EC2 instance
import boto3

# Initialize AWS client for EC2
ec2 = boto3.client('ec2')

# Example usage: Launch EC2 instance
response = ec2.run_instances(
    ImageId='ami-12345678',
    InstanceType='t2.micro',
    MaxCount=1,
    MinCount=1
)
instance_id = response['Instances'][0]['InstanceId']
print("Launched EC2 instance with ID:", instance_id)


########## Scalability ######################################################################################################
# Example using AWS SDK (boto3) for autoscaling
import boto3

# Initialize AWS client for Autoscaling
autoscaling = boto3.client('autoscaling')

# Example usage: Create autoscaling group
response = autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='my-autoscaling-group',
    LaunchConfigurationName='my-launch-configuration',
    MinSize=1,
    MaxSize=10,
    DesiredCapacity=2
)
print("Created autoscaling group:", response['AutoScalingGroupARN'])


########## Monitoring ######################################################################################################
# Example using AWS SDK (boto3) for CloudWatch monitoring
import boto3

# Initialize AWS client for CloudWatch
cloudwatch = boto3.client('cloudwatch')

# Example usage: Get metric statistics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/EC2',
    MetricName='CPUUtilization',
    StartTime='2024-02-01T00:00:00Z',
    EndTime='2024-02-02T00:00:00Z',
    Period=3600,
    Statistics=['Average'],
    Dimensions=[{'Name': 'InstanceId', 'Value': 'i-1234567890abcdef0'}]
)
print("CPU Utilization statistics:", response['Datapoints'])


########## Resource Allocation ######################################################################################################
# Example using AWS SDK (boto3) for resource tagging
import boto3

# Initialize AWS client for EC2
ec2 = boto3.client('ec2')

# Example usage: Tag an EC2 instance
instance_id = 'i-1234567890abcdef0'
tags = [{'Key': 'Name', 'Value': 'MyInstance'}, {'Key': 'Environment', 'Value': 'Production'}]
ec2.create_tags(Resources=[instance_id], Tags=tags)
print("Tags created for instance:", instance_id)


########## Monitoring ######################################################################################################
