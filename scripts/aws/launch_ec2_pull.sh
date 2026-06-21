#!/usr/bin/env bash
# Launch a throwaway EC2 box that pulls 3 years of Binance-spot L2 from Tardis straight
# to S3 as Hive-partitioned parquet, then self-terminates. Streams (download->convert->
# upload->delete) so a 30 GB disk suffices. Idempotent: re-running skips what's in S3.
#
# Cost: ~m7g.large $0.0816/hr + tiny gp3; transfer free; ~$8-15 for a 2-4 day pull.
# Prereq: AWS CLI authed, .env has TARDIS_API_KEY, the L2 bucket already exists.
set -euo pipefail

REGION=us-west-2
ACCOUNT=873750256216
BUCKET=math199-l2-873750256216
INSTANCE_TYPE=m7g.large
ROLE=tardis-puller-role
PROFILE=tardis-puller-profile
POLICY=tardis-puller-policy
KEY_PARAM=/tardis/api_key
FROM=2023-01-01
TO=2025-12-31
HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root

echo "== 1. Tardis key -> SSM SecureString =="
TKEY=$(grep '^TARDIS_API_KEY=' "$HERE/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
aws ssm put-parameter --name "$KEY_PARAM" --type SecureString --value "$TKEY" \
  --overwrite --region "$REGION" >/dev/null && echo "  ok"

echo "== 2. IAM role + least-privilege policy + instance profile =="
cat > /tmp/trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
cat > /tmp/policy.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Sid":"List","Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::$BUCKET"},
 {"Sid":"RW","Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
 {"Sid":"Key","Effect":"Allow","Action":"ssm:GetParameter","Resource":"arn:aws:ssm:$REGION:$ACCOUNT:parameter$KEY_PARAM"}
]}
JSON
aws iam create-role --role-name "$ROLE" --assume-role-policy-document file:///tmp/trust.json 2>/dev/null || echo "  role exists"
aws iam put-role-policy --role-name "$ROLE" --policy-name "$POLICY" --policy-document file:///tmp/policy.json
aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
aws iam create-instance-profile --instance-profile-name "$PROFILE" 2>/dev/null || echo "  profile exists"
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE" 2>/dev/null || true
sleep 10  # let the instance profile propagate

echo "== 3. bootstrap (puller + universe) -> s3://$BUCKET/_bootstrap/ =="
aws s3 cp "$HERE/scripts/tardis_to_s3.py" "s3://$BUCKET/_bootstrap/tardis_to_s3.py" --only-show-errors
aws s3 cp "$HERE/data/l2_universe_top50.txt" "s3://$BUCKET/_bootstrap/universe.txt" --only-show-errors
echo "  ok"

echo "== 4. render user-data =="
cat > /tmp/user-data.sh <<UD
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/user-data.log) 2>&1
dnf -y install python3 python3-pip
pip3 install --upgrade boto3 duckdb tardis-dev pandas pyarrow
mkdir -p /opt/puller
aws s3 cp s3://$BUCKET/_bootstrap/tardis_to_s3.py /opt/puller/tardis_to_s3.py
aws s3 cp s3://$BUCKET/_bootstrap/universe.txt   /opt/puller/universe.txt
KEY=\$(aws ssm get-parameter --name $KEY_PARAM --with-decryption --query Parameter.Value --output text --region $REGION)
cat > /opt/run_pull.sh <<RUN
#!/bin/bash
export TARDIS_API_KEY="\$KEY"
cd /opt/puller
python3 tardis_to_s3.py --bucket $BUCKET --symbols-file /opt/puller/universe.txt \\
   --from $FROM --to $TO --storage-class STANDARD_IA --region $REGION >> /var/log/tardis-pull.log 2>&1
shutdown -h now
RUN
chmod +x /opt/run_pull.sh
cat > /etc/systemd/system/tardis-pull.service <<UNIT
[Unit]
Description=Tardis 3yr L2 -> S3
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
ExecStart=/opt/run_pull.sh
Restart=on-failure
RestartSec=30
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now tardis-pull.service
UD

echo "== 5. launch m7g.large (AL2023 arm64, 30GB gp3, IMDSv2, self-terminate) =="
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --query Parameter.Value --output text --region "$REGION")
IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile Name="$PROFILE" --user-data file:///tmp/user-data.sh \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --metadata-options '{"HttpTokens":"required","HttpEndpoint":"enabled"}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tardis-puller}]' \
  --count 1 --query 'Instances[0].InstanceId' --output text)
echo "  launched $IID"
echo
echo "monitor:   aws ssm start-session --target $IID --region $REGION   then: tail -f /var/log/tardis-pull.log"
echo "progress:  aws s3 ls --recursive --summarize s3://$BUCKET/ | tail -2"
echo "terminate: aws ec2 terminate-instances --instance-ids $IID --region $REGION   (auto-terminates when done)"
