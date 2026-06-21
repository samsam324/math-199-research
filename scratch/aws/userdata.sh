#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/user-data.log) 2>&1
dnf -y install python3 python3-pip
mkdir -p /opt/puller
# fetch the puller BEFORE touching python (system aws CLI must stay intact)
aws s3 cp s3://math199-l2-873750256216/_bootstrap/tardis_to_s3.py /opt/puller/tardis_to_s3.py
aws s3 cp s3://math199-l2-873750256216/_bootstrap/universe.txt   /opt/puller/universe.txt
# deps go in an ISOLATED venv so they never clobber the system python the aws CLI uses
python3 -m venv /opt/venv
/opt/venv/bin/pip install --upgrade pip
/opt/venv/bin/pip install boto3 duckdb tardis-dev pandas
cat > /opt/run_pull.sh <<'RUN'
#!/bin/bash
export TARDIS_API_KEY=$(aws ssm get-parameter --name /tardis/api_key --with-decryption --query Parameter.Value --output text --region us-west-2)
cd /opt/puller
/opt/venv/bin/python3 tardis_to_s3.py --bucket math199-l2-873750256216 --symbols-file /opt/puller/universe.txt \
   --from 2023-01-01 --to 2025-12-31 --storage-class STANDARD_IA --region us-west-2 --workers 10 >> /var/log/tardis-pull.log 2>&1
shutdown -h now
RUN
chmod +x /opt/run_pull.sh
cat > /etc/systemd/system/tardis-pull.service <<'UNIT'
[Unit]
Description=Tardis 3yr L2 to S3
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
