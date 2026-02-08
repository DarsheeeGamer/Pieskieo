# Pieskieo Production Deployment Guide

## Overview

Pieskieo is a production-ready multimodal database with enterprise-grade security, durability, and performance features. This guide covers deployment best practices.

## Quick Start (Production)

### 1. Build with TLS Support

```bash
cargo build --release --features tls -p pieskieo-server
```

Binaries will be in `target/release/`:
- `pieskieo-server` - Main server binary
- `load` - Load testing tool
- `pieskieo` - CLI client

### 2. Production Environment Variables

Create a `.env` file or set environment variables:

```bash
# Data directory (REQUIRED in production)
export PIESKIEO_DATA=/var/lib/pieskieo

# Listen address
export PIESKIEO_LISTEN=0.0.0.0:8443

# TLS Configuration (REQUIRED for production)
export PIESKIEO_TLS_CERT=/etc/pieskieo/cert.pem
export PIESKIEO_TLS_KEY=/etc/pieskieo/key.pem

# Authentication (choose ONE method)
# Method 1: JSON array of users (RECOMMENDED)
export PIESKIEO_USERS='[{"user":"admin","pass":"YourSecureP@ssw0rd!","role":"admin"}]'

# Method 2: Single user via env vars
export PIESKIEO_AUTH_USER=admin
export PIESKIEO_AUTH_PASSWORD='YourSecureP@ssw0rd!'

# Method 3: Bearer token for automation
export PIESKIEO_TOKEN='your-secret-bearer-token-here'

# Shard Configuration
export PIESKIEO_SHARD_TOTAL=4  # More shards = better parallelism

# HNSW Vector Configuration
export PIESKIEO_EF_SEARCH=50           # Query-time accuracy (higher = more accurate, slower)
export PIESKIEO_EF_CONSTRUCTION=200    # Build-time accuracy (higher = better index quality)
export PIESKIEO_LINK_K=4               # Mesh graph connectivity

# Resource Limits
export PIESKIEO_BODY_LIMIT_MB=100      # Max request size
export PIESKIEO_RATE_MAX=300           # Rate limit per IP (requests)
export PIESKIEO_RATE_WINDOW_SECS=60    # Rate limit window

# WAL & Snapshots
export PIESKIEO_WAL_FLUSH_MS=50                # Group commit interval
export PIESKIEO_SNAPSHOT_INTERVAL_SECS=3600    # Auto-snapshot every hour
export PIESKIEO_REBUILD_INTERVAL_SECS=86400    # Rebuild HNSW daily

# Auth Security
export PIESKIEO_AUTH_MAX_FAILURES=5         # Lockout after N failures
export PIESKIEO_AUTH_LOCKOUT_SECS=300       # Lockout duration (5 min)
export PIESKIEO_AUTH_WINDOW_SECS=900        # Failure window (15 min)

# Logging
export PIESKIEO_LOG_MODE=both          # stdout | file | both
export PIESKIEO_LOG_DIR=/var/log/pieskieo
export PIESKIEO_AUDIT_MAX_MB=10        # Audit log rotation size
export RUST_LOG=info,pieskieo=debug    # Log levels
```

### 3. Password Requirements

Production passwords MUST meet these criteria:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special symbol

Example: `S3cure!Pass@2024`

### 4. Start the Server

```bash
# As systemd service (see below)
sudo systemctl start pieskieo

# Or manually
./target/release/pieskieo-server
```

## Systemd Service (Linux)

### Service File

Create `/etc/systemd/system/pieskieo.service`:

```ini
[Unit]
Description=Pieskieo Multimodal Database
After=network.target

[Service]
Type=simple
User=pieskieo
Group=pieskieo
EnvironmentFile=/etc/pieskieo/pieskieo.env
ExecStart=/usr/local/bin/pieskieo-server
Restart=on-failure
RestartSec=5s
LimitNOFILE=65536

# Security hardening
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
NoNewPrivileges=true
ReadWritePaths=/var/lib/pieskieo /var/log/pieskieo

[Install]
WantedBy=multi-user.target
```

### Setup Steps

```bash
# Create user
sudo useradd -r -s /bin/false pieskieo

# Create directories
sudo mkdir -p /var/lib/pieskieo /var/log/pieskieo /etc/pieskieo
sudo chown -R pieskieo:pieskieo /var/lib/pieskieo /var/log/pieskieo
sudo chmod 700 /var/lib/pieskieo /etc/pieskieo

# Install binary
sudo cp target/release/pieskieo-server /usr/local/bin/
sudo chmod 755 /usr/local/bin/pieskieo-server

# Create environment file
sudo tee /etc/pieskieo/pieskieo.env << 'EOF'
PIESKIEO_DATA=/var/lib/pieskieo
PIESKIEO_LISTEN=0.0.0.0:8443
PIESKIEO_TLS_CERT=/etc/pieskieo/cert.pem
PIESKIEO_TLS_KEY=/etc/pieskieo/key.pem
PIESKIEO_USERS=[{"user":"admin","pass":"YourP@ssw0rd","role":"admin"}]
PIESKIEO_SHARD_TOTAL=4
RUST_LOG=info,pieskieo=debug
EOF

sudo chmod 600 /etc/pieskieo/pieskieo.env
sudo chown root:pieskieo /etc/pieskieo/pieskieo.env

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable pieskieo
sudo systemctl start pieskieo
sudo systemctl status pieskieo
```

## TLS Certificate Generation

### Self-Signed (Development)

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout /etc/pieskieo/key.pem \
  -out /etc/pieskieo/cert.pem \
  -days 365 \
  -subj "/CN=pieskieo.local"
```

### Let's Encrypt (Production)

```bash
# Using certbot
sudo certbot certonly --standalone -d db.example.com

# Link certificates
sudo ln -s /etc/letsencrypt/live/db.example.com/fullchain.pem /etc/pieskieo/cert.pem
sudo ln -s /etc/letsencrypt/live/db.example.com/privkey.pem /etc/pieskieo/key.pem
```

## Monitoring & Operations

### Health Check

```bash
curl https://localhost:8443/healthz
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0.2",
  "uptime_seconds": 3600,
  "total_docs": 1000,
  "total_rows": 500,
  "total_vectors": 10000,
  "shard_count": 4,
  "auth_enabled": true
}
```

### Prometheus Metrics

```bash
curl https://localhost:8443/metrics
```

Key metrics:
- `pieskieo_docs` - Total documents
- `pieskieo_rows` - Total rows
- `pieskieo_vectors` - Total vectors
- `pieskieo_rate_rejects` - Rate limit rejections
- `pieskieo_shard_*{shard="N"}` - Per-shard metrics

### Backup

```bash
# Trigger backup (admin only)
curl -X POST https://localhost:8443/v1/admin/backup \
  -H "Authorization: Bearer your-token"

# List backups
curl https://localhost:8443/v1/admin/backups \
  -H "Authorization: Bearer your-token"
```

### Resharding (Live)

```bash
# Reshard to 8 shards (from current 4)
curl -X POST https://localhost:8443/v1/admin/reshard \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"shards": 8}'

# Check status
curl https://localhost:8443/v1/admin/reshard/status \
  -H "Authorization: Bearer your-token"
```

## Security Checklist

- [ ] TLS enabled with valid certificates
- [ ] Strong admin password configured
- [ ] Default admin credentials changed
- [ ] Rate limiting configured appropriately
- [ ] Audit logs enabled and monitored
- [ ] File permissions secured (700 for data dir)
- [ ] Service running as non-root user
- [ ] Firewall configured (allow only 8443)
- [ ] Regular backups scheduled
- [ ] Monitoring alerts configured

## Performance Tuning

### Shard Count

Rule of thumb: `SHARD_TOTAL = number_of_CPU_cores / 2`

More shards = better parallelism, but more overhead for small datasets.

### HNSW Parameters

**For accuracy:**
- `EF_CONSTRUCTION=400`
- `EF_SEARCH=100`

**For speed:**
- `EF_CONSTRUCTION=100`
- `EF_SEARCH=20`

### WAL Flush Interval

- Higher `WAL_FLUSH_MS` = better write throughput, slightly higher latency
- Lower `WAL_FLUSH_MS` = better write latency, more CPU overhead
- Default 50ms is balanced for most workloads

## Troubleshooting

### Check Logs

```bash
# Systemd journal
sudo journalctl -u pieskieo -f

# Application logs
tail -f /var/log/pieskieo/pieskieo.log

# Audit logs
tail -f /var/lib/pieskieo/logs/audit.log
```

### Common Issues

**"Failed to bind address"**
- Port already in use: `sudo lsof -i :8443`
- Permission denied: Use port ≥1024 or grant CAP_NET_BIND_SERVICE

**"TLS certificate error"**
- Check file paths and permissions
- Verify certificate format (PEM)
- Ensure key is unencrypted

**"Auth failure loop"**
- Check password meets complexity requirements
- Verify user exists: `grep user /etc/pieskieo/pieskieo.env`
- Check account lockout status in logs

## High Availability Setup

### Master-Replica Configuration

**Master:**
```bash
export PIESKIEO_DATA=/var/lib/pieskieo/master
export PIESKIEO_LISTEN=0.0.0.0:8443
# Standard config...
```

**Replica:**
```bash
# Use pieskieo CLI to tail WAL from master
pieskieo follow \
  --leader https://master:8443 \
  --follower https://replica:8443 \
  --leader-token master-token \
  --follower-token replica-token \
  --interval 2
```

## API Usage Examples

### Create User (Admin)

```bash
curl -X POST https://localhost:8443/v1/auth/users \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "user": "analyst",
    "pass": "Secure!Pass123",
    "role": "read"
  }'
```

### Vector Search

```bash
curl -X POST https://localhost:8443/v1/vector/search \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, ...],
    "k": 10,
    "filter_meta": {"category": "books"}
  }'
```

### PQL Query

```bash
curl -X POST https://localhost:8443/v1/sql \
  -u admin:password \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT id, name FROM users WHERE age > 25 ORDER BY name"
  }'
```

## Support

- GitHub Issues: https://github.com/DarsheeeGamer/Pieskieo/issues
- Documentation: See README.md and PLAN.md
- License: GPL-2.0-only

---

**Production Checklist Summary:**

✅ TLS enabled  
✅ Strong passwords  
✅ Regular backups  
✅ Monitoring enabled  
✅ Audit logs configured  
✅ Rate limiting active  
✅ Graceful shutdown tested  
✅ Security hardening applied  

**You're ready for production!** 🚀
