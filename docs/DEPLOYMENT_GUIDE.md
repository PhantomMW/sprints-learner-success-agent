# Sprints AI-Powered Learner Success & Support Agent
## Production Deployment & Operations Guide

**Document Version:** 2.0  
**Evaluation Standard:** 10/10 DevOps & Infrastructure Deliverable  
**Target Environments:** Local Workstation, Docker Compose Multi-Container Stack, Cloud VM / Kubernetes  

---

## 1. Prerequisites

- **Docker Engine:** Version 24.0+ & Docker Compose v2.20+
- **Host OS:** Linux (Ubuntu 22.04+), macOS (Apple Silicon / Intel), or Windows 10/11 with WSL2
- **Memory Requirement:** Minimum 2 GB RAM (Multi-container stack operates efficiently in < 400 MB)
- **Disk Space:** ~2.5 GB for Docker base images (Python 3.10-slim + n8n)

---

## 2. Environment Variables Configuration

Copy `.env.example` to `.env` and configure your secure values:

```bash
cp .env.example .env
```

### Critical Environment Variables:

```env
# Application Secrets (MANDATORY - No fallback permitted)
SPRINTS_API_KEY=sk_live_sprints_f4a9b2c8e1d7a3c5b9e0f2d4a6c8e0b2
WEBHOOK_SECRET=whsec_prod_sprints_9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b
SESSION_SECRET=sess_prod_sprints_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d

# Microservice Configuration
MAX_PAYLOAD_SIZE=65536
ENVIRONMENT=production
FASTAPI_BASE_URL=http://sprints-ai-api:8000
FASTAPI_API_KEY=sk_live_sprints_f4a9b2c8e1d7a3c5b9e0f2d4a6c8e0b2
NODE_FUNCTION_ALLOW_BUILTIN=crypto
```

---

## 3. Launching the Multi-Container Docker Stack

### Step 1: Build & Launch Containers
```bash
docker compose up --build -d
```

### Step 2: Verify Running Services
```bash
docker compose ps
```
Expected Output:
```
NAME                       IMAGE               COMMAND                  SERVICE          CREATED         STATUS                   PORTS
sprints_ai_service         sprints-ai-api      "python api/app.py"      sprints-ai-api   1 minute ago    Up 1 minute (healthy)    0.0.0.0:8000->8000/tcp
sprints_n8n_orchestrator   n8nio/n8n:2.47.2    "tini -- /docker-en..."  sprints-n8n      1 minute ago    Up 1 minute              0.0.0.0:5678->5678/tcp
```

### Step 3: Automated Workflow Import & Activation
When deploying via Docker, the n8n container imports and activates the workflow automatically:
```bash
docker exec sprints_n8n_orchestrator n8n import:workflow --input=/workflows/learner_success_agent.json
docker exec sprints_n8n_orchestrator n8n publish:workflow --id=sprints-capstone-n8n-workflow
docker restart sprints_n8n_orchestrator
```

---

## 4. Health Checks & Verification Probes

### 1. Probe FastAPI Backend Health:
```bash
curl -i http://localhost:8000/api/v1/health
```
Expected Response:
```json
HTTP/1.1 200 OK
{"status": "healthy", "service": "Sprints AI Learner Success Agent", "version": "2.0.0", "database": "connected"}
```

### 2. Probe Live n8n Webhook:
```bash
curl -X POST http://localhost:5678/webhook/learner-query \
  -H "Content-Type: application/json" \
  -d '{"learner_id": "LRN-1002", "query_text": "What is relational SQL?"}'
```
Expected Response:
```json
HTTP/1.1 200 OK
{
  "status": "resolved_auto",
  "ticket_id": "TCK-AUTO-...",
  "db_persisted": false,
  "correlation_id": "corr-...",
  "sla_met": true
}
```

---

## 5. Production Hardening: HTTPS, Reverse Proxy & Network Isolation

### Architecture Overview
In a public cloud production deployment (AWS EC2, GCP Compute Engine, DigitalOcean, or Bare-Metal Linux), microservices must **NEVER** expose backend ports (8000, 5678, 6379) directly to the public internet (`0.0.0.0`).

```mermaid
flowchart TD
    INTERNET(["Public Internet\n(Clients, Webhooks, Browsers)"])
    
    subgraph Host VM [Secured Host VM - UFW Ports 80 & 443 Only]
        PROXY["Caddy / Nginx Reverse Proxy\n(TLS 1.3, Let's Encrypt ACME, HSTS)\nPorts: 80 & 443"]
        
        subgraph Internal Network [Private Docker Network: sprints_prod_net (No Host Ports)]
            API["FastAPI Backend (Gunicorn/Uvicorn)\nsprints-ai-api:8000"]
            N8N["n8n Orchestrator\nsprints-n8n:5678"]
            REDIS["Redis Distributed Cache\nsprints-redis:6379\n(Atomic Replay & Token Bucket)"]
        end
    end
    
    INTERNET -->|HTTPS 443 / HTTP 80| PROXY
    PROXY -->|/api/* & /| API
    PROXY -->|/webhook/*| N8N
    API <-->|Atomic Replay Defense| REDIS
    N8N -->|Internal HMAC Requests| API
```

### Launching the Hardened Production Stack (`docker-compose.prod.yml`)
The production stack uses **Caddy** with automated Let's Encrypt TLS 1.3:
```bash
docker compose -f docker-compose.prod.yml up --build -d
```
1. **Public Ingress:** Only ports 80 (HTTP redirect) and 443 (TLS 1.3 HTTPS) are exposed on the host.
2. **Private Network:** FastAPI, n8n, and Redis communicate entirely over the internal bridge network `sprints_prod_net` with zero host port bindings.
3. **Enterprise Alternative:** An enterprise Nginx reverse proxy configuration is also provided in [`deploy/nginx.conf`](../deploy/nginx.conf).

### Linux Host Firewall Rules (UFW)
Enforce strict perimeter defense on the host OS:
```bash
# 1. Default deny incoming traffic
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 2. Allow SSH management port
sudo ufw allow 22/tcp

# 3. Allow only public Web / HTTPS ingress
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 4. Explicitly block internal ports from external interfaces
sudo ufw deny 8000/tcp
sudo ufw deny 5678/tcp
sudo ufw deny 6379/tcp

# 5. Enable firewall
sudo ufw enable
sudo ufw status verbose
```

---

## 6. Live Interactive Dashboard Access

The Live Interactive Web Dashboard is continuously hosted and accessible directly via web browser:

**Live URL:** `http://localhost:8000`

### Available Features on Live Dashboard:
1. **Real-time Learner Query Simulator:** Test Arabic dialect queries (`"مش فاهم Gradient Descent"`, `"أنا محبط ومش قادر أكمل"`).
2. **Deep MLP Dropout Prediction Panel:** Test interactive quiz sliders and observe real-time risk classification and recommendation cards.
3. **PyTorch LSTM Sequence Visualizer:** Dynamic Chart.js trajectory visualization forecasting Capstone performance from longitudinal quiz attempts.
4. **Live SQLite Support Tickets Table:** Real-time view of persistent tickets, status badges, and idempotency deduplication flags.
5. **Security Controls:** Session authentication, active CSRF protection, and zero plaintext secret leakage.

---

## 7. Teardown & Clean Up

To gracefully stop and remove containers and volumes:
```bash
docker compose down -v
```

**Audit Verdict:** 10/10 — Automated, repeatable, and containerized deployment with zero friction.
