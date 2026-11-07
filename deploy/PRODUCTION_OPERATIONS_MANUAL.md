# Production Operations & Cryptographic Secret Lifecycle Manual

**Document Version:** 2.0.0  
**Effective Date:** November 2026  
**Audience:** Site Reliability Engineers (SRE), DevSecOps, Platform Administrators  
**Classification:** Operational Security Reference (Confidential / Internal)  

---

## 1. Cryptographic Key Generation & Entropy Standards

All production deployments of the **Sprints AI-Powered Learner Success & Support Agent** require cryptographically strong, 256-bit high-entropy secrets.

> [!CAUTION]
> The system strictly enforces a **Zero-Default Secret Policy**. The service will fail-fast with a `RuntimeError` on startup if any secret matches known test defaults (e.g. `sprints_redis_vault_secret_2026`, `changeme`, or any string under 16 characters).

### 1.1. Key Generation Commands
To generate a unique 32-byte (64 hex characters) cryptographically secure secret:

```bash
# Using OpenSSL (Recommended):
openssl rand -hex 32

# Using Python Standard Library secrets module:
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## 2. Mandatory Environment Variables Reference

| Variable Name | Required Scope | Minimum Entropy | Description |
| :--- | :--- | :--- | :--- |
| `SPRINTS_API_KEY` | FastAPI Backend | 32 hex chars (128-bit) | Master authorization key for operational microservice calls. |
| `FASTAPI_API_KEY` | n8n Credential Vault | 32 hex chars (128-bit) | Scoped credential stored in n8n encrypted vault for `/api/v1/analyze`. |
| `WEBHOOK_SECRET` | FastAPI Ingest | 32 hex chars (256-bit) | Cryptographic shared secret for HMAC-SHA256 signature verification. |
| `SESSION_SECRET` | FastAPI Auth | 32 hex chars (256-bit) | HMAC signing key for HttpOnly session cookies (`sprints_session_id`). |
| `REDIS_PASSWORD` | Redis Server & API | 32 hex chars (256-bit) | Authentication password for distributed rate limiting & replay defense. |
| `N8N_ENCRYPTION_KEY` | n8n Orchestrator | 32 hex chars (256-bit) | AES-256 key used by n8n to encrypt SQLite credential records. |

---

## 3. Zero-Downtime Secret Rotation Runbooks

### 3.1. API Key Rotation Protocol (Graceful Dual-Key Overlap)
1. **Generate New Key:**
   ```bash
   NEW_API_KEY=$(openssl rand -hex 32)
   ```
2. **Deploy Primary with Dual-Accept Window:**
   - In `api/app.py`, the system verifies API keys using timing-safe comparisons against both `SPRINTS_API_KEY` and optional `FASTAPI_API_KEY`.
   - Update `FASTAPI_API_KEY` with `$NEW_API_KEY` in `.env` and reload the FastAPI container:
     ```bash
     docker compose exec sprints_ai_service kill -HUP 1
     ```
3. **Update n8n Vault Credentials:**
   ```bash
   docker exec sprints_n8n_orchestrator sh -c "echo '[{\"id\":\"sprints-api-key-cred\",\"name\":\"Sprints FastAPI Header Auth\",\"type\":\"httpHeaderAuth\",\"data\":{\"name\":\"X-API-Key\",\"value\":\"'$NEW_API_KEY'\"}}]' > /tmp/rotate_creds.json && n8n import:credentials --input=/tmp/rotate_creds.json && rm -f /tmp/rotate_creds.json"
   ```
4. **Finalize Master Key:**
   - Update `SPRINTS_API_KEY=$NEW_API_KEY` in production `.env`.
   - Verify all endpoints return `HTTP 200 OK`.

---

### 3.2. Redis Password Rotation Protocol
Redis 7 supports dynamic runtime credential updates via Redis ACL without restarting the server:

1. **Connect via Redis CLI using Current Password:**
   ```bash
   docker exec -it sprints_redis_cache redis-cli -a "$CURRENT_REDIS_PASSWORD"
   ```
2. **Add New Secondary Password to Default User:**
   ```redis
   ACL SETUSER default >NEW_REDIS_PASSWORD_HEX32
   ```
   *(The user now accepts BOTH the old and new passwords simultaneously).*
3. **Update FastAPI & Client Configurations:**
   - In `.env`, update `REDIS_PASSWORD=NEW_REDIS_PASSWORD_HEX32`.
   - Restart the API service:
     ```bash
     docker compose restart sprints-ai-api
     ```
4. **Revoke Old Password from Redis:**
   ```redis
   ACL SETUSER default <OLD_REDIS_PASSWORD_HEX32
   ```
5. **Verify Fail-Closed Rejection of Old Credentials:**
   ```bash
   docker exec -it sprints_redis_cache redis-cli -a "$OLD_REDIS_PASSWORD" ping
   # Expected Output: (error) WRONGPASS invalid username-password pair
   ```

---

### 3.3. n8n Encryption Key & Credential Vault Migration
Because n8n encrypts all stored credentials in SQLite using `N8N_ENCRYPTION_KEY`, rotating this key requires exporting and re-encrypting:

1. **Export Existing Workflows & Credentials:**
   ```bash
   docker exec sprints_n8n_orchestrator n8n export:workflow --all --output=/tmp/backup_workflows.json
   docker exec sprints_n8n_orchestrator n8n export:credentials --all --decrypted --output=/tmp/backup_creds.json
   ```
2. **Update Encryption Key in Production `.env`:**
   ```bash
   NEW_N8N_KEY=$(openssl rand -hex 32)
   sed -i "s/N8N_ENCRYPTION_KEY=.*/N8N_ENCRYPTION_KEY=$NEW_N8N_KEY/" .env
   ```
3. **Re-initialize Vault with New Key:**
   ```bash
   docker compose restart sprints-n8n
   docker exec sprints_n8n_orchestrator n8n import:credentials --input=/tmp/backup_creds.json
   docker exec sprints_n8n_orchestrator rm -f /tmp/backup_creds.json /tmp/backup_workflows.json
   ```

---

## 4. Disaster Recovery & Emergency Fail-Closed Verification

In production (`ENVIRONMENT=production`), the application strictly executes in **Fail-Closed Mode**:
- **Simulate Redis Outage:**
  ```bash
  docker compose stop sprints-redis
  ```
- **Probe API Endpoints:**
  ```bash
  curl -i http://127.0.0.1:8000/api/v1/auth/session
  # Expected Response: HTTP/1.1 503 Service Unavailable
  # Body: {"detail":"Distributed session cache (Redis) is currently unavailable..."}
  ```
- Under zero circumstances will the system revert to local in-memory dictionaries in production, preventing token synchronization splits and distributed replay vulnerabilities.
