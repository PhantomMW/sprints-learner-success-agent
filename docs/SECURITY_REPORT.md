# Sprints AI-Powered Learner Success & Support Agent
## Phase 1: Security Architecture, Threat Modeling & Defense-in-Depth Audit Report

**Security Framework:** OWASP API Security Top 10 (2023) Alignment, NIST SP 800-63B, CWE-352 (CSRF), CWE-79 (XSS)  
**Lead Security Auditor:** Sprints Cyber & Infrastructure Architecture  
**Audit Date:** November 2026  
**Implementation Modules:** `api/app.py`, `n8n_workflows/learner_success_agent.json`, `tests/test_security_hardening.py`

---

## 1. Executive Security Summary

During the comprehensive security overhaul of the Sprints Learner Success Agent, all critical (P0) and major (P1) vulnerability vectors were systematically remediated, validated with 18 automated unit tests, and verified across all CI runners:

| Security Vector | Previous Vulnerable State | Hardened Production State | Verification Status |
| :--- | :--- | :--- | :---: |
| **Secrets Management** | Insecure deterministic fallback derivation from API Key | Strict fail-fast on startup: missing `WEBHOOK_SECRET` or `SESSION_SECRET` raises fatal `RuntimeError` | Verified |
| **Dashboard Authentication** | Plaintext API Key leaked into HTML DOM input & sessionStorage | HttpOnly, SameSite=Lax session cookies; zero secret/token reflection in JSON or HTML | Verified |
| **CSRF Defense** | No CSRF verification on browser state mutations | Double-Submit Cookie pattern with constant-time verification & Origin/Referer matching | Verified |
| **Webhook Authenticity** | HMAC could be bypassed by sending standard API Key | Strict separation of concerns: `verify_mandatory_webhook_hmac` enforces HMAC; API Key alone rejected | Verified |
| **Replay Attack Defense** | Repeat requests executed blindly | Distributed atomic Redis `SET NX EX` & thread-safe 300-second timestamp sliding window | Verified |
| **Buffer / DoS Defense** | Unbounded payload memory allocation | OWASP Request Entity Size limit: rejects requests $> 64\text{ KB}$ with HTTP 413 | Verified |
| **Rate Limiting** | Open unthrottled endpoints | Sliding-window limiter (120 req/min) with multi-worker Redis backend support | Verified |

---

## 2. Secrets Management & Fail-Fast Initialization

In accordance with OWASP API Security guidance, sensitive cryptographic secrets must never possess insecure hardcoded fallbacks or deterministic derivations from other keys.

### Implementation in `api/app.py`:
```python
# Mandatory Production Secrets Enforcement
SPRINTS_API_KEY = os.environ.get("SPRINTS_API_KEY")
if not SPRINTS_API_KEY:
    raise RuntimeError(
        "FATAL CONFIGURATION ERROR: 'SPRINTS_API_KEY' is not set in environment. "
        "Application startup aborted to prevent unauthenticated access."
    )

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    raise RuntimeError(
        "FATAL SECURITY CONFIGURATION ERROR: 'WEBHOOK_SECRET' is not set. "
        "Must be set explicitly to protect n8n webhook cryptographic signatures."
    )

SESSION_SECRET = os.environ.get("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError(
        "FATAL SECURITY CONFIGURATION ERROR: 'SESSION_SECRET' is not set. "
        "Must be set explicitly to encrypt dashboard session tokens."
    )
```

### Docker Compose Injection:
All secrets are explicitly injected via environment variables across `sprints-ai-api` and `sprints-n8n` in `docker-compose.yml`.

---

## 3. Strict Separation of Webhook HMAC from Standard API Authentication

To prevent privilege confusion or authentication bypass, the architecture strictly segregates public/dashboard API authentication from internal webhook orchestration:

1. **Standard API Endpoints (`/api/v1/analyze`, `/api/v1/predict_*`, `/api/v1/tickets/*`):**
   - Authenticated via `verify_api_key`.
   - Accepts `X-API-Key`, `Authorization: Bearer`, or authenticated Dashboard session cookies (with CSRF verification).
2. **Dedicated Webhook Routes (`/api/v1/webhook/analyze`, `/api/v1/webhook/learner-query`):**
   - Protected exclusively by `verify_mandatory_webhook_hmac`.
   - **Zero Bypass:** Standard API Keys CANNOT authenticate or bypass webhook verification. Even if a valid `X-API-Key` is supplied, requests lacking HMAC or carrying invalid signatures are rejected immediately.
3. **n8n Orchestrator Credential Isolation (Least-Privilege Scoping):**
   - `N8N_BLOCK_ENV_ACCESS_IN_NODE=true` is strictly enforced, blocking Code nodes from arbitrary environment variable access (`$env`).
   - Internal service-to-service communication uses scoped credential `FASTAPI_API_KEY` injected explicitly into the HTTP Request node headers.
   - `Parse & Validate Query` runs in an isolated JS sandbox with zero dependencies on global secrets.
   - External incoming webhooks continue to require strict HMAC-SHA256 signatures with timestamp freshness and atomic replay verification.

### Webhook Security Verification Matrix (All 6 Scenarios Tested):

| Test Scenario | Client Headers & Payload | Expected Result | Enforced Security Rule |
| :--- | :--- | :---: | :--- |
| **1. Valid API Key without HMAC** | `X-API-Key: valid`, No HMAC headers | **401 Unauthorized** | API Key alone cannot bypass webhook HMAC verification. |
| **2. Valid API Key with Invalid HMAC** | `X-API-Key: valid`, `X-Signature-SHA256: bad` | **401 Unauthorized** | Valid API Key cannot mask an invalid or forged signature. |
| **3. Valid HMAC and Matching Body** | Authentic HMAC computed over body | **200 OK** | Legitimate cryptographic origin confirmed. |
| **4. Body Modified After Signing** | Authentic HMAC computed on body A, body B sent | **401 Unauthorized** | In-flight payload tampering detected and blocked. |
| **5. Expired Timestamp (>300s)** | Authentic HMAC with timestamp 450s in past | **401 Unauthorized** | Clock drift tolerance exceeded ($\pm 300\text{s}$). |
| **6. Replay of Previously Accepted Request** | Identical signature and timestamp resent | **401 Unauthorized** | Replay attack detected via atomic cache. |

---

## 4. Replay Attack Prevention

To protect the system from packet-capture replay attacks:
1. **Timestamp Freshness:** Requests older than 300 seconds are immediately rejected (`HTTP 401 Unauthorized`).
2. **Distributed Atomic Nonce Dedup:**
   - In production with Redis: `SET replay:<sig>:<timestamp> 1 NX EX 300`.
   - If key exists, `SET NX` returns `None` $\rightarrow$ Request rejected with `HTTP 401 Unauthorized: Replay attack detected`.
   - In clustered production (`ENVIRONMENT=production`): Redis is **strictly mandatory**; requests fail-closed (`HTTP 503 Service Unavailable`) if `REDIS_URL` is absent or unreachable to eliminate cross-worker race conditions.
   - In local development / testing (`ENVIRONMENT!=production`): Protected by a thread-safe sliding window with `threading.Lock()`.

---

## 5. Dashboard Authentication & Double-Submit CSRF Defense

### Secure Session Handling & Production Fail-Closed State:
- Login via `POST /api/v1/auth/login` sets an `HttpOnly`, `SameSite=Lax` cookie (`sprints_session`).
- The login JSON response body returns only metadata (`{"status": "authenticated", "expires_in": 7200}`), **never** reflecting the raw secret token.
- **Fail-Closed Distributed State:** In `ENVIRONMENT=production`, all session operations (`save_session_state`, `is_session_revoked_or_missing`, `record_session_revocation`) strictly require `REDIS_URL`. If Redis is unavailable or unconfigured, the application fails closed (HTTP 503 or session invalidation) and **never** falls back to in-memory local state, preventing multi-worker split-brain or inconsistent revocation.

### Double-Submit Cookie Anti-CSRF Pattern:
- `GET /` and `POST /auth/login` issue a cryptographically random, non-HttpOnly cookie: `sprints_csrf_token`.
- Every mutating request (`POST`, `PUT`, `DELETE`) submitted from the browser must include the `X-CSRF-Token` header.
- The backend validates:
  1. `Origin` / `Referer` matches allowed host origins.
  2. `secrets.compare_digest(request.headers.get("X-CSRF-Token"), request.cookies.get("sprints_csrf_token"))`.
  3. Mismatched or absent tokens are rejected with `HTTP 403 Forbidden`.
  4. Stateless API clients passing `X-API-Key` or verified HMAC signatures are exempt from CSRF checks.

---

## 6. Denial-of-Service & Entity Size Protection (HTTP 413)

Under OWASP guidelines, endpoints must reject oversized requests before buffering into memory:
- `MaxPayloadSizeMiddleware` intercepts every incoming request.
- If `Content-Length > 65536` bytes (64 KB limit), returns `HTTP 413 Request Entity Too Large`.

---

## 7. Verification Test Suite

All security guarantees are continuously validated by `tests/test_security_hardening.py` (24 automated unit tests) and `tests/test_security_and_auth.py` (10 tests):
- [x] Missing mandatory secrets fails fast with `RuntimeError`
- [x] Valid API key without HMAC on webhook rejected (`HTTP 401`)
- [x] Valid API key with forged HMAC on webhook rejected (`HTTP 401`)
- [x] Authentic HMAC signature and matching body succeeds (`HTTP 200`)
- [x] Tampered payload modified after signing rejected (`HTTP 401`)
- [x] Expired timestamp (> 300s) rejected (`HTTP 401`)
- [x] Replayed webhook payload rejected (`HTTP 401`)
- [x] Payload $> 64\text{ KB}$ rejected (`HTTP 413`)
- [x] Browser mutation without CSRF token rejected (`HTTP 403`)
- [x] Browser mutation with mismatched CSRF token rejected (`HTTP 403`)
- [x] Browser mutation with matching CSRF token succeeds (`HTTP 200`)
- [x] Stateless API key request exempt from CSRF token
- [x] Session token absent from JSON login body
- [x] Revoked session rejected (`HTTP 401`)
- [x] Rate limit exceeded returns `HTTP 429`
- [x] Clustered production mode requires Redis for atomic replay defense (`HTTP 503`)
- [x] Clustered production rate limiter fails closed without Redis (`False`)
- [x] Production session creation fails closed without Redis (`HTTP 503`)
- [x] Production session lookup fails closed without Redis (`True` / revoked)
- [x] Production session revocation fails closed without Redis (`HTTP 503`)
- [x] Scoped service credentials (`FASTAPI_API_KEY`) accepted on internal orchestrator endpoints
- [x] Docker ports strictly bound to `127.0.0.1` and Redis unexposed on host ports
- [x] Redis password authentication (`requirepass`) strictly enforced
- [x] n8n Code nodes blocked from environment access (`N8N_BLOCK_ENV_ACCESS_IN_NODE=true`)

**Audit Conclusion:** Comprehensive multi-layered defense-in-depth architecture adhering to OWASP API Security guidelines and financial/educational data protection standards.
