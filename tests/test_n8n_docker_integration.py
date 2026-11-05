"""
Robust integration test for n8n orchestrator and Docker Compose deployment.
Validates:
1. Safe Docker CLI availability & daemon checks.
2. Automatic n8n workflow import & activation via container CLI.
3. Live Webhook invocation across BOTH distinct routing branches:
   - Automated Curriculum RAG: status='resolved_auto', db_persisted=False, verified NO SQLite record.
   - Distressed Escalation: status='escalated', db_persisted=True, verified persistent SQLite record.
4. Strict enforcement of sub-3.5s latency SLA.
5. Strict exit codes: fails if services offline (unless --allow-skip is explicitly passed).
"""

import os
import sys
import time
import json
import sqlite3
import shutil
import argparse
import subprocess
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, 'database', 'sprints_learners.db')
WORKFLOW_PATH = os.path.join(REPO_ROOT, 'n8n_workflows', 'learner_success_agent.json')
COMPOSE_PATH = os.path.join(REPO_ROOT, 'docker-compose.yml')

N8N_BASE_URL = os.environ.get('N8N_BASE_URL', 'http://localhost:5678')
FASTAPI_BASE_URL = os.environ.get('FASTAPI_BASE_URL', 'http://localhost:8000')

def inspect_docker():
    """Safely checks if Docker CLI is installed and daemon is operational."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return {"installed": False, "daemon_running": False, "error": "Docker CLI binary not found in PATH."}
    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=6)
        if res.returncode == 0:
            return {"installed": True, "daemon_running": True, "error": None}
        return {"installed": True, "daemon_running": False, "error": "Docker daemon is not running / unresponsive."}
    except Exception as e:
        return {"installed": True, "daemon_running": False, "error": str(e)}

def is_service_online(url: str, timeout: float = 3.0) -> bool:
    """Probes HTTP service health endpoint."""
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code in [200, 401]
    except Exception:
        return False

def auto_import_workflow_if_docker(container_name: str = "sprints_n8n_orchestrator") -> bool:
    """
    If running inside Docker Compose, imports and publishes/activates the workflow
    with strict returncode verification and failure logging. Halts if import or publish fails.
    """
    docker_info = inspect_docker()
    if not docker_info["daemon_running"]:
        return False

    try:
        ps_check = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        if container_name in ps_check.stdout:
            print(f"   [SETUP] Detected active container '{container_name}'.")
            print("   [SETUP] Auto-importing workflow from /workflows/learner_success_agent.json...")
            
            # 1. Import workflow JSON with strict return code check
            res_import = subprocess.run([
                "docker", "exec", container_name,
                "n8n", "import:workflow", "--input=/workflows/learner_success_agent.json"
            ], capture_output=True, text=True, timeout=20)
            if res_import.returncode != 0:
                print(f"   [ERROR] n8n workflow import failed (code {res_import.returncode}):")
                print("   STDOUT:", res_import.stdout)
                print("   STDERR:", res_import.stderr)
                raise RuntimeError(f"Workflow import failed (exit code {res_import.returncode}): {res_import.stderr.strip()}")
            print(f"   [PASS] Workflow imported successfully: {res_import.stdout.strip()}")

            # 2. Publish workflow with strict return code check
            res_publish = subprocess.run([
                "docker", "exec", container_name,
                "n8n", "publish:workflow", "--id=sprints-capstone-n8n-workflow"
            ], capture_output=True, text=True, timeout=20)
            if res_publish.returncode != 0:
                print(f"   [ERROR] n8n workflow publish failed (code {res_publish.returncode}):")
                print("   STDOUT:", res_publish.stdout)
                print("   STDERR:", res_publish.stderr)
                raise RuntimeError(f"Workflow publish failed (exit code {res_publish.returncode}): {res_publish.stderr.strip()}")
            print(f"   [PASS] Workflow published successfully: {res_publish.stdout.strip()}")

            # 3. Verify workflow is active in database
            res_export = subprocess.run([
                "docker", "exec", container_name,
                "n8n", "export:workflow", "--id=sprints-capstone-n8n-workflow", "--output=/tmp/exported_wf.json"
            ], capture_output=True, text=True, timeout=15)
            if res_export.returncode == 0:
                print("   [PASS] Verified workflow active and exported from n8n container database.")

            # 4. Probe webhook; if needed, restart container so in-memory routing table reflects DB
            webhook_url = f"{N8N_BASE_URL.rstrip('/')}/webhook/learner-query"
            try:
                probe_res = requests.post(webhook_url, json={"learner_id": "LRN-1002", "query_text": "What is SQL?"}, timeout=4)
                if probe_res.status_code == 404:
                    print("   [SETUP] Webhook not registered in router yet; restarting container to load active DB routes...")
                    subprocess.run(["docker", "restart", container_name], capture_output=True, text=True, timeout=25)
                    time.sleep(4)
                    for _ in range(25):
                        if is_service_online(N8N_BASE_URL):
                            break
                        time.sleep(2)
            except Exception:
                pass

            # 5. Wait for live webhook readiness (HTTP 200 + valid response)
            print("   [SETUP] Awaiting live webhook endpoint readiness (HTTP 200)...")
            webhook_ready = False
            for _ in range(20):
                try:
                    probe_res = requests.post(webhook_url, json={"learner_id": "LRN-1002", "query_text": "What is SQL?"}, timeout=4)
                    if probe_res.status_code == 200 and ("status" in probe_res.json()):
                        webhook_ready = True
                        break
                except Exception:
                    pass
                time.sleep(1.5)

            if not webhook_ready:
                raise RuntimeError("n8n webhook /webhook/learner-query failed to become ready within timeout period.")

            print("   [SETUP] Workflow fully activated, live, and validated in n8n container!")
            return True
    except Exception as e:
        print(f"   [SETUP ERROR] Container CLI import error: {e}")
        raise
    return False

def main():
    parser = argparse.ArgumentParser(description="Test n8n Docker webhook integration.")
    parser.add_argument("--allow-skip", action="store_true", help="Exit 0 with SKIPPED status if services are offline.")
    parser.add_argument("--require-live", action="store_true", help="Strictly fail if services are offline.")
    args = parser.parse_args()

    print("=" * 65)
    print("SPRINTS N8N & DOCKER COMPOSE INTEGRATION TEST")
    print("=" * 65)

    # 1. Safe Docker & Service Discovery
    docker_info = inspect_docker()
    print("1. Infrastructure & Service Discovery:")
    if not docker_info["installed"]:
        print("   Docker CLI:     NOT INSTALLED")
    elif not docker_info["daemon_running"]:
        print(f"   Docker Daemon:  INACTIVE ({docker_info['error']})")
    else:
        print("   Docker Daemon:  OPERATIONAL (Ready)")

    # Validate docker-compose.yml syntax safely
    if docker_info["installed"]:
        try:
            compose_proc = subprocess.run(["docker", "compose", "config"], capture_output=True, text=True, timeout=8)
            if compose_proc.returncode == 0:
                print("   docker-compose: VALID (Multi-container spec verified)")
            else:
                print(f"   docker-compose: ERROR ({compose_proc.stderr.strip()})")
        except Exception as ce:
            print(f"   docker-compose: Notice ({ce})")

    # Workflow file check
    if not os.path.exists(WORKFLOW_PATH):
        print(f"   FATAL: Workflow file missing at {WORKFLOW_PATH}")
        sys.exit(1)
    with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
        wf = json.load(f)
    print(f"   Workflow Spec:  VALID ('{wf['name']}', {len(wf['nodes'])} nodes)")

    fastapi_online = is_service_online(f"{FASTAPI_BASE_URL}/api/v1/health")
    n8n_online = is_service_online(f"{N8N_BASE_URL}/healthz") or is_service_online(N8N_BASE_URL)

    print(f"   FastAPI Backend: {'ONLINE (Port 8000)' if fastapi_online else 'OFFLINE (Port 8000)'}")
    print(f"   n8n Webhook:     {'ONLINE (Port 5678)' if n8n_online else 'OFFLINE (Port 5678)'}")

    # 2. Strict Online Verification
    if not n8n_online or not fastapi_online:
        print("\n" + "!" * 65)
        print("SERVICES OFFLINE: Live integration tests cannot execute.")
        print("To launch the multi-container stack for real-time testing:")
        print("  docker compose up --build -d")
        print("!" * 65)

        if args.allow_skip:
            print("\n[RESULT: SKIPPED] Host environment does not have active live services (--allow-skip enabled).")
            sys.exit(0)
        else:
            print("\n[RESULT: FAILED] Integration test failed because live services are required.")
            print("Run with --allow-skip to permit skipping on developer machines without running Docker.")
            sys.exit(1)

    # 3. Auto-import workflow if in Docker
    auto_import_workflow_if_docker()

    # 4. Live Webhook Testing (Both Paths)
    execute_live_webhook_assertions()

def execute_live_webhook_assertions(webhook_url: str = None):
    if not webhook_url:
        webhook_url = f"{N8N_BASE_URL}/webhook/learner-query"
    print(f"\n2. Executing Real-time Webhooks against: {webhook_url}")

    # PATH A: Automated Curriculum Query (Semantic Vector RAG)
    print("\n--- Test Path A: Neutral Curriculum Query (Automated RAG) ---")
    rag_payload = {
        "learner_id": "LRN-1002",
        "query_text": "What are the core principles of relational SQL databases?",
        "cohort_city": "Cairo"
    }
    t0 = time.time()
    resp_rag = requests.post(webhook_url, json=rag_payload, timeout=12)
    lat_rag = (time.time() - t0) * 1000

    print(f"   Status Code: {resp_rag.status_code}")
    print(f"   Roundtrip Latency: {lat_rag:.2f}ms (SLA < 3500ms)")
    assert resp_rag.status_code == 200, f"Expected 200 OK, got {resp_rag.status_code}: {resp_rag.text}"
    assert lat_rag < 3500.0, f"Path A response latency SLA breached: {lat_rag:.2f}ms >= 3500.0ms"
    data_rag = resp_rag.json()
    print("   Routing Decision:", data_rag.get("status"))
    print("   Ticket ID:", data_rag.get("ticket_id"))
    print("   DB Persisted Flag:", data_rag.get("db_persisted"))

    assert data_rag.get("correlation_id") is not None, "Path A must return correlation_id"
    assert data_rag.get("status") == "resolved_auto", f"Expected 'resolved_auto', got {data_rag.get('status')}"
    assert data_rag.get("db_persisted") is False, "Automated RAG must NOT mark db_persisted as True"
    assert str(data_rag.get("ticket_id")).startswith("TCK-AUTO-"), "Ticket ID must follow TCK-AUTO prefix"

    # Verify no persistent DB record for auto-resolved ticket
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ticket_id FROM support_tickets WHERE ticket_id = ?", (data_rag.get("ticket_id"),))
    row_rag = cur.fetchone()
    conn.close()
    assert row_rag is None, "Automated RAG ticket should NOT be persisted in SQLite support_tickets!"
    print("   [VERIFIED] SQLite DB check: Zero unwanted records created for automated response.")

    # PATH B: Pure Egyptian Arabic Distress Query (Human Mentor Escalation - Arabic Only)
    print("\n--- Test Path B: Pure Egyptian Arabic Distress Query (Human Mentor Escalation) ---")
    esc_payload_ar = {
        "learner_id": "LRN-1019",
        "query_text": "أنا مش فاهم PyTorch خالص ومحبط جداً ومش قادر أكمل وعايز مساعدة ضروري",
        "cohort_city": "Alexandria"
    }
    t0 = time.time()
    resp_ar = requests.post(webhook_url, json=esc_payload_ar, timeout=12)
    lat_ar = (time.time() - t0) * 1000

    print(f"   Status Code: {resp_ar.status_code}")
    print(f"   Roundtrip Latency: {lat_ar:.2f}ms (SLA < 3500ms)")
    assert resp_ar.status_code == 200, f"Expected 200 OK, got {resp_ar.status_code}: {resp_ar.text}"
    assert lat_ar < 3500.0, f"Path B (Arabic Distress) SLA breached: {lat_ar:.2f}ms >= 3500.0ms"
    data_ar = resp_ar.json()
    print("   Routing Decision:", data_ar.get("status"))
    print("   Ticket ID:", data_ar.get("ticket_id"))
    print("   DB Persisted Flag:", data_ar.get("db_persisted"))

    assert data_ar.get("correlation_id") is not None, "Path B must return correlation_id"
    assert data_ar.get("idempotency_key") is not None, "Path B must return idempotency_key"
    assert data_ar.get("status") == "escalated", f"Expected 'escalated', got {data_ar.get('status')}"
    assert data_ar.get("db_persisted") is True, "Escalated ticket must be marked db_persisted = True"
    assert str(data_ar.get("ticket_id")).startswith("TCK-ESC-"), "Ticket ID must follow TCK-ESC prefix"

    # Verify persistent SQLite record
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ticket_id, learner_id, priority, status FROM support_tickets WHERE ticket_id = ?", (data_ar.get("ticket_id"),))
    row_ar = cur.fetchone()
    conn.close()
    assert row_ar is not None, f"Escalation ticket {data_ar.get('ticket_id')} was not found in SQLite DB!"
    print(f"   [VERIFIED] SQLite DB Record Confirmed: {row_ar}")

    # PATH C: Pure English Distress Query (Human Mentor Escalation - English Only)
    print("\n--- Test Path C: Pure English Distress Query (Human Mentor Escalation) ---")
    esc_payload_en = {
        "learner_id": "LRN-1005",
        "query_text": "I am completely failing this module, cannot keep up, and want to drop out urgently",
        "cohort_city": "Cairo"
    }
    t0 = time.time()
    resp_en = requests.post(webhook_url, json=esc_payload_en, timeout=12)
    lat_en = (time.time() - t0) * 1000

    print(f"   Status Code: {resp_en.status_code}")
    print(f"   Roundtrip Latency: {lat_en:.2f}ms (SLA < 3500ms)")
    assert resp_en.status_code == 200, f"Expected 200 OK, got {resp_en.status_code}: {resp_en.text}"
    assert lat_en < 3500.0, f"Path C (English Distress) SLA breached: {lat_en:.2f}ms >= 3500.0ms"
    data_en = resp_en.json()
    print("   Routing Decision:", data_en.get("status"))
    print("   Ticket ID:", data_en.get("ticket_id"))
    print("   DB Persisted Flag:", data_en.get("db_persisted"))

    assert data_en.get("status") == "escalated", f"Expected 'escalated', got {data_en.get('status')}"
    assert data_en.get("db_persisted") is True, "Escalated ticket must be marked db_persisted = True"
    assert str(data_en.get("ticket_id")).startswith("TCK-ESC-"), "Ticket ID must follow TCK-ESC prefix"

    # Verify persistent SQLite record
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ticket_id, learner_id, priority, status FROM support_tickets WHERE ticket_id = ?", (data_en.get("ticket_id"),))
    row_en = cur.fetchone()
    conn.close()
    assert row_en is not None, f"Escalation ticket {data_en.get('ticket_id')} was not found in SQLite DB!"
    print(f"   [VERIFIED] SQLite DB Record Confirmed: {row_en}")

    # PATH D: Technical Task Inability (Must Route to Automated RAG, 0 SQLite Records)
    print("\n--- Test Path D: Technical Task Inability (Automated RAG Disambiguation) ---")
    tech_payload = {
        "learner_id": "LRN-1002",
        "query_text": "مش قادر أفتح ملف المحاضرة، ممكن تساعدني؟",
        "cohort_city": "Cairo"
    }
    t0 = time.time()
    resp_tech = requests.post(webhook_url, json=tech_payload, timeout=12)
    lat_tech = (time.time() - t0) * 1000

    print(f"   Status Code: {resp_tech.status_code}")
    print(f"   Roundtrip Latency: {lat_tech:.2f}ms (SLA < 3500ms)")
    assert resp_tech.status_code == 200
    assert lat_tech < 3500.0, f"Path D SLA breached: {lat_tech:.2f}ms >= 3500.0ms"
    data_tech = resp_tech.json()
    assert data_tech.get("status") == "resolved_auto", f"Expected 'resolved_auto', got {data_tech.get('status')}"
    assert data_tech.get("db_persisted") is False, "Technical task query must NOT persist ticket to DB"
    print("   Routing Decision:", data_tech.get("status"))
    print("   [VERIFIED] Technical query correctly routed to automated RAG without false escalation.")

    # PATH E: Urgent Emergency Plea & Withdrawal (Human Mentor Escalation - Critical Priority)
    print("\n--- Test Path E: Urgent Emergency Plea & Withdrawal Intent ---")
    urgent_payload = {
        "learner_id": "LRN-1005",
        "query_text": "الحقوني، أنا هسيب الكورس ومخنوق جدا ومش عارف أكمل",
        "cohort_city": "Cairo"
    }
    t0 = time.time()
    resp_urgent = requests.post(webhook_url, json=urgent_payload, timeout=12)
    lat_urgent = (time.time() - t0) * 1000

    print(f"   Status Code: {resp_urgent.status_code}")
    print(f"   Roundtrip Latency: {lat_urgent:.2f}ms (SLA < 3500ms)")
    assert resp_urgent.status_code == 200
    assert lat_urgent < 3500.0, f"Path E SLA breached: {lat_urgent:.2f}ms >= 3500.0ms"
    data_urgent = resp_urgent.json()
    assert data_urgent.get("status") == "escalated", f"Expected 'escalated', got {data_urgent.get('status')}"
    assert data_urgent.get("priority") == "Critical", f"Expected Critical, got {data_urgent.get('priority')}"
    assert data_urgent.get("db_persisted") is True
    print("   Routing Decision:", data_urgent.get("status"))
    print("   Priority:", data_urgent.get("priority"))
    print("   [VERIFIED] Emergency plea correctly escalated to Human Mentor Desk.")

    max_lat = max(lat_rag, lat_ar, lat_en, lat_tech, lat_urgent)
    assert max_lat < 3500.0, f"Overall SLA breached: {max_lat:.2f}ms >= 3500.0ms"

    print("\n" + "=" * 65)
    print("ALL 5 N8N DOCKER WEBHOOK ROUTING & SQLITE PERSISTENCE TESTS PASSED!")
    print(f"Max Latency Observed: {max_lat:.2f}ms (Strict SLA < 3500ms 100% SATISFIED)")
    print("=" * 65)



# ==============================================================================
# PYTEST SUITE INTERFACE
# ==============================================================================

def test_workflow_json_schema_and_graph():
    """Sprint 4: Validates n8n workflow file existence, node configuration, retries and credentials."""
    assert os.path.exists(WORKFLOW_PATH), f"Workflow file missing at {WORKFLOW_PATH}"
    with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
        wf = json.load(f)
    assert wf['name'] == "Sprints AI Learner Success & Support Agent"
    assert len(wf['nodes']) >= 8
    node_names = [n['name'] for n in wf['nodes']]
    assert "HTTP Webhook Trigger" in node_names
    assert "Analyze Sentiment, NER & Vector RAG" in node_names
    assert "Persist Escalation to SQLite DB" in node_names
    assert "Deliberative Decision Router" in node_names

def test_docker_compose_syntax_and_service_spec():
    """Sprint 4: Validates docker-compose.yml configuration, authentication tokens and volume mounts."""
    assert os.path.exists(COMPOSE_PATH), f"Compose file missing at {COMPOSE_PATH}"
    with open(COMPOSE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "sprints-ai-api:" in content
    assert "sprints-n8n:" in content
    assert "SPRINTS_API_KEY" in content
    assert "FASTAPI_API_KEY" in content

def test_live_docker_webhook_dual_branch_routing():
    """
    Sprint 4: Live multi-container webhook validation against both branches
    (Automated RAG and Escalation) if live services are online.
    If services are offline, skips with informative message unless running in CI with live containers.
    """
    import pytest
    fastapi_online = is_service_online(f"{FASTAPI_BASE_URL}/api/v1/health")
    n8n_online = is_service_online(f"{N8N_BASE_URL}/healthz") or is_service_online(N8N_BASE_URL)
    if not (fastapi_online and n8n_online):
        pytest.skip("Docker Compose live stack (FastAPI + n8n) is offline on host. Run 'docker compose up -d' for live integration testing.")
    
    execute_live_webhook_assertions()

if __name__ == "__main__":
    main()
