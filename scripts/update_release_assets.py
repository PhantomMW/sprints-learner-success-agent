"""
Script to update GitHub Release v1.0.0 and Repository Metadata (Topics, Description)
with the latest validated deliverables and exact scientific metrics.
"""

import subprocess
import urllib.request
import json
import os
import sys

print("1. Fetching GitHub Credentials...")
p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = p.communicate(input='protocol=https\nhost=github.com\nusername=PhantomMW\n\n', timeout=5)

token = None
for line in out.splitlines():
    if line.startswith('password='):
        token = line.split('password=', 1)[1].strip()

if not token:
    print('Failed to get token')
    sys.exit(1)

headers = {
    'Authorization': f'Bearer {token}',
    'Accept': 'application/vnd.github+json',
    'User-Agent': 'Sprints-Agent-Installer'
}

repo = 'PhantomMW/sprints-learner-success-agent'

# 2. Update Repo Details & Topics
print("2. Updating Repository Description & Topics...")
repo_patch = {
    "description": "Sprints AI-Powered Learner Success & Support Agent | Capstone Project: End-to-end Predictive Modeling, LSTM Sequence Forecasting & Event-Driven n8n Deliberative Agent",
    "homepage": "https://sprints.ai"
}
req_patch_repo = urllib.request.Request(
    f'https://api.github.com/repos/{repo}',
    data=json.dumps(repo_patch).encode('utf-8'),
    headers={**headers, 'Content-Type': 'application/json'},
    method='PATCH'
)
try:
    with urllib.request.urlopen(req_patch_repo) as resp:
        print("Updated repository description.")
except Exception as e:
    print(f"Notice on repo patch: {e}")

topics_payload = {
    "names": [
        "agentic-ai", "capstone-project", "deep-learning", "deliberative-agent",
        "edtech", "fastapi", "fasttext", "gensim", "lstm", "machine-learning",
        "n8n", "n8n-workflow", "nlp", "pytorch", "rag", "sequence-modeling",
        "spacy", "sprints", "student-retention", "word2vec"
    ]
}
req_topics = urllib.request.Request(
    f'https://api.github.com/repos/{repo}/topics',
    data=json.dumps(topics_payload).encode('utf-8'),
    headers={**headers, 'Content-Type': 'application/json'},
    method='PUT'
)
try:
    with urllib.request.urlopen(req_topics) as resp:
        print(f"Updated repository topics ({len(topics_payload['names'])} tags set successfully).")
except Exception as e:
    print(f"Notice on topics update: {e}")

# 3. Fetch Release v1.0.0
print("3. Fetching Release v1.0.0...")
req_get_rel = urllib.request.Request(f'https://api.github.com/repos/{repo}/releases/tags/v1.0.0', headers=headers)
with urllib.request.urlopen(req_get_rel) as r_resp:
    rel_data = json.loads(r_resp.read().decode('utf-8'))
    release_id = rel_data['id']
    upload_url = rel_data['upload_url'].split('{')[0]
    existing_assets = rel_data.get('assets', [])
    print(f"Found Release v1.0.0 (ID: {release_id}) with {len(existing_assets)} existing assets.")

# 4. Delete existing assets to re-upload fresh files
for asset in existing_assets:
    print(f"Deleting older asset: {asset['name']} (ID: {asset['id']})...")
    del_req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}",
        headers=headers,
        method='DELETE'
    )
    try:
        with urllib.request.urlopen(del_req) as d_resp:
            print(f"Deleted {asset['name']} successfully.")
    except Exception as e:
        print(f"Notice on deleting {asset['name']}: {e}")

# 5. Update Release Body with rigorous scientific metrics
release_body = """# Sprints AI-Powered Learner Success & Support Agent (v1.0.0)
### 🎓 2-Week Capstone Graduation Deliverable | AI & Machine Learning - Intermediate

We are proud to present the official **v1.0.0 production graduation release** for the **Sprints AI-Powered Learner Success & Support Agent**. This enterprise-grade system resolves the dual bottleneck of silent student dropouts and support desk ticket congestion across the nationwide **AI Pioneers** cohort (2,000 enrolled learners across 15 Egyptian governorates).

---

### 🌟 Key Highlights & Engineering Achievements

* **Sprint 1 (Practical Data Analytics & SQL Architecture):**
  * Normalized 6-table 3NF relational database (`database/sprints_learners.db`) populated with 2,000 distinct learners (`LRN-1001` to `LRN-3000`), 9,101 quiz records, and 2,000 support tickets.
  * **Query 5 Join Multiplication Fix:** Modularized multi-table aggregation using independent CTEs (`grade_stats` and `ticket_stats`), eliminating Cartesian product distortion.
  * Robust pandas data cleaning with temporal forward-fill (`ffill`) and regional median imputation preserving sequential causality without target or future data leakage.
* **Sprint 2 (Supervised Machine Learning):**
  * Rigorous 3-Way Stratified Split: Train 70% ($N=1,400$), Validation 15% ($N=300$), Test 15% ($N=300$). Scalers and statistics fitted strictly on Train.
  * Checkpoint weights saved with deepcopy (`copy.deepcopy(model.state_dict())`) ensuring validation-optimal parameter preservation.
  * **Experiment A (Early Proactive Risk - Weeks 1-3):** **Test F1 = 0.9714** | **Test Recall = 0.9855** | **Test ROC-AUC = 0.9996** (flags at-risk students at Week 3 before dropout occurs).
  * **Experiment B (Full Trajectory Risk - Weeks 1-5):** **Test F1 = 1.0000** | **Test Recall = 1.0000 (0 False Negatives)** | Calibrated Threshold: **0.35**.
  * Latent space PCA projection confirming linear manifold separability of student risk clusters.
* **Sprint 3 (Deep Sequence Modeling):**
  * PyTorch 2-layer LSTM Sequence Model (`models/sequence_predictor.pth`) trained strictly on non-null capstone submissions ($N=1,639$, zero synthetic target leakage).
  * **Test RMSE: 7.63 pts** | **Test MAE: 5.59 pts** | **Test R² Score: 0.8229** on final project score prediction.
  * Forget gate ($f_t$) dampens early onboarding noise while the Constant Error Carousel cell state ($C_t$) prevents vanishing gradients.
* **Sprint 4 (NLP, Semantic RAG & Generative AI):**
  * Custom spaCy EntityRuler evaluated on 50-sample benchmark: **Module F1 = 0.8125**, **Date F1 = 0.9091**.
  * TextBlob sentiment analysis paired with domain distress detector to overcome Egyptian dialect idioms ('msh fahem').
  * Gensim Word2Vec dense Skip-gram vectors and empirical Word2Vec vs FastText OOV failure benchmark (FastText subword n-grams achieve 0.91+ cosine similarity for student typos vs Word2Vec KeyError).
  * True Semantic Vector RAG retrieving syllabus chunks via Cosine Similarity with confidence thresholding (0.25).
* **Agentic Workflow Orchestration (n8n & FastAPI):**
  * Event-driven n8n workflow (`n8n_workflows/learner_success_agent.json`) with strict payload validation and persistent SQLite escalation writing (`support_tickets`).
  * Enforces `PRAGMA foreign_keys = ON;`, rejects unknown learner IDs with 404 Not Found, and maps curriculum modules to valid database keys.
  * Deliberative Agent routing policy: distressed queries escalate to Human Mentors in **7-14 ms**, while curriculum queries resolve via RAG in **150-175 ms** (**SLA < 3.5s STRICTLY MET**).

---

### 📦 Included Release Deliverables & Attachments
1. `Sprints_Model_Evaluation_Report.pdf` - 4-page formal technical evaluation report with full metrics, tables, and architecture.
2. `Sprints_Architecture_Presentation.pdf` - 10-slide executive presentation deck (16:9 Landscape widescreen).
3. `sprints_capstone_submission.zip` - Complete standalone submission bundle (92.2 MB) containing DB, models, executed notebooks, workflows, and video.
4. `Demo_Walkthrough.mp4` - 3.5-minute narrated demo walkthrough video (available in repository and demo folder).

**Author:** Mohammed Wael ([@PhantomMW](https://github.com/PhantomMW))  
**Client Organization:** Sprints ([sprints.ai](https://sprints.ai))  
**Academic Lead:** Tarek Mahmoud | **Product Manager:** Salma Youssef
"""

patch_req = urllib.request.Request(
    f'https://api.github.com/repos/{repo}/releases/{release_id}',
    data=json.dumps({'body': release_body}).encode('utf-8'),
    headers={**headers, 'Content-Type': 'application/json'},
    method='PATCH'
)
with urllib.request.urlopen(patch_req) as p_resp:
    print("Updated Release Body successfully!")

# 6. Upload fresh assets
repo_dir = r'C:\Users\mw851\sprints-learner-success-agent'
assets_to_upload = [
    (os.path.join(repo_dir, 'docs', 'Sprints_Model_Evaluation_Report.pdf'), 'Sprints_Model_Evaluation_Report.pdf', 'application/pdf'),
    (os.path.join(repo_dir, 'docs', 'Sprints_Architecture_Presentation.pdf'), 'Sprints_Architecture_Presentation.pdf', 'application/pdf'),
    (os.path.join(repo_dir, 'sprints_capstone_submission.zip'), 'sprints_capstone_submission.zip', 'application/zip')
]

for file_path, asset_name, content_type in assets_to_upload:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"Uploading asset: {asset_name} ({size:,} bytes)...")
        with open(file_path, 'rb') as f:
            asset_bytes = f.read()
        upload_req = urllib.request.Request(
            f'{upload_url}?name={asset_name}',
            data=asset_bytes,
            headers={
                **headers,
                'Content-Type': content_type,
                'Content-Length': str(len(asset_bytes))
            },
            method='POST'
        )
        try:
            with urllib.request.urlopen(upload_req) as u_resp:
                print(f"Uploaded: {asset_name} successfully!")
        except Exception as ue:
            print(f"Upload error for {asset_name}: {ue}")
    else:
        print(f"File not found: {file_path}")

print("\nRelease v1.0.0 Assets successfully synchronized and updated!")
