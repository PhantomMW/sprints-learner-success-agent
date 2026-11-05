<div align="center">

# 🚀 Sprints AI-Powered Learner Success & Support Agent
### End-to-End Predictive Modeling, Deep Sequence Forecasting & Event-Driven Agentic Workflow
**Client:** [Sprints](https://sprints.ai) | **Engagement:** 2-Week Capstone Project (Graduation Deliverable)  
**Track:** AI & Machine Learning - Intermediate | **Author:** Mohammed Wael ([@PhantomMW](https://github.com/PhantomMW))

---

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![n8n](https://img.shields.io/badge/n8n-Event--Driven%20RAG-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8%20NER-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
<br/>
[![Release](https://img.shields.io/badge/Release-v1.2.0-059669?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhantomMW/sprints-learner-success-agent/releases/tag/v1.2.0)
[![CI Pipeline](https://github.com/PhantomMW/sprints-learner-success-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/PhantomMW/sprints-learner-success-agent/actions/workflows/ci.yml)
[![F1-Score](https://img.shields.io/badge/F1_Score-1.0000_(Target_%E2%89%A5_0.82)-success?style=for-the-badge)](#-acceptance-criteria-verification-matrix)
[![Latency SLA](https://img.shields.io/badge/Latency-154ms_(SLA_%3C_3.5s)-blueviolet?style=for-the-badge)](#-acceptance-criteria-verification-matrix)
[![Zero False Negatives](https://img.shields.io/badge/False_Negatives-0_(Recall_1.00)-brightgreen?style=for-the-badge)](#-acceptance-criteria-verification-matrix)

<p align="center">
  <a href="#-executive-summary--business-situation"><b>Business Problem</b></a> •
  <a href="#-system-architecture"><b>Architecture</b></a> •
  <a href="#-acceptance-criteria-verification-matrix"><b>Quality Bar & Rubric</b></a> •
  <a href="#-quickstart--execution-guide"><b>Quickstart</b></a> •
  <a href="#-capstone-presentation--video-walkthrough"><b>Reports & Demo</b></a> •
  <a href="https://github.com/PhantomMW/sprints-learner-success-agent/releases/tag/v1.2.0"><b>Download Release Assets</b></a>
</p>

</div>

---

## 📌 Executive Summary & Business Situation

As **Sprints** rapidly scales its intensive technology bootcamps across Egypt and the MENA region, the massive volume of the nationwide **"AI Pioneers"** cohort created a critical **dual operational bottleneck**:
1. **Silent Dropouts:** LMS telemetry and assessment metrics are fragmented. Struggling students quietly fall behind without early warning indicators, slipping under the radar before mentors can intervene.
2. **Support Desk Congestion:** Repetitive, syntax-level curriculum questions flood the support desk, causing delays in human response times for high-urgency student blockers.

To solve this, Sprints leadership commissioned the development of an **end-to-end automated Learner Success Agent**. This system bridges relational analytics, supervised deep learning, temporal sequence modeling, and an event-driven agentic RAG workflow:
* **Sprint 1 (Practical Data Analytics):** Normalized 6-table SQLite database (`database/sprints_learners.db`) with analytical queries to extract interaction logs and detect progressive grade degradation ($\ge 15$ pts).
* **Sprint 2 (Intro to Machine Learning):** PyTorch Supervised Feed-Forward Neural Network (`models/risk_classifier.pth`) classifying learners as **High Risk** vs **Low Risk**, achieving **F1 = 1.000** and **0 False Negatives** (exceeding the $\ge 0.82$ target).
* **Sprint 3 (Deep Learning):** PyTorch 2-layer LSTM Sequence Model (`models/sequence_predictor.pth`) processing 5-week quiz trajectories $[q_1..q_5]$ to forecast final capstone project outcomes ($R^2 = 0.8229$, RMSE = $7.63$ pts, MAE = $5.59$ pts), leveraging its **forget gate** to discard early-week noise and eliminate vanishing gradients.
* **Sprint 4 (NLP & Generative AI):** Custom spaCy Named Entity Recognition (NER), TextBlob sentiment analysis, Gensim Word2Vec dense embeddings (`data/processed_embeddings.bin`), empirical FastText subword n-grams benchmark, and dense Semantic Vector Retrieval.
* **Orchestration (n8n & FastAPI):** Event-driven n8n workflow (`n8n_workflows/learner_success_agent.json`) executing a Deliberative Agent routing policy (escalating distressed queries to human mentors within **14 ms** and answering curriculum queries via Semantic Retrieval in **152 ms**, strictly meeting the $< 3.5$s SLA).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer [1. Storage & Ingestion Layer]
        CSV["sprints_learner_data.csv\n(2,000 Learners)"] --> IMP["pandas Imputation Engine\n(Forward Fill & City Median)"]
        IMP --> DB[("sprints_learners.db\n(SQLite 3NF Schema)")]
        DB --> SQL_EXT["setup_and_queries.sql\n(Grade Degradation & Regional KPIs)"]
    end

    subgraph Modeling Layer [2. Predictive PyTorch Core]
        IMP --> FFNN["PyTorch Risk Classifier (MLP)\n(models/risk_classifier.pth)\nF1 = 1.000 | Recall = 1.000"]
        IMP --> LSTM["PyTorch Sequence LSTM\n(models/sequence_predictor.pth)\nR² = 0.8229 | Forget Gate Dynamics"]
        FFNN --> PCA_PROJ["Latent Space PCA Projection\n(Behavioral Cluster Validation)"]
    end

    subgraph NLPLayer [3. NLP & Semantic Intelligence]
        FEEDBACK["Unstructured Student Inquiries"] --> SPACY["spaCy Custom EntityRuler\n(Curriculum Modules & Dates)"]
        FEEDBACK --> W2V["Gensim Word2Vec Model\n(models/word2vec.model)"]
        FEEDBACK --> TEXTBLOB["TextBlob Sentiment Analysis\n(Polarity & Subjectivity)"]
    end

    subgraph Agentic Orchestration [4. n8n Event-Driven Deliberative Agent]
        WEBHOOK["HTTP Webhook Node\nPOST /webhook/learner-query"] --> ROUTER{"Deliberative Router\n(Polarity < -0.15 OR Distress?)"}
        ROUTER -->|Negative Sentiment| ESCALATE["Human Mentor Priority Queue\n(Critical Escalation: 9 ms)"]
        ROUTER -->|Neutral / Positive| RAG["Curriculum Knowledge RAG\n(Synthesized Pedagogical Answer: 154 ms)"]
        ESCALATE --> RESP["Respond to Webhook\n(Unified JSON Response < 3.5s SLA)"]
        RAG --> RESP
    end

    SQL_EXT -.-> FFNN
    SPACY -.-> ROUTER
    TEXTBLOB -.-> ROUTER
```

---

## 📊 Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    COHORTS ||--o{ LEARNERS : "enrolls"
    LEARNERS ||--o{ QUIZ_GRADES : "earns"
    LEARNERS ||--o{ SUPPORT_TICKETS : "submits"
    LEARNERS ||--o{ INTERACTION_LOGS : "generates"
    COURSE_MODULES ||--o{ QUIZ_GRADES : "assesses"
    COURSE_MODULES ||--o{ SUPPORT_TICKETS : "references"

    COHORTS {
        varchar(20) cohort_id PK
        varchar(100) cohort_name
        varchar(50) track
        date start_date
        date end_date
    }
    LEARNERS {
        varchar(20) learner_id PK
        varchar(20) cohort_id FK
        varchar(100) full_name
        varchar(120) email UK
        varchar(50) cohort_city
        varchar(20) enrollment_status
        boolean dropped_out
    }
    COURSE_MODULES {
        varchar(20) module_id PK
        varchar(100) module_name
        int sprint_number
        int week_number
        varchar(20) difficulty_level
    }
    QUIZ_GRADES {
        int grade_id PK
        varchar(20) learner_id FK
        varchar(20) module_id FK
        int week_number
        float score
    }
    SUPPORT_TICKETS {
        varchar(30) ticket_id PK
        varchar(20) learner_id FK
        text query_text
        float sentiment_polarity
        varchar(20) status
        varchar(15) priority
    }
    INTERACTION_LOGS {
        int log_id PK
        varchar(20) learner_id FK
        varchar(50) action_type
        varchar(100) lms_page
        int session_duration_sec
    }
```

---

## 🎯 Acceptance Criteria Verification Matrix

| Evaluation Criterion | Minimum Client Threshold | Outcome Achieved by Sprints Agent | Performance Status |
| :--- | :--- | :--- | :---: |
| **Model F1 Score (Exp A: W1-W3 Proactive)** | $\ge 0.82$ on Test Set | **0.9714 (Recall: 0.9855, ROC-AUC: 0.9996)** | 🟢 **PASSED (EARLY RISK INTERVENTION)** |
| **Model F1 Score (Exp B: W1-W5 Full)** | $\ge 0.82$ on Test Set | **1.0000 (Recall: 1.0000, 0 False Negatives)** | 🟢 **OPTIMAL SAFETY NET** |
| **Workflow Latency** | $< 3.5$ seconds (3,500 ms) | **154.5 ms (Max: 296 ms)** | 🟢 **22x FASTER THAN SLA** |
| **Sequence Modeling (No Target Leakage)** | LSTM on real submissions ($N=1,639$) | **$R^2 = 0.8229$, RMSE = 7.63 pts, MAE = 5.59** | 🟢 **PASSED** |
| **spaCy NER Benchmark (50 Samples)** | Module & Date Extraction | **Module F1: 0.8125 \| Date F1: 0.9091** | 🟢 **PASSED** |
| **Word2Vec vs FastText OOV Test** | Empirical Benchmark | Tested: Word2Vec KeyError on typos vs FastText n-grams | 🟢 **PASSED** |
| **Semantic Vector RAG** | Dense Vector Retrieval | Cosine similarity ranking over syllabus vectors | 🟢 **PASSED** |
| **Egyptian Arabic Dialect Distress NLP** | $\ge 0.90$ Precision & Recall (26 Cases) | **Precision: 1.0000 \| Recall: 1.0000 \| F1: 1.0000** | 🟢 **OPTIMAL DISAMBIGUATION (0 FP, 0 FN)** |
| **Technical vs Distress Disambiguation** | Prevent false escalation of technical tasks | Tested: 'مش قادر أفتح الملف' -> Automated RAG (0 FP) | 🟢 **PASSED** |
| **Ticket Deduplication & Idempotency** | Prevent duplicate tickets on retried requests | Tested in `/api/v1/tickets/escalate` and SQLite DB | 🟢 **PASSED** |
| **Deliberative Routing & DB Persistence**| Escalate negative sentiment | Escalated tickets persisted to SQLite `support_tickets` | 🟢 **PASSED** |

---

## 📂 Project Submission Tree

```
sprints_capstone/
├── data/
│   ├── sprints_learner_data.csv       # Cleaned 2,000-learner dataset (LRN-1001..3000)
│   └── processed_embeddings.bin       # Gensim dense Word2Vec binary embeddings
├── database/
│   ├── schema.sql                     # Modular 3NF DDL schema definition & indexes
│   ├── seed.sql                       # Clean reference seed data
│   ├── analytics.sql                  # Analytical queries including Query 5 CTE fix
│   ├── schema_design.erd              # Detailed ERD specifications and data dictionary
│   ├── setup_and_queries.sql          # Unified SQL setup and verified queries
│   └── sprints_learners.db            # SQLite relational database (2,000 learners, 9,101 quizzes)
├── models/
│   ├── risk_classifier.pth            # Supervised PyTorch Risk MLP (Exp A F1=0.9714 / Exp B F1=1.000)
│   ├── sequence_predictor.pth         # PyTorch Sequence LSTM (R²=0.8229, RMSE=7.63 pts)
│   └── word2vec.model                 # Trained Gensim curriculum Word2Vec model
├── notebooks/
│   ├── 01_eda_and_sql_sprint1.ipynb   # Executed EDA, SQL Query 5 CTE, & governorate charts
│   ├── 02_ml_dl_modeling_sprint2_3.ipynb # PyTorch models, LSTM gates, & latent space PCA
│   └── 03_nlp_processing_sprint4.ipynb  # spaCy NER benchmark, Word2Vec vs FastText, & n8n simulation
├── n8n_workflows/
│   └── learner_success_agent.json     # Production n8n workflow with Webhook, Router & DB Escalation
├── docs/
│   ├── Sprints_Model_Evaluation_Report.pdf # Comprehensive 4-page technical evaluation PDF
│   ├── Sprints_Architecture_Presentation.pdf # 10-slide executive presentation PDF
│   ├── confusion_matrix.png           # Zero-false-negative confusion matrix plot
│   ├── latent_space_distribution.png  # Penultimate layer PCA cluster projection
│   └── lstm_sequence_predictions.png  # 5-week trajectory vs capstone forecast plot
├── scripts/
│   ├── populate_db_from_csv.py        # Populates 2,000 learners and 9,101 quizzes into SQLite
│   ├── train_pytorch.py               # Leak-free PyTorch training & evaluation script
│   ├── nlp_pipeline.py                # spaCy NER benchmark, Word2Vec, Semantic RAG & Router
│   ├── test_db.py                     # Database execution verification script
│   ├── test_api_and_n8n.py            # Automated API latency & Deliberative Agent test
│   ├── build_notebooks.py             # Notebook synthesis and execution script
│   ├── build_report_pdf.py            # Technical report PDF compiler
│   ├── build_presentation_pdf.py      # 10-slide presentation PDF compiler
│   └── generate_demo_video.py         # Video generator script
├── api/
│   └── app.py                         # Production FastAPI inference microservice with SQLite persistence
├── demo/
│   ├── Demo_Walkthrough.mp4           # 3.5-minute narrated demo walkthrough video
│   └── DEMO_SCRIPT.md                 # Verbatim walkthrough script & timestamps
├── Demo_Walkthrough.mp4               # Root copy of video walkthrough
├── Dockerfile                         # Container build definition for API service
├── docker-compose.yml                 # Multi-container orchestration (FastAPI + n8n)
├── requirements.txt                   # Pinned production Python dependencies
└── README.md                          # Comprehensive project documentation
```

---

## ⚡ Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/PhantomMW/sprints-learner-success-agent.git
cd sprints-learner-success-agent

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 2. Run Database Setup & SQL Queries (Sprint 1)
```bash
python scripts/test_db.py
```

### 3. Train PyTorch Models (Sprints 2 & 3)
```bash
python scripts/train_pytorch.py --epochs 60 --batch_size 32
```

### 4. Execute NLP Pipeline (Sprint 4)
```bash
python scripts/nlp_pipeline.py
```

### 5. Launch FastAPI Service & Automated Test Suites
```bash
# Start API server
python api/app.py

# Run end-to-end API & deliberative router integration test
python scripts/test_api_and_n8n.py

# Run feature parity unit tests (6 boundary cases comparing DataFrame vs Inference extractor)
python tests/test_feature_parity.py

# Run n8n JavaScript code nodes & graph routing execution test (Node.js VM engine)
node tests/test_n8n_workflow_engine.js
```

### 6. Launch via Docker Compose (n8n + API)
```bash
docker compose up --build -d

# Verify live n8n orchestrator integration & SQLite webhook persistence
python tests/test_n8n_docker_integration.py
```
* Access **n8n Web UI** at: `http://localhost:5678`
* Import workflow from `n8n_workflows/learner_success_agent.json`
* Access **FastAPI Docs** at: `http://localhost:8000/docs`

---

## 🔬 Deep Dive: Theoretical & Architectural Highlights

### 1. How the Hidden State in LSTM Mitigates Data Loss & Vanishing Gradients
A major pitfall of standard RNNs on multi-week academic trajectories is the **exponential decay of backpropagated gradients**. When modeling a 5-week sequence $[q_1, q_2, q_3, q_4, q_5]$, an isolated bad week (e.g. sickness or local power outages in Week 1) could either dominate or vanish in traditional architectures.
* **The Forget Gate ($f_t$):** Evaluates $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$. For temporary noise in early weeks, $f_t$ selectively dampens transient errors while maintaining cumulative mastery.
* **Additive Constant Error Carousel ($C_t$):** The update rule $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ provides a linear shortcut for gradient propagation, entirely circumventing vanishing gradients.
* **Hidden State ($h_t$):** Filtered through $o_t \odot \tanh(C_t)$, the final hidden state $h_5$ encodes an accurate, noise-filtered representation of student velocity used to forecast capstone outcomes ($R^2 = 0.8229$, RMSE = $7.63$ pts).

### 2. Scientific Evaluation: Experiment B ($F_1 = 1.000$) vs Experiment A ($F_1 = 0.9714$)
A rigorous empirical examination of the synthetic cohort dataset explains the performance dynamics between the two experiments:
* **The Week 5 Dropout Reality (Experiment B):** In the student telemetry dataset, learners who drop out overwhelmingly stop submitting quizzes by the end of the cohort: **74.2% (345 / 465) of dropped students have a missing/null Week 5 quiz**, and their average Week 5 score drops to **$10.0$ pts** (compared to **$86.0$ pts** for retained learners). At Week 5, dropout is an accomplished fact; hence, Experiment B acts as a *retrospective safety net* achieving $F_1 = 1.000$.
* **True Proactive Early Intervention (Experiment A):** To provide real pedagogical value, intervention must occur *before* students abandon the bootcamp. Experiment A uses **Weeks 1-3 only**, when students are still engaged (Week 1 quiz mean: 54.6 for at-risk vs 83.8 for retained). Achieving **Test $F_1 = 0.9714$** and **Test Recall = $0.9855$ (0.9996 ROC-AUC)** proves that proactive early flagging is highly feasible at Week 3.

### 3. Latent Space Manifold Separability
To ensure the supervised risk classifier was not overfitting to spurious tabular artifacts, we extracted the penultimate 16-dimensional activation vectors from the MLP and projected them onto a 2D manifold via PCA. The resulting distribution demonstrates **clear geometric separation** between the High Risk cluster and the Retained Learners cluster, validating that our classification boundaries reflect true underlying behavioral dynamics.

### 4. Word2Vec vs. FastText Empirical Benchmark
* **Gensim Word2Vec:** Operates on atomic whole-word tokens (Skip-Gram). Ultra-compact model footprint (~308 KB) and instantaneous $O(1)$ dictionary lookups make it ideal for the strict sub-second latency SLA of our real-time n8n Webhook.
* **FastText Subwords:** Decomposes words into character n-grams ($3 \le n \le 6$, compact bucket=10,000). Crucial for out-of-vocabulary (OOV) tokens and student spelling mistakes. In our live Gensim benchmark, Word2Vec raises `KeyError` on `pytorxh` and `lstmm`, whereas FastText reconstructs valid subword embeddings with **0.9113** and **0.8016** cosine similarity to the canonical terms.
* **Strategic Dual Architecture:** Word2Vec powers the low-latency real-time inference loop; FastText runs in the asynchronous forum scraper pipeline.

### 5. Semantic Vector Retrieval vs. Full Generative LLM RAG
To ensure architectural transparency:
* **Current Production System:** Implements a **Dense Semantic Vector Retrieval Grounding Engine**. Incoming queries are mapped to 100-dimensional Word2Vec space and matched against verified curriculum documentation chunks via **Cosine Similarity** with a confidence threshold ($0.25$) and graceful fallback. This architecture guarantees deterministic pedagogical accuracy and sub-second speed (**~150 ms**) with zero external LLM API cost or hallucination risk.
* **Generative Expansion Ready:** The retrieved curriculum chunk and student metadata are pre-formatted so an enterprise generative model (e.g. Gemini 1.5 Pro) can be plugged directly into the n8n downstream pipeline for open-ended conversational synthesis.

---

## 🎓 Capstone Presentation & Video Walkthrough

* **Technical Evaluation Report:** [`docs/Sprints_Model_Evaluation_Report.pdf`](docs/Sprints_Model_Evaluation_Report.pdf) (4-page technical publication)
* **Architecture Presentation Deck:** [`docs/Sprints_Architecture_Presentation.pdf`](docs/Sprints_Architecture_Presentation.pdf) (10-slide executive presentation)
* **Demo Walkthrough Video:** [`demo/Demo_Walkthrough.mp4`](demo/Demo_Walkthrough.mp4) (3.5-minute video walkthrough)
* **Verbatim Demo Script:** [`demo/DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md)
* **GitHub Official Release:** [v1.0.0 Graduation Release](https://github.com/PhantomMW/sprints-learner-success-agent/releases/tag/v1.2.0)

---

## 👨‍💻 Author & Acknowledgments
* **Author:** Mohammed Wael ([@PhantomMW](https://github.com/PhantomMW))
* **Organization:** Sprints (sprints.ai)
* **Lead AI Researcher:** Tarek Mahmoud
* **Product Manager:** Salma Youssef
