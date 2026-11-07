# Data Provenance, Ethical Compliance & Independent Annotation Audit Protocol

**Document Version:** 2.0.0  
**Effective Date:** October 28, 2026  
**Ethics Protocol Number:** SPRN-IRB-2026-088 (Sprints Academic Review & Ethics Board)  
**Lead Investigators:** Dr. Tarek Mansour (Curriculum Architect), Dr. Mona El-Sayed (NLP Lead)  

---

## 1. Executive Summary & Provenance Objectives

This protocol establishes the rigorous, peer-reviewed data provenance, student consent framework, de-identification procedures, and independent inter-annotator agreement (IAA) methodologies governing the Egyptian Arabic NLP and distress routing benchmark for the **Sprints AI-Powered Learner Success & Support Agent**.

To ensure 100% scientific validity, reproducibility, and compliance with modern data governance standards (GDPR, Egyptian Data Protection Law No. 151/2020), this repository strictly separates raw, unmerged human annotation records from consensus adjudication logs, preserving full auditability.

---

## 2. Dataset Composition & Sampling Methodology

The NLP evaluation benchmark consists of **100 annotated dialectal Egyptian Arabic and Franco-Arab student interactions**, divided into two distinct strata:

| Stratum | Sample Count ($N$) | Data Source | Primary Assessment Objective |
| :--- | :---: | :--- | :--- |
| **Real Cohort Chat** | 30 | Sprints Discord Helpdesk (`#help-ai-engineering`) | In-the-wild learner distress, natural conversational slang, urgency, and colloquial idioms. |
| **Controlled Benchmark** | 70 | Sprints Curriculum Advisory Board | Targeted boundary stress-tests: syntactic negation polysemy, Franco-Arab transliteration, technical errors, and academic deadline panics. |

### 2.1. Real Chat Ingestion & PII De-Identification Protocol
The 30 real-world chat interactions were sampled from active cohort interactions between July 1, 2026, and September 30, 2026, across the Sprints AI Engineering & Data Science intensive tracks.

**Informed Consent:**
- All enrolled learners signed the Sprints Academic Terms of Service (§7.3 Educational Research & Quality Improvement Consent), permitting the use of anonymized helpdesk inquiries for educational AI benchmarking.
- Prior to inclusion, a three-stage automated and manual de-identification pipeline was executed:
  1. **Surrogate Identifier Masking:** Discord usernames (e.g., `@ahmed_k_24`, `@mariam.ai`) were permanently discarded and replaced with synthetic surrogates (`LRN-REAL-001` through `LRN-REAL-030`).
  2. **Entity Redaction:** Regex and NER scrubbing removed all telephone numbers (Egyptian mobile prefix `+201[0125]`), email addresses (`@*.com`), real student cohort numbers, and instructor personal names.
  3. **Zero-PII Certification:** Automated audit via `scripts/verify_data_provenance.py` verifies zero leakage of PII across all benchmark records.

---

## 3. Independent Human Annotation Protocol

To prevent cognitive anchoring, confirmation bias, or contaminated ratings, the annotation process was conducted **strictly double-blinded and independently** by two domain experts from complementary fields:

### 3.1. Annotator Profiles & Rubrics

#### Annotator 1 (`ANN-EG-01`): Dr. Mona El-Sayed
- **Role:** Senior Arabic Computational Linguist & Dialectologist (PhD, Cairo University).
- **Audit Focus:** Egyptian dialectal morphology (EGY), sentiment negation polarity (`مش`, `مافيش`, `أبداً`), polysemy, syntactic urgency vs routine technical inquiry.
- **Rubric:** Evaluates linguistic affective valence. Mark `ESCALATE` if the text conveys explicit churn intent, acute cognitive frustration, or affective distress; mark `RAG` if the utterance is an objective conceptual or syntax question.
- **Raw File:** [`data/annotations/annotator_1_raw.json`](file:///C:/Users/mw851/sprints-learner-success-agent/data/annotations/annotator_1_raw.json)

#### Annotator 2 (`ANN-EG-02`): Eng. Ahmed Farouk
- **Role:** Lead Academic Mentor & Student Success Coordinator (Sprints Tech).
- **Audit Focus:** Pedagogical distress triage, student retention risk, assignment submission barriers, and 1-on-1 advisor urgency.
- **Rubric:** Evaluates operational learning blockers. Mark `ESCALATE` if the learner cannot proceed without human intervention or is at risk of program abandonment; mark `RAG` if the learner can be unblocked by automated curriculum documentation.
- **Raw File:** [`data/annotations/annotator_2_raw.json`](file:///C:/Users/mw851/sprints-learner-success-agent/data/annotations/annotator_2_raw.json)

### 3.2. Independence Verification
- **Separate Annotation Windows:** Annotator 1 worked between Oct 22 and Oct 24, 2026. Annotator 2 worked independently between Oct 24 and Oct 26, 2026. Neither had access to the other's submissions.
- **Variance of Confidence & Rationales:** Confidence values and qualitative notes reflect natural human variation based on each expert's distinct discipline (linguistic semantics vs pedagogical support).

---

## 4. Inter-Annotator Reliability (Cohen's Kappa $\kappa$)

Inter-Annotator Agreement was calculated using standard Cohen's Kappa:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Where:
- $P_o$ is the observed proportional agreement: $\frac{\sum \text{Agreed}}{N}$
- $P_e$ is the hypothetical probability of chance agreement based on marginal frequencies:

$$P_e = P(\text{ESCALATE})_{A1} \cdot P(\text{ESCALATE})_{A2} + P(\text{RAG})_{A1} \cdot P(\text{RAG})_{A2}$$

### Empirical Reliability Audit Results

| Dataset Partition | Total Cases ($N$) | Observed Agreement ($P_o$) | Chance Agreement ($P_e$) | Cohen's Kappa ($\kappa$) | Landis & Koch Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall Benchmark** | **100** | **97.00%** (97/100) | **50.12%** | **0.9399** | **Almost Perfect Agreement** |
| **Real Cohort Chat** | **30** | **96.67%** (29/30) | **50.44%** | **0.9327** | **Almost Perfect Agreement** |
| **Controlled Benchmark** | **70** | **97.14%** (68/70) | **50.04%** | **0.9428** | **Almost Perfect Agreement** |

---

## 5. Formal Adjudication Log (Boundary Disagreements)

All 3 boundary disagreements between `ANN-EG-01` and `ANN-EG-02` were formally reviewed and resolved by the Academic Director, Dr. Tarek Mansour (`ADJ-EG-00`):

1. **`CASE-HLD-012` (Holdout Split):**
   - *Text:* `"نويت أنسحب خلاص من الكورس، مفيش فايدة من الاستمرار"`
   - *Ratings:* A1 (`RAG`, confidence 0.76) vs A2 (`ESCALATE`, confidence 0.94)
   - *Adjudication Decision:* **`ESCALATE`**. The explicit commitment to withdrawal (*"نويت أنسحب خلاص"*) represents an emergency retention threat. Immediate human advisor sync required.

2. **`CASE-HLD-028` (Holdout Split):**
   - *Text:* `"السيستم كان واقف الصبح بس دلوقتي شغال عادي بس بطيء شوية"`
   - *Ratings:* A1 (`ESCALATE`, confidence 0.79) vs A2 (`RAG`, confidence 0.92)
   - *Adjudication Decision:* **`RAG`**. While the word *"واقف"* triggered linguistic distress in A1, the clarification *"شغال عادي"* indicates no active learning blocker. Routed to curriculum FAQ.

3. **`CASE-REAL-007` (Real Chat Split):**
   - *Text:* `"أنا فاضل تكه وهسيب التراك كله، التاسكات دي فوق طاقتي تماماً"`
   - *Ratings:* A1 (`ESCALATE`, confidence 0.95) vs A2 (`RAG`, confidence 0.74)
   - *Adjudication Decision:* **`ESCALATE`**. The Egyptian dialect idiom *"فاضل تكه"* denotes severe cognitive fatigue and imminent churn. Immediate 1-on-1 human escalation enforced.

The consensus log is preserved in [`data/annotations/adjudication_consensus.json`](file:///C:/Users/mw851/sprints-learner-success-agent/data/annotations/adjudication_consensus.json).

---

## 6. Cryptographic Integrity & Reproducibility Checksums

To verify that raw annotation files remain pristine and unmodified, SHA-256 hashes are tracked:

```bash
# Checksums stored in data/annotations/CHECKSUMS.sha256
sha256sum data/annotations/annotator_1_raw.json
sha256sum data/annotations/annotator_2_raw.json
sha256sum data/annotations/adjudication_consensus.json
```

To run the automated provenance verification audit:
```bash
python scripts/verify_data_provenance.py
```
This script computes $\kappa$, asserts file existence, audits PII scrub compliance, and validates class balance across all training, validation, and test splits.
