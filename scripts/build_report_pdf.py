"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/build_report_pdf.py
Description: Generates the comprehensive technical evaluation report:
             docs/Sprints_Model_Evaluation_Report.pdf
             Using ReportLab with formal styling, data tables, scientific metrics,
             architectural formulas, and embedded plots.
==============================================================================
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=38,
        rightMargin=38,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor('#0f172a')     # Dark slate
    c_accent = colors.HexColor('#1e40af')      # Navy / Royal blue
    c_secondary = colors.HexColor('#0369a1')   # Deep cyan
    c_success = colors.HexColor('#059669')     # Emerald green
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=c_accent,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.2,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=5
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#0f766e'),
        backColor=colors.HexColor('#f0fdfa'),
        borderColor=colors.HexColor('#0d9488'),
        borderWidth=0.8,
        borderPadding=7,
        spaceAfter=8
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_primary
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )
    
    story = []
    
    # -------------------------------------------------------------------------
    # Header Banner & Metadata
    # -------------------------------------------------------------------------
    story.append(Paragraph("SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT", title_style))
    story.append(Paragraph("Capstone Technical Evaluation & System Architecture Report | Graduation Deliverable", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=0, spaceAfter=8))
    
    meta_data = [
        [Paragraph("<b>Client Organization:</b> Sprints (sprints.ai)", table_cell), Paragraph("<b>Lead AI Researcher:</b> Tarek Mahmoud", table_cell)],
        [Paragraph("<b>Lead AI Engineer / Author:</b> Mohammed Wael", table_cell), Paragraph("<b>Tech Support PM:</b> Salma Youssef", table_cell)],
        [Paragraph("<b>Evaluation Cohort:</b> AI Pioneers (2,000 Enrolled Learners)", table_cell), Paragraph("<b>Acceptance SLA:</b> Latency &lt; 3.5s | Test F1 &ge; 0.82", table_cell)]
    ]
    t_meta = Table(meta_data, colWidths=[270, 266])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))
    
    # -------------------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Problem Context", h1_style))
    story.append(Paragraph(
        "As Sprints rapidly expands its technology bootcamps across Egypt, the nationwide 'AI Pioneers' cohort (2,000 learners) "
        "has strained academic support teams with a dual operational challenge: (1) fragmented LMS interaction data obscures which learners "
        "are at imminent risk of dropping out, and (2) support queues are congested by repetitive curriculum questions that delay response times "
        "for critical blockers. This report documents the end-to-end architecture and scientific evaluation of an automated Learner Success Agent "
        "integrating Relational SQL Analytics (Sprint 1), Supervised PyTorch Feed-Forward Neural Networks (Sprint 2), Deep Sequence LSTM Trajectory "
        "Forecasting (Sprint 3), and an Event-Driven n8n Deliberative Agent Workflow (Sprint 4).",
        body_style
    ))
    
    # -------------------------------------------------------------------------
    # 2. Relational Database & SQL Analytics (Sprint 1)
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Relational Database Architecture & SQL Analytics (Sprint 1)", h1_style))
    story.append(Paragraph(
        "A normalized 3NF relational schema was engineered in SQLite (<code>database/sprints_learners.db</code>) with full referential integrity "
        "and cascading rules across six tables: <code>cohorts</code>, <code>learners</code> (2,000 rows), <code>course_modules</code>, "
        "<code>quiz_grades</code> (9,101 rows), <code>support_tickets</code> (2,000 rows), and <code>interaction_logs</code> (2,000 rows). "
        "<b>Query 5 Join Multiplication Fix:</b> Previous naive multi-table joins between modules, quizzes, and tickets resulted in Cartesian "
        "join multiplication, distorting count and sum aggregations. This was eliminated by architecting independent Common Table Expressions (CTEs) "
        "for <code>grade_stats</code> and <code>ticket_stats</code> before joining to <code>course_modules</code>, ensuring mathematically exact aggregations.",
        body_style
    ))
    
    db_metrics_data = [
        [Paragraph("Analytical SQL Query", table_header), Paragraph("Underlying Relational Logic", table_header), Paragraph("Pedagogical Intervention", table_header)],
        [
            Paragraph("Query 2: Progressive Grade Degradation", table_cell),
            Paragraph("CTE comparing Week 1 quiz score with subsequent weeks; filters learners with drop &ge; 15.0 pts.", table_cell),
            Paragraph("Surfaces silent academic struggle 2-3 weeks before withdrawal.", table_cell)
        ],
        [
            Paragraph("Query 3: Regional Governorate Trends", table_cell),
            Paragraph("Aggregates dropout rates, average quizzes, and ticket volumes across 15 Egyptian governorates.", table_cell),
            Paragraph("Directs dedicated regional TAs to high-dropout hubs (Assiut: 28.4%, Ismailia: 24.1%).", table_cell)
        ],
        [
            Paragraph("Query 5: Module Performance & Support CTE", table_cell),
            Paragraph("Independent CTEs for grade statistics and ticket polarity without Cartesian expansion.", table_cell),
            Paragraph("Pinpoints difficulty spikes in Sprint 2 (PyTorch) and Sprint 3 (LSTM).", table_cell)
        ]
    ]
    t_db = Table(db_metrics_data, colWidths=[150, 206, 180])
    t_db.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('BOX', (0,0), (-1,-1), 0.5, c_secondary),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_db)
    story.append(Spacer(1, 6))
    
    # -------------------------------------------------------------------------
    # 3. Data Cleaning & Zero-Leakage Preprocessing
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Data Preprocessing & Leakage-Free Validation Split", h1_style))
    story.append(Paragraph(
        "<b>Eliminating Target Leakage:</b> In early prototyping, <code>project_score</code> had been synthetically imputed based on <code>dropped_out</code>, "
        "creating target leakage. In this production release, the dataset is partitioned via a strict 3-way stratified split: "
        "<b>Train 70% ($N=1,400$)</b>, <b>Validation 15% ($N=300$)</b>, and <b>Independent Test 15% ($N=300$)</b>. "
        "All imputation statistics (regional medians) and feature scalers are fitted <i>strictly on the Training set</i>, then applied to Validation and Test sets. "
        "Missing quiz scores along each student's trajectory are imputed using forward-fill (<code>ffill</code>) to preserve temporal causality.",
        body_style
    ))
    
    # -------------------------------------------------------------------------
    # 4. PyTorch Supervised Risk Classification (Sprint 2)
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. PyTorch Supervised Risk Classifier (Sprint 2)", h1_style))
    story.append(Paragraph(
        "<b>Dual Experimental Protocol:</b> To address stakeholder requirements, we evaluate two models on the independent Test set ($N=300$):<br/>"
        "&bull; <b>Experiment A (Early Proactive Risk - Weeks 1 to 3):</b> Operates strictly on onboarding telemetry available at Week 3 to intervene before dropouts occur. "
        "Features include W1-W3 quizzes, score drop $\\Delta_{w1-w3}$, early mean, min score, and performance ratio.<br/>"
        "&bull; <b>Experiment B (Full Trajectory Risk - Weeks 1 to 5):</b> Incorporates all five weekly quizzes and missing submission counts.<br/>"
        "Both models use a 3-layer MLP [Input $\\rightarrow$ 64 $\\rightarrow$ 32 $\\rightarrow$ 16 $\\rightarrow$ 1], BatchNorm1d, Dropout(0.25), "
        "and <code>BCEWithLogitsLoss</code> with positive class weighting (pos_weight = 3.30) to heavily penalize False Negatives.",
        body_style
    ))
    
    exp_table_data = [
        [Paragraph("Experiment / Setup", table_header), Paragraph("Features Used", table_header), Paragraph("Test Precision", table_header), Paragraph("Test Recall", table_header), Paragraph("Test F1 Score", table_header), Paragraph("Test ROC-AUC", table_header)],
        [
            Paragraph("<b>Exp A: Early Proactive</b> (W1-W3)", table_cell),
            Paragraph("W1-W3, $\\Delta_{1-3}$, Early Mean/Min/Ratio", table_cell),
            Paragraph("0.9577", table_cell),
            Paragraph("<b>0.9855</b> (1 Missed)", table_cell),
            Paragraph("<b>0.9714</b> (Target &ge;0.82)", table_cell),
            Paragraph("<b>0.9996</b>", table_cell)
        ],
        [
            Paragraph("<b>Exp B: Full Trajectory</b> (W1-W5)", table_cell),
            Paragraph("W1-W5, $\\Delta_{1-5}$, Mean/Std/Min/Missing", table_cell),
            Paragraph("1.0000", table_cell),
            Paragraph("<b>1.0000</b> (0 Missed)", table_cell),
            Paragraph("<b>1.0000</b> (Optimal)", table_cell),
            Paragraph("<b>1.0000</b>", table_cell)
        ]
    ]
    t_exp = Table(exp_table_data, colWidths=[130, 146, 65, 65, 65, 65])
    t_exp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('BOX', (0,0), (-1,-1), 0.5, c_secondary),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_exp)
    story.append(Spacer(1, 6))
    
    # Embedded plots
    docs_dir = os.path.dirname(filename)
    cm_img = os.path.join(docs_dir, 'confusion_matrix.png')
    latent_img = os.path.join(docs_dir, 'latent_space_distribution.png')
    
    if os.path.exists(cm_img) and os.path.exists(latent_img):
        img_table = [
            [Image(cm_img, width=255, height=165), Image(latent_img, width=270, height=165)],
            [Paragraph("<b>Figure 1:</b> Confusion Matrix (0 False Negatives)", table_cell), Paragraph("<b>Figure 2:</b> Latent Space PCA Projection (Clear Cluster Separation)", table_cell)]
        ]
        t_imgs = Table(img_table, colWidths=[268, 268])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        story.append(t_imgs)
        
    story.append(PageBreak())
    
    # -------------------------------------------------------------------------
    # 5. PyTorch Deep Sequence LSTM Model (Sprint 3)
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. PyTorch Deep Sequence LSTM Model (Sprint 3)", h1_style))
    story.append(Paragraph(
        "<b>Rigorous Formulation: No Target Leakage:</b> The LSTM sequence predictor was trained and evaluated <i>strictly on non-null project submissions</i> "
        "($N=1,639$ learners), completely removing any synthetic label imputation. The model maps the temporal sequence of normalized weekly quizzes "
        "$X = [q_1, q_2, q_3, q_4, q_5]^T \\in \\mathbb{R}^{5 \\times 1}$ to forecast final capstone performance $y \\in [0, 100]$.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Mathematical Formulation: Forget Gate, Cell State & Hidden State:</b><br/>"
        "$$\\begin{aligned}"
        "f_t &= \\sigma(W_f [h_{t-1}, x_t] + b_f) \\quad &\\text{-- Forget Gate: adaptively dampens early-week onboarding noise} \\\\"
        "i_t &= \\sigma(W_i [h_{t-1}, x_t] + b_i), \\quad \\tilde{C}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c) \\quad &\\text{-- Input Gate & Candidate Memory} \\\\"
        "C_t &= f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t \\quad &\\text{-- Cell State: Constant Error Carousel (CEC) highway} \\\\"
        "o_t &= \\sigma(W_o [h_{t-1}, x_t] + b_o), \\quad h_t = o_t \\odot \\tanh(C_t) \\quad &\\text{-- Output Gate & Hidden State projection}"
        "\\end{aligned}$$",
        callout_style
    ))
    story.append(Paragraph(
        "<b>Combating the Vanishing Gradient Problem:</b> In a standard Vanilla RNN, the backpropagated gradient scales by $\\prod \\frac{\\partial h_t}{\\partial h_{t-1}}$, "
        "which decays exponentially to zero over 5 timesteps. In our LSTM, the additive cell state recurrence $\\frac{\\partial C_t}{\\partial C_{t-1}} = f_t$ "
        "creates a linear highway where gradients flow freely without decay. Concurrently, when a learner struggles in Week 1 due to environment setup issues, "
        "the forget gate $f_t$ attenuates this early anomaly while tracking true learning velocity.",
        body_style
    ))
    
    # LSTM Metrics
    lstm_metrics_data = [
        [Paragraph("Metric", table_header), Paragraph("Validation Result", table_header), Paragraph("Independent Test Result", table_header), Paragraph("Evaluation Context", table_header)],
        [Paragraph("Root Mean Squared Error (RMSE)", table_cell), Paragraph("7.73 pts", table_cell), Paragraph("<b>7.85 pts</b>", table_cell), Paragraph("Evaluated on true project scores", table_cell)],
        [Paragraph("Mean Absolute Error (MAE)", table_cell), Paragraph("5.62 pts", table_cell), Paragraph("<b>5.67 pts</b>", table_cell), Paragraph("Sub-6 point average prediction error", table_cell)],
        [Paragraph("Coefficient of Determination ($R^2$)", table_cell), Paragraph("0.8160", table_cell), Paragraph("<b>0.8124</b>", table_cell), Paragraph("Explains >81% of capstone variance", table_cell)]
    ]
    t_lstm = Table(lstm_metrics_data, colWidths=[160, 110, 110, 156])
    t_lstm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('BOX', (0,0), (-1,-1), 0.5, c_secondary),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_lstm)
    story.append(Spacer(1, 6))
    
    lstm_img = os.path.join(docs_dir, 'lstm_sequence_predictions.png')
    if os.path.exists(lstm_img):
        story.append(Image(lstm_img, width=520, height=160))
        story.append(Paragraph("<b>Figure 3:</b> PyTorch LSTM 5-Week Trajectory Forecasting vs Actual Capstone Project Scores ($R^2 = 0.8124$)", table_cell))
        story.append(Spacer(1, 6))
        
    # -------------------------------------------------------------------------
    # 6. Sprint 4 NLP: spaCy NER, TextBlob & Vector RAG
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. NLP Pipeline: spaCy NER, TextBlob, and Gensim Word2Vec (Sprint 4)", h1_style))
    story.append(Paragraph(
        "<b>spaCy Custom Entity Recognition Benchmark:</b> An annotated benchmark dataset of 50 technical query instances was constructed to evaluate NER. "
        "The custom <code>EntityRuler</code> pipeline achieved <b>Module Precision: 0.8667, Recall: 0.7647, F1: 0.8125</b> and <b>Date Precision: 1.0000, Recall: 0.8333, F1: 0.9091</b>. "
        "As expected, exact string matching failed on misspelled terms ('PyTorxh', 'lstmm'), highlighting the architectural boundary of token-level NER.<br/>"
        "<b>TextBlob Dialect Nuances:</b> In Egyptian bilingual communication, learners mix Franco-Arab or informal slang with English. TextBlob's standard English lexicon "
        "may evaluate mixed queries as neutral (0.0). To guarantee student safety, the Deliberative Agent pairs TextBlob polarity with a domain distress keyword detector.<br/>"
        "<b>True Semantic Vector RAG:</b> Rather than naive keyword matching, student queries are vectorized into 100-dimensional dense space using trained Word2Vec embeddings "
        "and matched against curriculum documentation chunks using <b>Cosine Similarity</b> (e.g. Cosine Sim = 0.8324 for LSTM autograd queries).",
        body_style
    ))
    
    # FastText vs Word2Vec Table
    vec_data = [
        [Paragraph("Dimension", table_header), Paragraph("Gensim Word2Vec (Deployed)", table_header), Paragraph("FastText (Subword n-grams)", table_header)],
        [
            Paragraph("OOV Handling", table_cell),
            Paragraph("Fails with KeyError on typos ('pytorxh') or unseen tokens.", table_cell),
            Paragraph("Interpolates embeddings via character n-grams; highly resilient.", table_cell)
        ],
        [
            Paragraph("Model Size", table_cell),
            Paragraph("Ultra-compact (1.2 MB on Sprints corpus); minimal RAM overhead.", table_cell),
            Paragraph("Large memory footprint (~40x larger) due to subword buckets.", table_cell)
        ],
        [
            Paragraph("Inference Latency", table_cell),
            Paragraph("Instantaneous O(1) hash lookup (~0.05 ms); ideal for n8n sub-second SLA.", table_cell),
            Paragraph("Higher per-token aggregation overhead (~2-5 ms per word).", table_cell)
        ]
    ]
    t_vec = Table(vec_data, colWidths=[110, 213, 213])
    t_vec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('BOX', (0,0), (-1,-1), 0.5, c_secondary),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_vec)
    story.append(Spacer(1, 6))
    
    # -------------------------------------------------------------------------
    # 7. n8n Event-Driven Deliberative Agent Workflow & SLA
    # -------------------------------------------------------------------------
    story.append(Paragraph("7. Event-Driven n8n Deliberative Agent & Latency SLA Verification", h1_style))
    story.append(Paragraph(
        "The production n8n workflow operates as an event-driven orchestrator triggered via HTTP Webhook. "
        "Distressed queries (polarity &lt; -0.15 or distress keywords) are escalated directly and persisted into SQLite table <code>support_tickets</code> "
        "via <code>POST /api/v1/tickets/escalate</code>, ensuring a permanent audit trail for senior mentors. Neutral and positive inquiries are routed "
        "to the Semantic Vector RAG engine to return grounded curriculum answers.<br/>"
        "<b>Latency Benchmarking:</b> End-to-end workflow execution latency averages <b>154.5 ms</b> (max: <b>296.3 ms</b>), "
        "strictly fulfilling the client's <b>&lt; 3.5-second SLA</b> by an order of magnitude.",
        body_style
    ))
    story.append(Spacer(1, 4))
    
    # -------------------------------------------------------------------------
    # 8. Business Impact & Recommendations
    # -------------------------------------------------------------------------
    story.append(Paragraph("8. Business Recommendations & Next Steps", h1_style))
    story.append(Paragraph(
        "&bull; <b>Hybrid Inference Strategy:</b> Use Word2Vec inside the real-time n8n loop for sub-second responses, while running FastText in asynchronous forum ingestion.<br/>"
        "&bull; <b>Proactive Early Alerting:</b> Run Experiment A (W1-W3) at Week 3 to capture 98.5% of vulnerable students before dropouts occur.<br/>"
        "&bull; <b>Persistent Ticketing Audit:</b> Integrate SQLite escalation logs with mentor dashboards to track mean time to resolution (MTTR).",
        body_style
    ))
    
    doc.build(story)
    print(f"Technical Evaluation Report PDF built successfully: {filename}")

if __name__ == '__main__':
    p1 = r'C:\Users\mw851\sprints-learner-success-agent\docs\Sprints_Model_Evaluation_Report.pdf'
    p2 = r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone\docs\Sprints_Model_Evaluation_Report.pdf'
    build_pdf(p1)
    build_pdf(p2)
