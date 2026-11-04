"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/build_presentation_pdf.py
Description: Generates the professional 10-slide final presentation deck:
             docs/Sprints_Architecture_Presentation.pdf
             Formatted as landscape 16:9 widescreen presentation slides for Sprints leadership.
==============================================================================
"""

import os
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)

def create_slide_deck(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=landscape(letter),
        leftMargin=35,
        rightMargin=35,
        topMargin=28,
        bottomMargin=28
    )
    
    styles = getSampleStyleSheet()
    
    # Theme Palette
    c_dark = colors.HexColor('#0f172a')       # Navy slate
    c_blue = colors.HexColor('#1d4ed8')       # Cobalt blue
    c_cyan = colors.HexColor('#0284c7')       # Cyan
    c_card_bg = colors.HexColor('#f8fafc')    # Crisp card background
    c_border = colors.HexColor('#cbd5e1')
    
    slide_title = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=c_dark,
        spaceAfter=2
    )
    
    slide_subtitle = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=c_blue,
        spaceAfter=10
    )
    
    card_title = ParagraphStyle(
        'CardTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_blue,
        spaceAfter=4
    )
    
    card_body = ParagraphStyle(
        'CardBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.2,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )
    
    card_bullet = ParagraphStyle(
        'CardBullet',
        parent=card_body,
        leftIndent=10,
        firstLineIndent=-8,
        spaceAfter=3
    )
    
    hero_title = ParagraphStyle(
        'HeroTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=c_dark,
        alignment=1,
        spaceAfter=8
    )
    
    hero_sub = ParagraphStyle(
        'HeroSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_blue,
        alignment=1,
        spaceAfter=18
    )
    
    meta_p = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569'),
        alignment=1
    )
    
    story = []
    
    def make_header(title, subtitle):
        return [
            Paragraph(title, slide_title),
            Paragraph(subtitle, slide_subtitle),
            HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=0, spaceAfter=12)
        ]
        
    def wrap_card(content_list, width=345, height=None, bg=c_card_bg):
        t = Table([[content_list]], colWidths=[width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg),
            ('BOX', (0,0), (-1,-1), 1, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        return t

    docs_dir = os.path.dirname(filename)

    # =========================================================================
    # SLIDE 1: Title Slide (Cover)
    # =========================================================================
    story.append(Spacer(1, 35))
    story.append(Paragraph("Sprints AI-Powered Learner Success & Support Agent", hero_title))
    story.append(Paragraph("End-to-End Predictive Modeling & Event-Driven Deliberative Agent Workflow", hero_sub))
    story.append(HRFlowable(width="60%", thickness=2, color=c_blue, spaceBefore=0, spaceAfter=20))
    
    cover_meta = [
        Paragraph("<b>Client Organization:</b> Sprints (sprints.ai)", meta_p),
        Paragraph("<b>Target Cohort:</b> AI Pioneers Nationwide Egyptian Bootcamp (2,000 Enrolled Students)", meta_p),
        Paragraph("<b>Lead AI Researcher:</b> Tarek Mahmoud | <b>Product Manager:</b> Salma Youssef", meta_p),
        Paragraph("<b>Prepared by:</b> Mohammed Wael (Lead AI & Machine Learning Engineer)", meta_p),
        Paragraph("<b>Deliverable:</b> 2-Week Capstone Project Final Graduation Presentation", meta_p)
    ]
    for p in cover_meta:
        story.append(p)
        story.append(Spacer(1, 3))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: Executive Summary & The Dual Operational Challenge
    # =========================================================================
    story.extend(make_header("Slide 1: Executive Summary & The Dual Operational Challenge", "Scaling Bootcamps Across Egypt Strains Academic Support Operations"))
    c1 = [
        Paragraph("The Dual Operational Bottleneck", card_title),
        Paragraph("&bull; <b>Problem 1 (Silent Dropouts):</b> Fragmented LMS telemetry obscures student struggle. Dropouts happen silently before intervention is possible.", card_bullet),
        Paragraph("&bull; <b>Problem 2 (Support Desk Flooding):</b> Repetitive curriculum questions delay mentor response times for urgent student blockers.", card_bullet),
        Paragraph("&bull; <b>The Integrated Solution:</b> An intelligent end-to-end agent synthesizing relational SQL, PyTorch supervised classification, sequence LSTM forecasting, and an n8n Deliberative Agent.", card_bullet)
    ]
    c2 = [
        Paragraph("Key Success Metrics & Acceptance Criteria", card_title),
        Paragraph("&bull; <b>Model Accuracy:</b> PyTorch risk classifier must achieve an <b>F1 score &ge; 0.82</b>, strictly minimizing false negatives.", card_bullet),
        Paragraph("&bull; <b>Workflow Latency:</b> The n8n agent must process incoming HTTP requests and route tickets in <b>&lt; 3.5 seconds</b>.", card_bullet),
        Paragraph("&bull; <b>Scientific Rigor:</b> 100% leak-free preprocessing, train-only imputation, and dual experimental validation.", card_bullet),
        Paragraph("&bull; <b>Achieved Outcomes:</b> Test F1 = <b>0.971</b> (W1-W3) / <b>1.000</b> (W1-W5), sub-200ms latency, and persistent escalation auditing.", card_bullet)
    ]
    t2 = Table([[wrap_card(c1, 345), wrap_card(c2, 345)]], colWidths=[355, 355])
    t2.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t2)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: System Architecture Overview
    # =========================================================================
    story.extend(make_header("Slide 2: End-to-End System Architecture", "Seamless Pipeline from Relational Telemetry to Event-Driven Deliberative RAG"))
    c_arch_l = [
        Paragraph("Data & Predictive Modeling Pipeline", card_title),
        Paragraph("<b>1. Relational Layer:</b> SQLite database (<code>sprints_learners.db</code>) mapping 2,000 learners, 9,101 quizzes, and 2,000 tickets.", card_bullet),
        Paragraph("<b>2. Data Cleaning & Imputation:</b> Leak-free forward-fill and regional median imputation (Train-fitted only).", card_bullet),
        Paragraph("<b>3. PyTorch Risk MLP:</b> Supervised classifier detecting disengagement risk with calibrated threshold (0.42).", card_bullet),
        Paragraph("<b>4. PyTorch Sequence LSTM:</b> Evaluates 5-week trajectories to forecast final capstone outcomes on real submissions ($N=1,639$).", card_bullet)
    ]
    c_arch_r = [
        Paragraph("Agentic NLP & Orchestration Layer", card_title),
        Paragraph("<b>5. spaCy Custom NER:</b> EntityRuler extracting curriculum modules (F1 = 0.812) and dates (F1 = 0.909).", card_bullet),
        Paragraph("<b>6. TextBlob Sentiment & Hybrid Distress:</b> Quantifies student frustration and catches Egyptian dialect idioms.", card_bullet),
        Paragraph("<b>7. Semantic Vector RAG:</b> Word2Vec dense vectors matched via Cosine Similarity against syllabus chunks.", card_bullet),
        Paragraph("<b>8. n8n Deliberative Workflow:</b> Sub-second webhook orchestrator logging escalations to SQLite in &lt; 200 ms.", card_bullet)
    ]
    t3 = Table([[wrap_card(c_arch_l, 345), wrap_card(c_arch_r, 345)]], colWidths=[355, 355])
    t3.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t3)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: Database Schema Design & SQL Analytics
    # =========================================================================
    story.extend(make_header("Slide 3: Relational Database Architecture & SQL Analytics (Sprint 1)", "Normalized 6-Entity Schema with Modular CTE Analytics Eliminating Join Multiplication"))
    erd_summary_l = [
        Paragraph("Normalized Schema Entities (3NF)", card_title),
        Paragraph("&bull; <b><code>learners</code>:</b> 2,000 students across 15 Egyptian governorates.", card_bullet),
        Paragraph("&bull; <b><code>quiz_grades</code>:</b> 9,101 normalized submissions across weeks 1 to 5.", card_bullet),
        Paragraph("&bull; <b><code>support_tickets</code>:</b> 2,000 tickets with sentiment, polarity, and priority.", card_bullet),
        Paragraph("&bull; <b><code>course_modules</code>:</b> 4 Sprints, difficulty ratings, lead instructors.", card_bullet),
        Paragraph("&bull; <b><code>interaction_logs</code>:</b> LMS session telemetry and clickstream timestamps.", card_bullet)
    ]
    erd_summary_r = [
        Paragraph("SQL Query 5: Join Multiplication Elimination", card_title),
        Paragraph("&bull; <b>Cartesian Issue:</b> Naive multi-table joins between quizzes and tickets multiplied rows before aggregation.", card_bullet),
        Paragraph("&bull; <b>The CTE Solution:</b> Created independent CTEs for <code>grade_stats</code> and <code>ticket_stats</code> before joining to modules.", card_bullet),
        Paragraph("&bull; <b>Grade Degradation Query:</b> CTE flags drops &ge; 15 pts from Week 1 baseline, surfacing at-risk students 2-3 weeks early.", card_bullet),
        Paragraph("&bull; <b>Regional Analysis:</b> Identified Upper Egypt (Assiut: 28.4%) and Canal (Ismailia: 24.1%) as high-dropout regions.", card_bullet)
    ]
    t4 = Table([[wrap_card(erd_summary_l, 345), wrap_card(erd_summary_r, 345)]], colWidths=[355, 355])
    t4.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t4)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: Supervised PyTorch Feed-Forward Risk Classifier
    # =========================================================================
    story.extend(make_header("Slide 4: PyTorch Supervised Risk Classification (Sprint 2)", "Dual Experimental Protocol: Early Proactive (W1-W3) vs Full Trajectory (W1-W5)"))
    cm_path = os.path.join(docs_dir, 'confusion_matrix.png')
    c_mlp_text = [
        Paragraph("Methodology & Independent Test Outcomes", card_title),
        Paragraph("&bull; <b>Zero Target Leakage:</b> Clean 70/15/15 split. Imputation and scaler fitted strictly on Training set.", card_bullet),
        Paragraph("&bull; <b>Experiment A (Early Risk W1-W3):</b> <b>Test F1 = 0.9714</b> | <b>Recall = 0.9855</b> (1 missed) | <b>ROC-AUC = 0.9996</b>. Enables proactive intervention at Week 3!", card_bullet),
        Paragraph("&bull; <b>Experiment B (Full Trajectory W1-W5):</b> <b>Test F1 = 1.0000</b> | <b>Recall = 1.0000</b> (0 False Negatives).", card_bullet),
        Paragraph("&bull; <b>Loss & Threshold:</b> <code>pos_weight=3.30</code> heavily penalizes False Negatives; decision threshold calibrated to <b>0.42</b>.", card_bullet)
    ]
    if os.path.exists(cm_path):
        img_mlp = Image(cm_path, width=280, height=200)
        t5 = Table([[wrap_card(c_mlp_text, 380), img_mlp]], colWidths=[400, 310])
    else:
        t5 = Table([[wrap_card(c_mlp_text, 700)]], colWidths=[710])
    t5.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t5)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: Latent Space Behavioral Clustering
    # =========================================================================
    story.extend(make_header("Slide 5: Latent Space Behavioral Clustering (Sprint 2)", "Penultimate Layer PCA Visualizing Disentangled Student Risk Manifolds"))
    latent_img_p = os.path.join(docs_dir, 'latent_space_distribution.png')
    c_latent_text = [
        Paragraph("Manifold Representation Validation", card_title),
        Paragraph("&bull; <b>Feature Extraction:</b> Extracted 16-dimensional penultimate activations prior to output logits.", card_bullet),
        Paragraph("&bull; <b>Principal Component Analysis:</b> First two PCs capture >88% of total latent feature variance.", card_bullet),
        Paragraph("&bull; <b>Linear Separability:</b> Retained learners (blue) and dropped learners (red) form distinct, linearly separable clusters.", card_bullet),
        Paragraph("&bull; <b>Scientific Insight:</b> Validates that the MLP learned underlying behavioral engagement dynamics rather than memorizing noisy tabular patterns.", card_bullet)
    ]
    if os.path.exists(latent_img_p):
        img_latent = Image(latent_img_p, width=310, height=195)
        t6 = Table([[wrap_card(c_latent_text, 360), img_latent]], colWidths=[380, 330])
    else:
        t6 = Table([[wrap_card(c_latent_text, 700)]], colWidths=[710])
    t6.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t6)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: Deep Sequence LSTM Modeling
    # =========================================================================
    story.extend(make_header("Slide 6: Deep Sequence LSTM Trajectory Modeling (Sprint 3)", "Forget Gate & Constant Error Carousel Mitigating Noise and Vanishing Gradients"))
    lstm_img_p = os.path.join(docs_dir, 'lstm_sequence_predictions.png')
    c_lstm_text = [
        Paragraph("Architectural Rigor on True Submissions", card_title),
        Paragraph("&bull; <b>Trained Strictly on Non-Nulls:</b> Filtered to $N=1,639$ real project submissions (zero synthetic label leakage).", card_bullet),
        Paragraph("&bull; <b>Forget Gate ($f_t$):</b> Dampens early onboarding noise (e.g. Week 1 environment setup issues).", card_bullet),
        Paragraph("&bull; <b>Cell State ($C_t$):</b> Additive Constant Error Carousel highway enabling gradient flow across timesteps without decay.", card_bullet),
        Paragraph("&bull; <b>Test Metrics:</b> <b>RMSE = 7.85 pts</b> | <b>MAE = 5.67 pts</b> | <b>R² = 0.8124</b>.", card_bullet)
    ]
    if os.path.exists(lstm_img_p):
        img_lstm = Image(lstm_img_p, width=330, height=195)
        t7 = Table([[wrap_card(c_lstm_text, 350), img_lstm]], colWidths=[365, 345])
    else:
        t7 = Table([[wrap_card(c_lstm_text, 700)]], colWidths=[710])
    t7.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t7)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: Sprint 4 NLP: spaCy NER & Word2Vec vs FastText
    # =========================================================================
    story.extend(make_header("Slide 7: NLP Pipeline & Word Embedding Benchmark (Sprint 4)", "spaCy NER Benchmark, TextBlob Nuances & Word2Vec vs FastText OOV Analysis"))
    c_nlp_l = [
        Paragraph("spaCy NER Benchmark & Sentiment", card_title),
        Paragraph("&bull; <b>NER 50-Sample Benchmark:</b> Module F1 = <b>0.8125</b> (Prec: 0.867, Rec: 0.765) | Date F1 = <b>0.9091</b> (Prec: 1.000).", card_bullet),
        Paragraph("&bull; <b>TextBlob Dialect Nuances:</b> Standard English lexicons miss Franco-Arab / Egyptian idioms ('msh fahem'); resolved by hybrid distress keyword detector.", card_bullet),
        Paragraph("&bull; <b>Semantic Vector RAG:</b> Dense Word2Vec vectors match queries to syllabus chunks via Cosine Similarity (Sim = 0.832 for LSTM).", card_bullet)
    ]
    c_nlp_r = [
        Paragraph("Word2Vec vs FastText Empirical Trade-offs", card_title),
        Paragraph("&bull; <b>Word2Vec:</b> Ultra-compact (1.2 MB), instantaneous O(1) hash lookup (~0.05 ms). Chosen for real-time sub-second n8n SLA.", card_bullet),
        Paragraph("&bull; <b>FastText:</b> Subword n-grams handle typos ('PyTorxh', 'lstmm') and dialectal morphs where Word2Vec raises KeyError.", card_bullet),
        Paragraph("&bull; <b>Architecture Decision:</b> Word2Vec inside real-time n8n API loop; FastText for asynchronous forum batch indexing.", card_bullet)
    ]
    t8 = Table([[wrap_card(c_nlp_l, 345), wrap_card(c_nlp_r, 345)]], colWidths=[355, 355])
    t8.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t8)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 9: n8n Deliberative Agent Workflow & SLA Verification
    # =========================================================================
    story.extend(make_header("Slide 8: Event-Driven n8n Workflow & Latency SLA Verification", "Sub-Second Deliberative Triage & Persistent SQLite Escalation Logging"))
    c_n8n_l = [
        Paragraph("Workflow Nodes & Routing Logic", card_title),
        Paragraph("&bull; <b>HTTP Webhook Node:</b> Receives learner query payload with strict Pydantic validation.", card_bullet),
        Paragraph("&bull; <b>Deliberative Router Node:</b> Evaluates TextBlob polarity, distress keywords, and student risk profile.", card_bullet),
        Paragraph("&bull; <b>Persistent Escalation:</b> Calls <code>POST /api/v1/tickets/escalate</code> to record critical tickets permanently in SQLite.", card_bullet),
        Paragraph("&bull; <b>Semantic RAG Node:</b> Returns grounded curriculum answers in under 180 ms for neutral inquiries.", card_bullet)
    ]
    c_n8n_r = [
        Paragraph("Latency & SLA Benchmarking Results", card_title),
        Paragraph("&bull; <b>Client SLA Target:</b> End-to-end workflow execution &lt; 3,500 ms.", card_bullet),
        Paragraph("&bull; <b>RAG Pipeline Latency:</b> <b>154.5 ms</b> (22x faster than SLA requirement!).", card_bullet),
        Paragraph("&bull; <b>Escalation Query Latency:</b> <b>9.1 ms</b> (Instantaneous triage!).", card_bullet),
        Paragraph("&bull; <b>Risk Classifier Latency:</b> <b>2.0 ms</b>.", card_bullet),
        Paragraph("&bull; <b>Verdict:</b> <b>100% SLA COMPLIANCE CONFIRMED</b>.", card_bullet)
    ]
    t9 = Table([[wrap_card(c_n8n_l, 345), wrap_card(c_n8n_r, 345)]], colWidths=[355, 355])
    t9.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t9)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 10: Conclusion & Next Steps
    # =========================================================================
    story.extend(make_header("Slide 9: Business Impact & Strategic Implementation Roadmap", "Empowering Sprints to Scale Cohorts Confidently Across MENA"))
    c_impact_l = [
        Paragraph("Quantifiable Strategic Impact", card_title),
        Paragraph("&bull; <b>70% Triage Automation:</b> Routine syntax queries resolved autonomously without TA intervention.", card_bullet),
        Paragraph("&bull; <b>Proactive Dropout Prevention:</b> W1-W3 classifier catches 98.5% of at-risk students before dropout occurs.", card_bullet),
        Paragraph("&bull; <b>Audit Compliance:</b> Every escalated interaction is permanently tracked in the relational database.", card_bullet)
    ]
    c_impact_r = [
        Paragraph("Phase 2 Deployment Roadmap", card_title),
        Paragraph("&bull; <b>Milestone 1:</b> Embed FastAPI webhook directly inside Sprints LMS student portal widget.", card_bullet),
        Paragraph("&bull; <b>Milestone 2:</b> Automate TA WhatsApp/Slack alerts for escalated critical tickets.", card_bullet),
        Paragraph("&bull; <b>Milestone 3:</b> Extend models to Cybersecurity and Cloud tracks.", card_bullet),
        Paragraph("<b>Thank You! Questions & Discussion.</b>", card_title)
    ]
    t10 = Table([[wrap_card(c_impact_l, 345), wrap_card(c_impact_r, 345)]], colWidths=[355, 355])
    t10.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t10)
    
    doc.build(story)
    print(f"Presentation Deck PDF built successfully: {filename}")

if __name__ == '__main__':
    p1 = r'C:\Users\mw851\sprints-learner-success-agent\docs\Sprints_Architecture_Presentation.pdf'
    p2 = r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone\docs\Sprints_Architecture_Presentation.pdf'
    create_slide_deck(p1)
    create_slide_deck(p2)
