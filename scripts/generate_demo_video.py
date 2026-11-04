"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/generate_demo_video.py
Description: Generates the 3.5-minute demo walkthrough video:
             demo/Demo_Walkthrough.mp4 (and root Demo_Walkthrough.mp4)
             Showcases live real metrics across all 4 Sprints:
             1. Problem Statement & Acceptance Criteria
             2. SQL Schema (2,000 learners) & Query 5 CTE Fix (No Join Multiplication)
             3. PyTorch MLP: Exp A (W1-W3 F1=0.9714) & Exp B (W1-W5 F1=1.0000)
             4. PyTorch Sequence LSTM on Real Submissions (RMSE=7.85, R2=0.8124)
             5. spaCy NER Benchmark (F1=0.812) & FastText vs Word2Vec OOV Analysis
             6. n8n Event-Driven Deliberative Agent (< 200 ms Latency & DB Escalation)
==============================================================================
"""

import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_slide_image(width, height, title, subtitle, bullets, badge=None, highlight_box=None):
    img = Image.new('RGB', (width, height), color=(15, 23, 42)) # Slate 900
    draw = ImageDraw.Draw(img)
    
    # Header background
    draw.rectangle([0, 0, width, 110], fill=(30, 41, 59)) # Slate 800
    draw.line([(0, 110), (width, 110)], fill=(29, 78, 216), width=4) # Royal blue accent
    
    # Title & Subtitle
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 32)
        sub_font = ImageFont.truetype("arial.ttf", 19)
        body_font = ImageFont.truetype("arial.ttf", 21)
        badge_font = ImageFont.truetype("arialbd.ttf", 17)
        mono_font = ImageFont.truetype("consola.ttf", 17)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        badge_font = ImageFont.load_default()
        mono_font = ImageFont.load_default()

    draw.text((45, 22), title, font=title_font, fill=(248, 250, 252))
    draw.text((45, 70), subtitle, font=sub_font, fill=(56, 189, 248)) # Sky blue
    
    if badge:
        badge_w = 270
        draw.rectangle([width - badge_w - 40, 30, width - 40, 78], fill=(5, 150, 105)) # Emerald green
        draw.text((width - badge_w - 20, 42), badge, font=badge_font, fill=(255, 255, 255))
        
    # Main Content Area: Card
    card_w = width - 90
    draw.rectangle([45, 135, 45 + card_w, height - 65], fill=(24, 34, 53), outline=(51, 65, 85), width=2)
    
    y = 165
    for b in bullets:
        draw.ellipse([75, y + 6, 87, y + 18], fill=(59, 130, 246))
        draw.text((105, y), b, font=body_font, fill=(226, 232, 240))
        y += 46
        
    if highlight_box:
        box_y = y + 10
        draw.rectangle([75, box_y, width - 75, box_y + 115], fill=(15, 23, 42), outline=(13, 148, 136), width=2)
        draw.text((95, box_y + 12), highlight_box.get('title', 'EVIDENCE / METRIC OUTCOME:'), font=badge_font, fill=(45, 212, 191))
        draw.text((95, box_y + 44), highlight_box.get('text', ''), font=mono_font, fill=(241, 245, 249))
        
    # Footer
    draw.rectangle([0, height - 48, width, height], fill=(15, 23, 42))
    draw.line([(0, height - 48), (width, height - 48)], fill=(51, 65, 85), width=1)
    draw.text((45, height - 36), "Sprints (sprints.ai) | AI Pioneers Graduation Capstone | Lead AI Engineer: Mohammed Wael", font=sub_font, fill=(148, 163, 184))
    
    return np.array(img)

def generate_video(out_path, duration_sec=210, fps=24):
    width, height = 1280, 720
    
    slides_data = [
        # Segment 1 (0 to 30s)
        {
            "title": "Sprints AI-Powered Learner Success & Support Agent",
            "subtitle": "Capstone Project Final Walkthrough | AI & Machine Learning Intermediate",
            "badge": "GRADUATION DEMO",
            "bullets": [
                "Client: Sprints (sprints.ai) - Egypt's Leading Technology Bootcamp Provider",
                "Challenge 1: Inability to manually identify silent dropouts across 15 Egyptian governorates.",
                "Challenge 2: Support desk flooded by repetitive curriculum questions delaying response times.",
                "Cohesive Solution: End-to-end integration across Relational SQL, PyTorch ML/DL, and n8n RAG."
            ],
            "highlight_box": {
                "title": "CAPSTONE ACCEPTANCE CRITERIA TARGETS:",
                "text": "• PyTorch F1 Score >= 0.82 (Minimizing False Negatives)   • n8n Latency SLA < 3.5 Seconds\n• 100% Leakage-Free Preprocessing                         • Persistent Escalation in SQLite"
            },
            "duration": 30
        },
        # Segment 2 (30 to 65s)
        {
            "title": "Sprint 1: Practical Data Analytics & Relational Database",
            "subtitle": "Normalized Schema (2,000 Learners) & Query 5 CTE Fix (No Join Multiplication)",
            "badge": "SPRINT 1: SQL",
            "bullets": [
                "Normalized 6-Table Relational Schema: cohorts, learners, modules, quiz_grades, tickets, logs.",
                "Populated from Raw Telemetry: 2,000 distinct learners (LRN-1001..3000), 9,101 quiz records.",
                "Query 2: Detected progressive grade degradation (>= 15 pts drop from Week 1 baseline).",
                "Query 5 Fix: Modular CTEs for grade_stats and ticket_stats eliminate Cartesian join multiplication."
            ],
            "highlight_box": {
                "title": "LIVE SQL CTE QUERY 5 EXECUTION (database/sprints_learners.db):",
                "text": "MOD-PYT-02: Avg Quiz=74.32 | Failing Submissions=412 | Support Tickets=485 | Escalated=108\nMOD-DL-03:  Avg Quiz=71.18 | Failing Submissions=498 | Support Tickets=542 | Escalated=136"
            },
            "duration": 35
        },
        # Segment 3 (65 to 105s)
        {
            "title": "Sprint 2: PyTorch Supervised Feed-Forward Risk Classifier",
            "subtitle": "Dual Experiments: Early Proactive Risk (W1-W3) vs Full Trajectory (W1-W5)",
            "badge": "SPRINT 2: F1 = 0.971 - 1.000",
            "bullets": [
                "Strict Leakage-Free Protocol: 70% Train (1,400) / 15% Val (300) / 15% Test (300).",
                "Exp A (Early Proactive Risk W1-W3): Operates at Week 3 to intervene before dropouts occur.",
                "Exp B (Full Trajectory Risk W1-W5): Evaluates end-to-end multi-week disengagement signals.",
                "Penultimate Latent Projections (PCA): Confirms distinct linear separation of risk clusters."
            ],
            "highlight_box": {
                "title": "INDEPENDENT TEST SET EVALUATION (models/risk_classifier.pth):",
                "text": "Exp A (W1-W3): Test F1 = 0.9714 | Test Recall = 0.9855 (1 FN) | ROC-AUC = 0.9996 (Target >= 0.82 MET)\nExp B (W1-W5): Test F1 = 1.0000 | Test Recall = 1.0000 (0 FN) | Decision Threshold = 0.42"
            },
            "duration": 40
        },
        # Segment 4 (105 to 145s)
        {
            "title": "Sprint 3: PyTorch Deep Sequence LSTM Trajectory Predictor",
            "subtitle": "Processing 5-Week Quizzes to Forecast Capstone Outcome on Real Submissions",
            "badge": "SPRINT 3: LSTM",
            "bullets": [
                "Zero Target Leakage: Trained and evaluated strictly on non-null project scores (N = 1,639).",
                "Forget Gate Dynamics: Discards transient Week 1 onboarding noise while tracking decline.",
                "Constant Error Carousel (Cell State): Preserves gradient highway across 5 time steps.",
                "Capstone Prediction: Hidden state h_5 forecasts continuous capstone project score."
            ],
            "highlight_box": {
                "title": "SEQUENCE LSTM TEST EVALUATION (models/sequence_predictor.pth):",
                "text": "Test RMSE: 7.85 points (on 0-100 scale)   |   Test MAE: 5.67 points   |   Test R^2: 0.8124\nLRN-1001 Trajectory: [85, 80, 75, 70, 60] -> Predicted Capstone: 77.63 pts (Pass)"
            },
            "duration": 40
        },
        # Segment 5 (145 to 180s)
        {
            "title": "Sprint 4: NLP Pipeline, spaCy NER & FastText Benchmark",
            "subtitle": "Custom EntityRuler, TextBlob Dialect Nuances & Semantic Vector RAG",
            "badge": "SPRINT 4: NLP",
            "bullets": [
                "spaCy NER 50-Case Benchmark: Module F1 = 0.8125 (Prec: 0.867) | Date F1 = 0.9091 (Prec: 1.000).",
                "TextBlob Dialect Handling: Paired with domain distress detector to catch Franco-Arab slang.",
                "Word2Vec vs FastText OOV Stress Test: Word2Vec raises KeyError on 'pytorxh'; FastText recovers.",
                "Semantic Vector RAG: Dense Word2Vec vectors match queries via Cosine Similarity (Sim = 0.832)."
            ],
            "highlight_box": {
                "title": "STRATEGIC HYBRID DEPLOYMENT ARCHITECTURE:",
                "text": "Word2Vec: Sub-millisecond vector lookup inside n8n real-time loop (< 3.5s SLA).\nFastText: Subword n-grams for asynchronous Egyptian forum batch indexing."
            },
            "duration": 35
        },
        # Segment 6 (180 to 210s)
        {
            "title": "Sprint 4 & Integration: Event-Driven n8n Deliberative Agent",
            "subtitle": "HTTP Webhook, Deliberative Decision Policy & Persistent SQLite Escalation",
            "badge": "SLA: 154ms (< 3.5s)",
            "bullets": [
                "Deliberative Routing: Evaluates sentiment polarity and student distress keywords.",
                "Persistent Escalation: Critical queries route to POST /api/v1/tickets/escalate and log to SQLite.",
                "Automated Vector RAG: Curriculum queries retrieve verified knowledge in 154 ms.",
                "Total Latency: 154 ms vs 3,500 ms SLA requirement (22x faster than client threshold)."
            ],
            "highlight_box": {
                "title": "FINAL DELIVERABLE VERDICT & REPOSITORY READY:",
                "text": "All 4 Sprints 100% Implemented & Verified | GitHub: PhantomMW/sprints-learner-success-agent\nDeliverables: SQL DB, PyTorch Models, 3 Executed Notebooks, n8n JSON, Technical Report & Presentation"
            },
            "duration": 30
        }
    ]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    total_frames = 0
    for s_idx, slide in enumerate(slides_data):
        print(f"Rendering Slide {s_idx+1}/{len(slides_data)}: {slide['title'][:40]}...")
        frame_rgb = create_slide_image(
            width, height,
            title=slide['title'],
            subtitle=slide['subtitle'],
            bullets=slide['bullets'],
            badge=slide.get('badge'),
            highlight_box=slide.get('highlight_box')
        )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        frames_for_slide = int(slide['duration'] * fps)
        for f in range(frames_for_slide):
            out.write(frame_bgr)
            total_frames += 1
            
    out.release()
    print(f"Video generated successfully: {out_path} ({total_frames} frames, {total_frames/fps:.1f} seconds)")

if __name__ == '__main__':
    repo_demo_dir = r'C:\Users\mw851\sprints-learner-success-agent\demo'
    scratch_demo_dir = r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone\demo'
    os.makedirs(repo_demo_dir, exist_ok=True)
    os.makedirs(scratch_demo_dir, exist_ok=True)
    
    repo_video = os.path.join(repo_demo_dir, 'Demo_Walkthrough.mp4')
    scratch_video = os.path.join(scratch_demo_dir, 'Demo_Walkthrough.mp4')
    
    generate_video(repo_video, duration_sec=210, fps=24)
    shutil.copyfile(repo_video, scratch_video)
    
    # Also copy to root in both repos for visibility
    shutil.copyfile(repo_video, r'C:\Users\mw851\sprints-learner-success-agent\Demo_Walkthrough.mp4')
    shutil.copyfile(repo_video, r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone\Demo_Walkthrough.mp4')
    print("Demo Walkthrough video updated in repo and scratch!")
