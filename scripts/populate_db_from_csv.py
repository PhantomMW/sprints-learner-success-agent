"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/populate_db_from_csv.py
Description: Ingests the 2,000-learner CSV dataset into the SQLite relational database
             (database/sprints_learners.db), populating learners, quiz_grades,
             support_tickets, and interaction_logs.
==============================================================================
"""

import os
import sqlite3
import pandas as pd
from textblob import TextBlob

def populate_database(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    schema_path = os.path.join(base_dir, 'database', 'schema.sql')
    seed_path = os.path.join(base_dir, 'database', 'seed.sql')
    csv_path = os.path.join(base_dir, 'data', 'sprints_learner_data.csv')
    db_path = os.path.join(base_dir, 'database', 'sprints_learners.db')
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 1. Execute schema DDL
    with open(schema_path, 'r', encoding='utf-8') as f:
        cur.executescript(f.read())
    print("Schema initialized.")
    
    # 2. Execute base seed (Cohorts & Modules)
    with open(seed_path, 'r', encoding='utf-8') as f:
        cur.executescript(f.read())
    print("Cohorts & Course Modules seeded.")
    
    # 3. Read 2,000 learners CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} rows.")
    
    module_mapping = {
        1: 'MOD-101', # SQL
        2: 'MOD-102', # pandas
        3: 'MOD-201', # ML
        4: 'MOD-301', # LSTM
        5: 'MOD-401'  # NLP
    }
    
    learners_records = []
    grades_records = []
    tickets_records = []
    logs_records = []
    
    for idx, row in df.iterrows():
        lid = str(row['learner_id'])
        city = str(row['cohort_city'])
        dropped = 1 if (row['dropped_out'] in [True, 'True', 1]) else 0
        status = 'Dropped' if dropped == 1 else 'Active'
        name = f"Learner {lid.replace('LRN-', '')}"
        email = f"{lid.lower()}@sprints.ai"
        cohort_id = 'COHORT-AI-01'
        
        learners_records.append((lid, cohort_id, name, email, city, status, dropped))
        
        # Quizzes 1 to 5
        for w in range(1, 6):
            val = row.get(f'week_1_quiz' if w == 1 else f'week_{w}_quiz')
            if pd.notna(val):
                score = float(val)
                mod_id = module_mapping[w]
                grades_records.append((lid, mod_id, w, score, 0))
                
        # Support ticket from forum feedback
        feedback = str(row.get('forum_feedback_text', '')).strip()
        if feedback and feedback != 'nan':
            blob = TextBlob(feedback)
            pol = round(blob.sentiment.polarity, 4)
            subj = round(blob.sentiment.subjectivity, 4)
            is_escalated = 1 if (pol < -0.15 or dropped == 1) else 0
            t_status = 'Pending_Human_Escalation' if is_escalated else 'Resolved_By_Agent'
            priority = 'Critical' if pol < -0.4 else ('High' if is_escalated else 'Medium')
            t_id = f"TCK-{lid.replace('LRN-', '')}"
            
            # Module association
            mod_assoc = None
            f_lower = feedback.lower()
            if 'sql' in f_lower: mod_assoc = 'MOD-101'
            elif 'pandas' in f_lower or 'numpy' in f_lower: mod_assoc = 'MOD-102'
            elif 'supervised' in f_lower or 'gradient' in f_lower: mod_assoc = 'MOD-201'
            elif 'pytorch' in f_lower or 'lstm' in f_lower: mod_assoc = 'MOD-301'
            elif 'word2vec' in f_lower or 'spacy' in f_lower or 'textblob' in f_lower or 'n8n' in f_lower or 'rag' in f_lower: mod_assoc = 'MOD-401'
            
            tickets_records.append((t_id, lid, mod_assoc, feedback, pol, subj, t_status, priority, is_escalated))
            
        # Simulated interaction log
        logs_records.append((lid, 'lab_attempt', f"/courses/ai/sprint{min(idx%4+1, 4)}", 1200 + (idx % 800)))

    # Bulk insert
    cur.executemany("INSERT INTO learners (learner_id, cohort_id, full_name, email, cohort_city, enrollment_status, dropped_out) VALUES (?, ?, ?, ?, ?, ?, ?)", learners_records)
    cur.executemany("INSERT INTO quiz_grades (learner_id, module_id, week_number, score, is_makeup) VALUES (?, ?, ?, ?, ?)", grades_records)
    cur.executemany("INSERT INTO support_tickets (ticket_id, learner_id, module_id, query_text, sentiment_polarity, sentiment_subjectivity, status, priority, escalation_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tickets_records)
    cur.executemany("INSERT INTO interaction_logs (learner_id, action_type, lms_page, session_duration_sec) VALUES (?, ?, ?, ?)", logs_records)
    
    conn.commit()
    
    cur.execute("SELECT COUNT(*) FROM learners")
    print(f"Total learners ingested: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM quiz_grades")
    print(f"Total quiz grades ingested: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM support_tickets")
    print(f"Total support tickets ingested: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM interaction_logs")
    print(f"Total interaction logs ingested: {cur.fetchone()[0]}")
    
    conn.close()
    print("Database population complete!")

if __name__ == '__main__':
    populate_database()
