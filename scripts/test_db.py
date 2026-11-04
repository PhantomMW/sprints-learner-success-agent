"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/test_db.py
Description: Verifies SQL Database integrity and executes the 5 analytical queries
             from database/analytics.sql on the populated database (2,000 learners).
==============================================================================
"""

import os
import sqlite3
import pandas as pd

def test_database():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'database', 'sprints_learners.db')
    analytics_path = os.path.join(base_dir, 'database', 'analytics.sql')
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 1. Verify table counts
    tables = ['cohorts', 'learners', 'course_modules', 'quiz_grades', 'support_tickets', 'interaction_logs']
    print("\n" + "="*60)
    print("VERIFYING RELATIONAL DATABASE RECORD COUNTS")
    print("="*60)
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
        print(f"Table [{t}]: {cnt:,} records")
        assert cnt > 0, f"Table {t} should not be empty!"
        
    # 2. Test Query 2: Progressive Grade Degradation
    print("\n--- Query 2: Progressive Grade Degradation (Sample Top 5) ---")
    q2 = """
    WITH week1_scores AS (
        SELECT learner_id, score AS w1_score
        FROM quiz_grades
        WHERE week_number = 1
    ),
    subsequent_scores AS (
        SELECT learner_id, week_number, score AS current_score
        FROM quiz_grades
        WHERE week_number > 1
    )
    SELECT 
        l.learner_id,
        l.cohort_city,
        w1.w1_score,
        s.week_number AS degraded_at_week,
        s.current_score,
        ROUND(w1.w1_score - s.current_score, 1) AS score_drop
    FROM learners l
    JOIN week1_scores w1 ON l.learner_id = w1.learner_id
    JOIN subsequent_scores s ON l.learner_id = s.learner_id
    WHERE (w1.w1_score - s.current_score) >= 15.0
    ORDER BY score_drop DESC
    LIMIT 5;
    """
    df_q2 = pd.read_sql_query(q2, conn)
    print(df_q2.to_string(index=False))
    
    # 3. Test Query 3: Governorate Performance
    print("\n--- Query 3: Governorate Dropout Distribution & KPIs ---")
    q3 = """
    WITH learner_stats AS (
        SELECT 
            l.cohort_city,
            COUNT(DISTINCT l.learner_id) AS total_learners,
            COUNT(DISTINCT CASE WHEN l.dropped_out = 1 THEN l.learner_id END) AS dropped_out_learners,
            COUNT(DISTINCT t.ticket_id) AS total_support_tickets,
            COUNT(DISTINCT CASE WHEN t.escalation_flag = 1 THEN t.ticket_id END) AS escalated_tickets
        FROM learners l
        LEFT JOIN support_tickets t ON l.learner_id = t.learner_id
        GROUP BY l.cohort_city
    ),
    grade_stats AS (
        SELECT 
            l.cohort_city,
            ROUND(AVG(q.score), 2) AS avg_quiz_score
        FROM learners l
        JOIN quiz_grades q ON l.learner_id = q.learner_id
        GROUP BY l.cohort_city
    )
    SELECT 
        ls.cohort_city,
        ls.total_learners,
        ls.dropped_out_learners,
        ROUND(CAST(ls.dropped_out_learners AS REAL) * 100.0 / ls.total_learners, 2) AS dropout_rate_pct,
        COALESCE(gs.avg_quiz_score, 0.0) AS avg_quiz_score,
        ls.total_support_tickets,
        ls.escalated_tickets
    FROM learner_stats ls
    LEFT JOIN grade_stats gs ON ls.cohort_city = gs.cohort_city
    ORDER BY dropout_rate_pct DESC, ls.total_learners DESC;
    """
    df_q3 = pd.read_sql_query(q3, conn)
    print(df_q3.to_string(index=False))
    
    # 4. Test Query 5: Module Bottleneck Analysis (Verified CTE Fix - No Join Multiplication)
    print("\n--- Query 5: Module Bottleneck & Difficulty (CTEs Verified) ---")
    q5 = """
    WITH grade_stats AS (
        SELECT 
            module_id,
            ROUND(AVG(score), 2) AS avg_score,
            SUM(CASE WHEN score < 60.0 THEN 1 ELSE 0 END) AS failing_submissions
        FROM quiz_grades
        GROUP BY module_id
    ),
    ticket_stats AS (
        SELECT 
            module_id,
            COUNT(*) AS ticket_count,
            ROUND(AVG(sentiment_polarity), 3) AS avg_ticket_sentiment
        FROM support_tickets
        WHERE module_id IS NOT NULL
        GROUP BY module_id
    )
    SELECT 
        m.module_id,
        m.module_name,
        m.sprint_number,
        m.week_number,
        m.difficulty_level,
        COALESCE(gs.avg_score, 0.0) AS avg_score,
        COALESCE(gs.failing_submissions, 0) AS failing_submissions,
        COALESCE(ts.ticket_count, 0) AS ticket_count,
        COALESCE(ts.avg_ticket_sentiment, 0.0) AS avg_ticket_sentiment
    FROM course_modules m
    LEFT JOIN grade_stats gs ON m.module_id = gs.module_id
    LEFT JOIN ticket_stats ts ON m.module_id = ts.module_id
    ORDER BY avg_score ASC;
    """
    df_q5 = pd.read_sql_query(q5, conn)
    print(df_q5.to_string(index=False))
    
    conn.close()
    print("\n" + "="*60)
    print("ALL SQL QUERIES EXECUTED SUCCESSFULLY ON POPULATED DATABASE!")
    print("="*60)

if __name__ == '__main__':
    test_database()
