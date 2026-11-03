import sqlite3
import os

sql_path = r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone\database\setup_and_queries.sql'
db_path = r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone\database\sprints_learners.db'

with open(sql_path, 'r', encoding='utf-8') as f:
    sql_script = f.read()

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.executescript(sql_script)
conn.commit()

# Verify tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print('Tables in database:', [t[0] for t in tables])

# Query 3 test:
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
print('Governorate Performance & Drop-Out Distribution:')
for row in cur.execute(q3).fetchall():
    print(row)

conn.close()
print('\nSprint 1 Database Execution: 100% SUCCESS!')
