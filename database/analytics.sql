/* ==============================================================================
   SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
   Sprint 1: Analytical SQL Queries
   File: database/analytics.sql
   ============================================================================== */

-- ------------------------------------------------------------------------------
-- Query 1: Complete Historical Interaction & Performance Log Extraction
-- ------------------------------------------------------------------------------
SELECT 
    l.learner_id,
    l.full_name,
    l.cohort_city,
    l.enrollment_status,
    q.week_number,
    m.module_name,
    q.score AS quiz_score,
    t.ticket_id,
    t.query_text,
    t.sentiment_polarity,
    t.status AS ticket_status,
    t.escalation_flag
FROM learners l
LEFT JOIN quiz_grades q ON l.learner_id = q.learner_id
LEFT JOIN course_modules m ON q.module_id = m.module_id
LEFT JOIN support_tickets t ON l.learner_id = t.learner_id AND (t.module_id = m.module_id OR t.module_id IS NULL)
ORDER BY l.learner_id, q.week_number;

-- ------------------------------------------------------------------------------
-- Query 2: Progressive Grade Degradation Detection (Early Risk Identification)
-- ------------------------------------------------------------------------------
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
    l.full_name,
    l.cohort_city,
    w1.w1_score,
    s.week_number AS degraded_at_week,
    s.current_score,
    (w1.w1_score - s.current_score) AS score_drop
FROM learners l
JOIN week1_scores w1 ON l.learner_id = w1.learner_id
JOIN subsequent_scores s ON l.learner_id = s.learner_id
WHERE (w1.w1_score - s.current_score) >= 15.0
ORDER BY score_drop DESC;

-- ------------------------------------------------------------------------------
-- Query 3: Governorate Performance & Drop-Out Distribution
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- Query 4: Deliberative Escalation Target Identification
-- ------------------------------------------------------------------------------
SELECT 
    l.learner_id,
    l.full_name,
    l.cohort_city,
    ROUND(AVG(q.score), 2) AS current_avg_quiz_score,
    t.ticket_id,
    t.sentiment_polarity,
    t.query_text,
    t.priority,
    t.escalation_flag
FROM learners l
JOIN support_tickets t ON l.learner_id = t.learner_id
JOIN quiz_grades q ON l.learner_id = q.learner_id
WHERE t.sentiment_polarity < -0.15
GROUP BY l.learner_id, t.ticket_id
HAVING AVG(q.score) < 65.0
ORDER BY t.sentiment_polarity ASC;

-- ------------------------------------------------------------------------------
-- Query 5: Module Bottleneck & Difficulty Analysis (Fixed CTEs - No Join Multiplication)
-- ------------------------------------------------------------------------------
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
