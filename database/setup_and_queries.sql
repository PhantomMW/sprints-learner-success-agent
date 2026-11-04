/* ==============================================================================
   SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
   Sprint 1: Practical Data Analytics & Relational Database Architecture
   File: database/setup_and_queries.sql
   Dialect: SQLite / PostgreSQL ANSI-compatible
   ============================================================================== */

-- Enable Foreign Key constraints
PRAGMA foreign_keys = ON;

-- Clean existing tables if present
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS quiz_grades;
DROP TABLE IF EXISTS interaction_logs;
DROP TABLE IF EXISTS course_modules;
DROP TABLE IF EXISTS learners;
DROP TABLE IF EXISTS cohorts;

-- ==============================================================================
-- 1. DDL: TABLE DEFINITIONS & SCHEMA CONSTRAINTS
-- ==============================================================================

-- Table 1: Cohorts
CREATE TABLE cohorts (
    cohort_id VARCHAR(20) PRIMARY KEY,
    cohort_name VARCHAR(100) NOT NULL,
    track VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Learners
CREATE TABLE learners (
    learner_id VARCHAR(20) PRIMARY KEY,
    cohort_id VARCHAR(20) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    cohort_city VARCHAR(50) NOT NULL,
    enrollment_status VARCHAR(20) CHECK (enrollment_status IN ('Active', 'Dropped', 'Graduated', 'Suspended')) NOT NULL,
    dropped_out BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cohort_id) REFERENCES cohorts(cohort_id) ON DELETE CASCADE
);

-- Table 3: Course Modules
CREATE TABLE course_modules (
    module_id VARCHAR(20) PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,
    sprint_number INT CHECK (sprint_number BETWEEN 1 AND 4) NOT NULL,
    week_number INT CHECK (week_number BETWEEN 1 AND 5) NOT NULL,
    difficulty_level VARCHAR(20) CHECK (difficulty_level IN ('Beginner', 'Intermediate', 'Advanced')) NOT NULL,
    instructor_name VARCHAR(100) NOT NULL
);

-- Table 4: Quiz Grades & Project Assessments
CREATE TABLE quiz_grades (
    grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id VARCHAR(20) NOT NULL,
    module_id VARCHAR(20) NOT NULL,
    week_number INT CHECK (week_number BETWEEN 1 AND 5) NOT NULL,
    score REAL CHECK (score >= 0.0 AND score <= 100.0),
    is_makeup BOOLEAN DEFAULT 0,
    submission_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE,
    FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE CASCADE
);

-- Table 5: Support Tickets (LMS Support Desk Interactions)
CREATE TABLE support_tickets (
    ticket_id VARCHAR(30) PRIMARY KEY,
    learner_id VARCHAR(20) NOT NULL,
    module_id VARCHAR(20),
    query_text TEXT NOT NULL,
    sentiment_polarity REAL CHECK (sentiment_polarity >= -1.0 AND sentiment_polarity <= 1.0),
    sentiment_subjectivity REAL CHECK (sentiment_subjectivity >= 0.0 AND sentiment_subjectivity <= 1.0),
    status VARCHAR(20) CHECK (status IN ('Open', 'Pending_Human_Escalation', 'Resolved_By_Agent', 'Resolved_By_Mentor')) NOT NULL,
    priority VARCHAR(15) CHECK (priority IN ('Low', 'Medium', 'High', 'Critical')) NOT NULL,
    escalation_flag BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE,
    FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE SET NULL
);

-- Table 6: Interaction Logs (Fragmented LMS Activity)
CREATE TABLE interaction_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id VARCHAR(20) NOT NULL,
    action_type VARCHAR(50) NOT NULL, -- e.g., 'video_watch', 'lab_attempt', 'forum_post', 'doc_download'
    lms_page VARCHAR(100) NOT NULL,
    session_duration_sec INT CHECK (session_duration_sec >= 0),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE
);

-- Indexes for optimized querying and analytics
CREATE INDEX idx_learners_city ON learners(cohort_city);
CREATE INDEX idx_learners_status ON learners(enrollment_status);
CREATE INDEX idx_quiz_learner ON quiz_grades(learner_id, week_number);
CREATE INDEX idx_tickets_sentiment ON support_tickets(sentiment_polarity, escalation_flag);
CREATE INDEX idx_logs_learner_action ON interaction_logs(learner_id, action_type);

-- ==============================================================================
-- 2. DML: SEED DATA INSERTION (BOOTCAMP COHORTS ACROSS EGYPT)
-- ==============================================================================

-- Insert Cohorts
INSERT INTO cohorts (cohort_id, cohort_name, track, start_date, end_date) VALUES
('COHORT-AI-01', 'AI Pioneers National Cohort 2026', 'AI & Machine Learning', '2026-10-01', '2026-11-15'),
('COHORT-DS-05', 'Data Science Applied Bootcamp 2026', 'Data Science', '2026-09-15', '2026-10-30');

-- Insert Course Modules (Curriculum Modules)
INSERT INTO course_modules (module_id, module_name, sprint_number, week_number, difficulty_level, instructor_name) VALUES
('MOD-101', 'Relational Databases & SQL Analytics', 1, 1, 'Beginner', 'Tarek Mahmoud'),
('MOD-102', 'Data Wrangling with pandas & NumPy', 1, 2, 'Beginner', 'Salma Youssef'),
('MOD-201', 'Supervised Machine Learning & Feature Engineering', 2, 3, 'Intermediate', 'Dr. Ahmed Khaled'),
('MOD-301', 'Deep Sequence Modeling & PyTorch LSTMs', 3, 4, 'Advanced', 'Tarek Mahmoud'),
('MOD-401', 'NLP Embeddings (Word2Vec) & n8n RAG Agents', 4, 5, 'Advanced', 'Omar Reda');

-- Insert Sample Learners (From Cairo, Alexandria, Giza, Assiut, Mansoura, etc.)
INSERT INTO learners (learner_id, cohort_id, full_name, email, cohort_city, enrollment_status, dropped_out) VALUES
('LRN-1001', 'COHORT-AI-01', 'Ahmed Youssef', 'ahmed.y@sprints.ai', 'Cairo', 'Active', 0),
('LRN-1002', 'COHORT-AI-01', 'Salma Tareq', 'salma.t@sprints.ai', 'Alexandria', 'Active', 0),
('LRN-1003', 'COHORT-AI-01', 'Mahmoud Hassan', 'm.hassan@sprints.ai', 'Giza', 'Dropped', 1),
('LRN-1004', 'COHORT-AI-01', 'Nouran El-Sayed', 'nouran.s@sprints.ai', 'Mansoura', 'Active', 0),
('LRN-1005', 'COHORT-AI-01', 'Kareem Adel', 'kareem.a@sprints.ai', 'Assiut', 'Dropped', 1),
('LRN-1006', 'COHORT-AI-01', 'Mona Ibrahim', 'mona.i@sprints.ai', 'Cairo', 'Active', 0),
('LRN-1007', 'COHORT-AI-01', 'Omar Fathy', 'omar.f@sprints.ai', 'Tanta', 'Active', 0),
('LRN-1008', 'COHORT-AI-01', 'Sara Samir', 'sara.s@sprints.ai', 'Ismailia', 'Dropped', 1),
('LRN-1009', 'COHORT-AI-01', 'Youssef Kamal', 'youssef.k@sprints.ai', 'Cairo', 'Active', 0),
('LRN-1010', 'COHORT-AI-01', 'Hadeer Mostafa', 'hadeer.m@sprints.ai', 'Alexandria', 'Dropped', 1);

-- Insert Quiz Grades across 5 weeks
INSERT INTO quiz_grades (learner_id, module_id, week_number, score, is_makeup) VALUES
-- LRN-1001 (Cairo, Active, Steady/Slight drop in PyTorch)
('LRN-1001', 'MOD-101', 1, 85.0, 0),
('LRN-1001', 'MOD-102', 2, 80.0, 0),
('LRN-1001', 'MOD-201', 3, 75.0, 0),
('LRN-1001', 'MOD-301', 4, 70.0, 0),
('LRN-1001', 'MOD-401', 5, 60.0, 0),
-- LRN-1002 (Alexandria, Top Performer)
('LRN-1002', 'MOD-101', 1, 95.0, 0),
('LRN-1002', 'MOD-102', 2, 92.0, 0),
('LRN-1002', 'MOD-201', 3, 98.0, 0),
('LRN-1002', 'MOD-301', 4, 95.0, 0),
('LRN-1002', 'MOD-401', 5, 100.0, 0),
-- LRN-1003 (Giza, Dropped after week 2)
('LRN-1003', 'MOD-101', 1, 40.0, 0),
('LRN-1003', 'MOD-102', 2, 30.0, 0),
-- LRN-1004 (Mansoura, Active)
('LRN-1004', 'MOD-101', 1, 78.0, 0),
('LRN-1004', 'MOD-102', 2, 80.0, 1),
('LRN-1004', 'MOD-201', 3, 82.0, 0),
('LRN-1004', 'MOD-301', 4, 79.0, 0),
('LRN-1004', 'MOD-401', 5, 85.0, 0),
-- LRN-1005 (Assiut, Dropped - Steep decline)
('LRN-1005', 'MOD-101', 1, 60.0, 0),
('LRN-1005', 'MOD-102', 2, 55.0, 0),
('LRN-1005', 'MOD-201', 3, 45.0, 0),
('LRN-1005', 'MOD-301', 4, 40.0, 0),
('LRN-1005', 'MOD-401', 5, 35.0, 0),
-- LRN-1006 (Cairo, Strong)
('LRN-1006', 'MOD-101', 1, 88.0, 0),
('LRN-1006', 'MOD-102', 2, 85.0, 0),
('LRN-1006', 'MOD-201', 3, 90.0, 0),
('LRN-1006', 'MOD-401', 5, 89.0, 0),
-- LRN-1008 (Ismailia, Early drop)
('LRN-1008', 'MOD-101', 1, 25.0, 0),
('LRN-1008', 'MOD-102', 2, 20.0, 0),
('LRN-1008', 'MOD-201', 3, 15.0, 0);

-- Insert Support Tickets
INSERT INTO support_tickets (ticket_id, learner_id, module_id, query_text, sentiment_polarity, sentiment_subjectivity, status, priority, escalation_flag, created_at, resolved_at) VALUES
('TCK-2026-001', 'LRN-1001', 'MOD-301', 'I am struggling a bit with PyTorch tensors. The module on Oct 12 was too fast for me.', -0.25, 0.60, 'Resolved_By_Agent', 'Medium', 0, '2026-10-13 10:15:00', '2026-10-13 10:15:02'),
('TCK-2026-002', 'LRN-1002', 'MOD-101', 'Loved the SQL Advanced joins session! Great instructor.', 0.70, 0.60, 'Resolved_By_Agent', 'Low', 0, '2026-10-04 14:20:00', '2026-10-04 14:20:01'),
('TCK-2026-003', 'LRN-1003', 'MOD-102', 'I cannot keep up with the math in Week 2. Thinking of quitting.', -0.65, 0.80, 'Pending_Human_Escalation', 'Critical', 1, '2026-10-10 18:00:00', NULL),
('TCK-2026-004', 'LRN-1004', 'MOD-102', 'Where is the submission link for the pandas assignment due on Nov 5?', 0.00, 0.10, 'Resolved_By_Agent', 'Low', 0, '2026-10-08 09:30:00', '2026-10-08 09:30:01'),
('TCK-2026-005', 'LRN-1005', 'MOD-201', 'The gradient descent math is very confusing, I am completely lost.', -0.75, 0.90, 'Pending_Human_Escalation', 'High', 1, '2026-10-18 21:10:00', NULL),
('TCK-2026-006', 'LRN-1006', 'MOD-301', 'Can someone explain the forget gate in LSTM? The lecture from last Tuesday was cut off.', 0.05, 0.20, 'Resolved_By_Agent', 'Medium', 0, '2026-10-24 11:45:00', '2026-10-24 11:45:03'),
('TCK-2026-007', 'LRN-1008', 'MOD-101', 'Internet issues in my area made me miss the deadline for Sprint 1.', -0.30, 0.40, 'Pending_Human_Escalation', 'High', 1, '2026-10-07 16:30:00', NULL);

-- Insert Interaction Logs
INSERT INTO interaction_logs (learner_id, action_type, lms_page, session_duration_sec, timestamp) VALUES
('LRN-1001', 'video_watch', '/courses/ai/sprint3/lstm-architecture', 3420, '2026-10-12 11:00:00'),
('LRN-1001', 'lab_attempt', '/courses/ai/sprint3/pytorch-lab', 1800, '2026-10-12 12:30:00'),
('LRN-1002', 'lab_attempt', '/courses/ai/sprint1/sql-window-functions', 4500, '2026-10-03 16:00:00'),
('LRN-1003', 'doc_download', '/courses/ai/sprint1/numpy-cheatsheet', 120, '2026-10-05 14:00:00'),
('LRN-1005', 'forum_post', '/community/threads/gradient-descent', 240, '2026-10-18 21:05:00');

-- ==============================================================================
-- 3. ANALYTICAL QUERIES (SPRINT 1 REQUIREMENTS)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Query 1: Complete Historical Interaction & Performance Log Extraction
-- Purpose: Consolidate learner demographic, quiz performance, and ticket logs
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
-- Purpose: Flag learners whose score drops by 15+ points between Week 1 and any subsequent week
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
-- Purpose: Aggregate drop-out rates, quiz performance, and ticket volume by Egyptian governorate
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
-- Purpose: Find learners with negative sentiment support queries AND low academic scores (< 65)
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
-- Purpose: Identify modules causing the most learner drop and support tickets
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
