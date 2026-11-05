/* ==============================================================================
   SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
   Sprint 1: Relational Database Schema Definition (DDL)
   File: database/schema.sql
   Dialect: SQLite / PostgreSQL ANSI-compatible
   ============================================================================== */

PRAGMA foreign_keys = ON;

-- Clean existing tables if present
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS quiz_grades;
DROP TABLE IF EXISTS interaction_logs;
DROP TABLE IF EXISTS course_modules;
DROP TABLE IF EXISTS learners;
DROP TABLE IF EXISTS cohorts;

-- 1. Cohorts Table
CREATE TABLE cohorts (
    cohort_id VARCHAR(20) PRIMARY KEY,
    cohort_name VARCHAR(100) NOT NULL,
    track VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Learners Table
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

-- 3. Course Modules Table
CREATE TABLE course_modules (
    module_id VARCHAR(20) PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,
    sprint_number INT CHECK (sprint_number BETWEEN 1 AND 4) NOT NULL,
    week_number INT CHECK (week_number BETWEEN 1 AND 5) NOT NULL,
    difficulty_level VARCHAR(20) CHECK (difficulty_level IN ('Beginner', 'Intermediate', 'Advanced')) NOT NULL,
    instructor_name VARCHAR(100) NOT NULL
);

-- 4. Quiz Grades Table
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

-- 5. Support Tickets Table
CREATE TABLE support_tickets (
    ticket_id VARCHAR(30) PRIMARY KEY,
    learner_id VARCHAR(20) NOT NULL,
    module_id VARCHAR(20),
    query_text TEXT NOT NULL,
    sentiment_polarity REAL CHECK (sentiment_polarity >= -1.0 AND sentiment_polarity <= 1.0),
    sentiment_subjectivity REAL CHECK (sentiment_subjectivity >= 0.0 AND sentiment_subjectivity <= 1.0),
    status VARCHAR(30) CHECK (status IN ('Open', 'Pending_Human_Escalation', 'Resolved_By_Agent', 'Resolved_By_Mentor')) NOT NULL,
    priority VARCHAR(15) CHECK (priority IN ('Low', 'Medium', 'High', 'Critical')) NOT NULL,
    escalation_flag BOOLEAN DEFAULT 0,
    correlation_id VARCHAR(64),
    idempotency_key VARCHAR(64) UNIQUE,
    payload_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE,
    FOREIGN KEY (module_id) REFERENCES course_modules(module_id) ON DELETE SET NULL
);

-- 6. Interaction Logs Table
CREATE TABLE interaction_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id VARCHAR(20) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    lms_page VARCHAR(100) NOT NULL,
    session_duration_sec INT CHECK (session_duration_sec >= 0),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE
);

-- Optimized Performance Indexes
CREATE INDEX idx_learners_city ON learners(cohort_city);
CREATE INDEX idx_learners_status ON learners(enrollment_status);
CREATE INDEX idx_learners_dropped ON learners(dropped_out);
CREATE INDEX idx_quiz_learner_week ON quiz_grades(learner_id, week_number);
CREATE INDEX idx_tickets_sentiment ON support_tickets(sentiment_polarity, escalation_flag);
CREATE INDEX idx_tickets_status ON support_tickets(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_idempotency ON support_tickets(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_tickets_correlation ON support_tickets(correlation_id);
CREATE INDEX idx_logs_learner_action ON interaction_logs(learner_id, action_type);
