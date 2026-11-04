/* ==============================================================================
   SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
   Sprint 1: Relational Database Core Seed Data (DML)
   File: database/seed.sql
   ============================================================================== */

-- Insert Cohorts
INSERT INTO cohorts (cohort_id, cohort_name, track, start_date, end_date) VALUES
('COHORT-AI-01', 'AI Pioneers National Cohort 2026', 'AI & Machine Learning', '2026-10-01', '2026-11-15'),
('COHORT-DS-05', 'Data Science Applied Bootcamp 2026', 'Data Science', '2026-09-15', '2026-10-30');

-- Insert Course Modules
INSERT INTO course_modules (module_id, module_name, sprint_number, week_number, difficulty_level, instructor_name) VALUES
('MOD-101', 'Relational Databases & SQL Analytics', 1, 1, 'Beginner', 'Tarek Mahmoud'),
('MOD-102', 'Data Wrangling with pandas & NumPy', 1, 2, 'Beginner', 'Salma Youssef'),
('MOD-201', 'Supervised Machine Learning & Feature Engineering', 2, 3, 'Intermediate', 'Dr. Ahmed Khaled'),
('MOD-301', 'Deep Sequence Modeling & PyTorch LSTMs', 3, 4, 'Advanced', 'Tarek Mahmoud'),
('MOD-401', 'NLP Embeddings (Word2Vec) & n8n RAG Agents', 4, 5, 'Advanced', 'Omar Reda');
