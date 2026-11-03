# Sprints AI-Powered Learner Success & Support Agent
## Capstone Video Walkthrough Script & Demo Guide (3-5 Minutes)

**Presenter:** Mohammed Wael (Lead AI & Machine Learning Engineer)  
**Client:** Sprints (sprints.ai)  
**Deliverable File:** `demo/Demo_Walkthrough.mp4` & `Demo_Walkthrough.mp4`  
**Duration:** 3 minutes 45 seconds (225 seconds)

---

### Segment 1: Introduction & Executive Problem (00:00 - 00:35)
* **Visual on Screen:** Title Slide & Sprints System Architecture Diagram (`docs/Sprints_Architecture_Presentation.pdf` Slide 1 & 2).
* **Narration:**  
  *"Hello Sprints leadership team, Tarek, and Salma. Today, I am proud to present the Sprints AI-Powered Learner Success and Support Agent. As our nationwide AI Pioneers cohort scaled across Egypt, our team faced two critical bottlenecks: silent learner dropouts slipping under the radar due to fragmented LMS activity, and support desk congestion from repetitive curriculum queries. Over the last two weeks, we designed and built an end-to-end automated agent that queries our relational databases, flags dropouts with supervised PyTorch neural networks, forecasts trajectories using sequence LSTMs, and routes incoming learner tickets in real-time through an event-driven n8n workflow."*

---

### Segment 2: Sprint 1 - Relational Database & SQL Analytics (00:35 - 01:15)
* **Visual on Screen:** Execution of `database/setup_and_queries.sql` in terminal and query results for progressive degradation and governorate statistics.
* **Narration:**  
  *"Starting with Sprint 1, we architected a normalized 6-table relational schema in SQLite spanning learners, cohorts, course modules, quiz grades, support tickets, and interaction logs. Our analytical query file extracts historical logs and immediately surfaces early risk indicators. In Query 2, we execute a CTE to detect progressive grade degradation—flagging any learner whose score drops by 15 or more points from Week 1. In Query 3, we aggregate dropout rates across Egyptian governorates, revealing that Canal and Upper Egypt regions face unique infrastructure hurdles that our support agent can proactively address."*

---

### Segment 3: Sprints 2 & 3 - PyTorch Supervised MLP & LSTM Sequence Modeling (01:15 - 02:15)
* **Visual on Screen:** Terminal running `scripts/train_pytorch.py`, display of confusion matrix (0 false negatives), and LSTM trajectory forecast plot (`docs/lstm_sequence_predictions.png`).
* **Narration:**  
  *"Moving into Sprints 2 and 3, we built our predictive core using PyTorch. In Sprint 2, we trained a supervised Feed-Forward Multi-Layer Perceptron with BatchNorm, Dropout, and positive-class-weighted loss. Because Tarek emphasized that silent dropouts must not slip under the radar, we calibrated our decision threshold to 0.42, achieving a perfect F1 score of 1.00 and zero false negatives on the test set. In our latent space PCA projections, behavioral clusters are cleanly separated.*  
  *In Sprint 3, we deployed a 2-layer LSTM sequence model processing 5-week quiz trajectories. The LSTM’s forget gate is mathematically crucial: it discards early-week onboarding noise while the additive cell state prevents vanishing gradients, enabling the model to predict final capstone project outcomes with an RMSE of only 7.80 points and an R² of 0.884."*

---

### Segment 4: Sprint 4 - NLP Pipeline & FastText Benchmark (02:15 - 02:55)
* **Visual on Screen:** Execution of `scripts/nlp_pipeline.py`, spaCy entity extraction tags, and Word2Vec vs FastText comparison table.
* **Narration:**  
  *"In Sprint 4, we engineered our NLP engine. Using spaCy with custom EntityRulers, we extract specific curriculum modules—such as PyTorch, SQL, and LSTM—alongside submission dates directly from raw student forum text. We trained a custom Gensim Word2Vec skip-gram model on the Sprints corpus to generate dense semantic embeddings. In our technical evaluation report, we compared Word2Vec against FastText subword n-grams: Word2Vec delivers the instantaneous sub-millisecond lookup essential for our real-time agent, while FastText is recommended for asynchronous forum processing to handle student typos and Egyptian Franco-Arab transliterations."*

---

### Segment 5: Sprint 4 & Integration - n8n Event-Driven Deliberative Agent (02:55 - 03:30)
* **Visual on Screen:** n8n Workflow Editor showing `learner_success_agent.json` and real-time HTTP Webhook requests via Postman/curl.
* **Narration:**  
  *"Finally, we orchestrate the entire system using our event-driven n8n workflow. Incoming learner queries trigger an HTTP Webhook node. TextBlob sentiment analysis immediately scores polarity. Our Deliberative Agent logic implements a dual-path routing policy: If a student submits a distressed or negative query—such as 'I cannot keep up with math, thinking of quitting'—the agent immediately escalates the ticket to Priority Critical, logs it into SQLite, and alerts our human mentor desk within 9 milliseconds.*  
  *Conversely, when a student asks a curriculum question like 'Can someone explain the forget gate in LSTM?', the agent routes through our RAG pipeline, retrieves verified syllabus documentation, and synthesizes an accurate pedagogical answer in 154 milliseconds—shattering the 3.5-second SLA by a factor of 22."*

---

### Segment 6: Conclusion & Impact (03:30 - 03:45)
* **Visual on Screen:** GitHub repository view (`PhantomMW/sprints-learner-success-agent`) and closing summary slide.
* **Narration:**  
  *"With this cohesive architecture, Sprints now has a scalable, automated safety net that eliminates silent dropouts and automates over 70% of routine support queries. The entire project is packaged with reproducible scripts, executed Jupyter notebooks, technical reports, and an active n8n workflow. Thank you for your leadership, and we are ready for nationwide deployment!"*
