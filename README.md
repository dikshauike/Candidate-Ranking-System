Create a new text file in your Redrob_Hackathon folder, paste this inside, and save it as README.md. (You can edit this later to add your own style).

🧠 Redrob AI Candidate Ranking System
This project is a solution for the Redrob Hackathon - Intelligent Candidate Discovery & Ranking Challenge. It ranks 100,000 candidates for a Senior AI Engineer role based on semantic similarity, experience, behavioral signals, and culture fit, while actively filtering out "honeypot" (fake) profiles.

🏗️ Architecture & Methodology
The system uses a two-phase approach to bypass the 5-minute CPU compute limit:

Pre-Computation Phase (step3_embeddings.py):
Parses 100,000 candidate profiles (skills, experience, education) into text.
Uses sentence-transformers/all-MiniLM-L6-v2 to generate 384-dimensional embeddings.
Saves embeddings to a .pkl file for fast loading.
Ranking Phase (step4_final_ranker.py):
Loads pre-computed embeddings.
Calculates cosine similarity between the Job Description and candidates.
Honeypot Filtering: Loads a pre-scanned blacklist of 97 fake profiles (identified by experience mismatches, keyword stuffing, and impossible behavioral twins).
Hybrid Scoring: Calculates a final score using weights:
40% Semantic Similarity
25% Experience Relevance (Target: 5-9 years)
25% Behavioral Signals (Activity recency, response rate)
10% Culture Fit (Product company background)
Generates a 1-2 sentence reasoning for the top 100 candidates.
🚀 How to Run Locally
Clone the repository.
Install dependencies: pip install -r requirements.txt
Place candidates.jsonl in a data/ folder.
Step 1: Run python step5_final_honeypot_detector.py (Scans for traps and saves honeypots_detected.json).
Step 2: Run python step3_embeddings.py (Takes ~60-75 mins. Generates .pkl files).
Step 3: Run python step4_final_ranker.py (Takes ~30 secs. Generates submission.csv).
🛡️ Honeypot Detection Strategy
Instead of just keyword matching, the system scans for logical impossibilities:

Stated experience vs. Calculated experience (e.g., claims 13 years but history shows 1 year).
"Expert" proficiency in skills with 0 months of duration.
High search appearances but 0 applications (Behavioral twins).
🛠️ Tech Stack
Python 3.11
Sentence-Transformers (HuggingFace)
Pandas & Scikit-Learn
