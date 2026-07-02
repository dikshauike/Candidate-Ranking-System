import gradio as gr
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_candidates(file, jd_text):
    if file is None:
        return pd.DataFrame({"Error": ["Please upload a JSON file"]})
    
    with open(file.name, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    cand_texts = []
    for c in candidates:
        profile = c.get('profile', {})
        skills = ", ".join([s['name'] for s in c.get('skills', []) if isinstance(s, dict)][:10])
        text = f"Title: {profile.get('current_title', '')} with {profile.get('years_of_experience', 0)} years exp. Skills: {skills}."
        cand_texts.append(text)
        
    cand_embeddings = model.encode(cand_texts)
    jd_embedding = model.encode([jd_text])
    similarities = cosine_similarity(jd_embedding, cand_embeddings)[0]
    
    results = []
    for i, c in enumerate(candidates):
        profile = c.get('profile', {})
        total_exp = profile.get('years_of_experience', 0)
        
        # 1. Semantic Score (0 to 1)
        sem_score = (similarities[i] + 1) / 2
        
        # 2. Experience Score (JD wants 5-9 years)
        if 5 <= total_exp <= 9:
            exp_score = 1.0
        elif total_exp > 9:
            exp_score = 0.8
        else:
            exp_score = total_exp / 5.0
            
        # 3. Culture Fit Score (Must have tech/software background)
        culture_score = 0
        for job in c.get('career_history', []):
            if isinstance(job, dict):
                industry = job.get('industry', '').lower()
                if 'software' in industry or 'internet' in industry or 'tech' in industry:
                    culture_score = 1.0
                    break
                    
        # Calculate Final Hybrid Score (Same logic as your step4 script)
        final_score = (sem_score * 0.40) + (exp_score * 0.25) + (0.25 * 0.5) + (culture_score * 0.10)
        
        # Only include if they have a tech background
        if culture_score > 0:
            results.append({
                'Rank': 0, # Will sort and assign rank later
                'candidate_id': c.get('candidate_id', ''),
                'current_title': profile.get('current_title', ''),
                'years_exp': total_exp,
                'final_score': final_score
            })
        
    # Sort and get top 10
    df = pd.DataFrame(results).sort_values('final_score', ascending=False).head(10).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    return df

# Create the Gradio Web UI
interface = gr.Interface(
    fn=rank_candidates,
    inputs=[
        gr.File(label="Upload sample_candidates.json"),
        gr.Textbox(lines=5, label="Paste Job Description here", value="Senior AI Engineer with 5-9 years experience in embeddings, retrieval, and production ML.")
    ],
    outputs=gr.Dataframe(label="Top 10 Ranked Candidates"),
    title="🤖 AI Candidate Ranker (Sandbox)",
    description="Upload the 50-candidate sample file and paste a JD to see the ranking logic."
)

if __name__ == "__main__":
    interface.launch()