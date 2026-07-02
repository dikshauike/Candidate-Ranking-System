import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# mapping local working files
root_dir = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\data")
cand_source = root_dir / "candidates.jsonl"
jd_source = root_dir / "job_description.md"

# O/P data stores linked to the new output path
output_path = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\output")
vector_dump_file = output_path / "candidate_embeddings.pkl"
jd_dump_file = output_path / "jd_embedding.pkl"


def build_searchable_string(cand_obj):
    # squashes relevant profile fields into a single block of text for vector matching.
    # limits loops to prevent making individual text blobs overly large.
    
    text_segments = []
    summary_data = cand_obj.get("profile", {})

    # pulling primary baseline attributes
    text_segments.append(
        f"Title: {summary_data.get('current_title', '')} with {summary_data.get('years_of_experience', 0)} years exp."
    )

    # grabbing skills array and limiting to a top slice to save context window
    declared_skills = cand_obj.get("skills", [])
    parsed_skills = [
        f"{item['name']} ({item.get('proficiency','')})"
        for item in declared_skills
        if isinstance(item, dict)
    ][:15]

    if parsed_skills:
        text_segments.append("Skills: " + ", ".join(parsed_skills))

    # appending history summary up to 3 past nodes
    past_jobs = cand_obj.get("career_history", [])
    for job_node in past_jobs[:3]:
        if isinstance(job_node, dict):
            text_segments.append(
                f"Worked as {job_node.get('title', '')} at {job_node.get('company', '')}."
            )

    return " ".join(text_segments)


def main():
    bench_start = datetime.now()
    
    # Automatically create the output directory if it doesn't exist yet
    output_path.mkdir(parents=True, exist_ok=True)

    print("Initializing SentenceTransformer instance locally...")

    # using a small, lightweight architecture to keep the processing footprint fast on local hardware
    transformer = SentenceTransformer("all-MiniLM-L6-v2")

    # processing Job Description File
    print("Generating vector representation for target JD...")
    with open(jd_source, "r", encoding="utf-8") as f:
        full_jd_body = f.read()

    # taking the first chunk of text to prevent padding inflation issues
    target_jd_chunk = full_jd_body[:2000]
    jd_vector = transformer.encode([target_jd_chunk])[0]

    with open(jd_dump_file, "wb") as f:
        pickle.dump(jd_vector, f)

    # processing Massive Candidate Stream
    print("Starting chunked vectorized compression for the 100k repository...")
    saved_vectors = []
    saved_ids = []

    # Keeping cache queues shallow so low-spec machines don't bottleneck or memory leak
    batch_limit = 5000
    cache_string_buffer = []
    cache_id_buffer = []

    with open(cand_source, "r", encoding="utf-8") as stream:
        for cursor, raw_line in enumerate(stream):
            profile_payload = json.loads(raw_line)
            cache_string_buffer.append(build_searchable_string(profile_payload))
            cache_id_buffer.append(profile_payload.get("candidate_id", ""))

            # Flushing accumulated arrays out to transformer encoding phase
            if len(cache_string_buffer) >= batch_limit:
                vector_batch = transformer.encode(
                    cache_string_buffer, show_progress_bar=False
                )
                saved_vectors.extend(vector_batch)
                saved_ids.extend(cache_id_buffer)

                print(f" -> Dispatched embeddings counter: {len(saved_ids)} / 100000")
                cache_string_buffer = []
                cache_id_buffer = []

        # Catching leftover floating entries from the stream terminal boundary
        if cache_string_buffer:
            vector_batch = transformer.encode(
                cache_string_buffer, show_progress_bar=False
            )
            saved_vectors.extend(vector_batch)
            saved_ids.extend(cache_id_buffer)

    # --- Archiving Outputs ---
    print("Streaming memory references into local disk pickle registry...")
    packaged_payload = {
        "candidate_ids": saved_ids,
        "embeddings": np.array(saved_vectors),
    }

    with open(vector_dump_file, "wb") as out_stream:
        pickle.dump(packaged_payload, out_stream)

    execution_delta = (datetime.now() - bench_start).total_seconds() / 60.0
    print(f"\nExecution benchmark resolved successfully in {execution_delta:.2f} minutes.")


if __name__ == "__main__":
    main()