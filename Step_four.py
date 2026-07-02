import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Local storage configuration paths
base_dir = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\data")
output_dir = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\output")

candidate_pool_path = base_dir / "candidates.jsonl"
honeypot_blacklist_path = output_dir / "honeypots_detected.json"

# Intermediate pre-computed vector files
vector_pool_pkl = "candidate_embeddings.pkl"
jd_vector_pkl = "jd_embedding.pkl"

# Final target compilation asset
output_submission_csv = output_dir / "submission.csv"

# Hyperparameter weights configured post-testing
w_s = 0.40
w_e = 0.25
w_b = 0.25
w_c = 0.10


def evaluate_candidate_metrics(candidate_record, semantic_metric):
    # Calculates a multi-dimensional weighted score using candidate profile states and pre-calculated semantic distances.
    profile_summary = candidate_record.get("profile", {})
    years_logged = profile_summary.get("years_of_experience", 0)

    # Experience heuristic optimization: Target window is strictly 5-9 years based on JD
    if 5 <= years_logged <= 9:
        experience_score = 1.0
    elif years_logged > 9:
        experience_score = 0.8
    else:
        # Scale lower experience linearly up to the minimum threshold bounds
        experience_score = years_logged / 5.0

    # Parsing telemetry arrays for operational platform visibility metrics
    platform_signals = candidate_record.get("redrob_signals", {})
    days_since_active = 999  # Safe penalty fallback bound

    try:
        activity_timestamp = datetime.strptime(
            platform_signals.get("last_active_date", ""), "%Y-%m-%d"
        )
        # Using competition absolute baseline cutoff date to calculate active delta windows accurately
        days_since_active = (datetime(2026, 6, 28) - activity_timestamp).days
    except (ValueError, TypeError):
        pass

    outreach_response_rate = platform_signals.get("recruiter_response_rate", 0)
    calculated_activity_index = max(0, 1 - (days_since_active / 90))

    # Merging interactive profile parameters to determine behavioral health
    behavioral_score = (calculated_activity_index * 0.6) + (
        outreach_response_rate * 0.4
    )

    # Screening past organizational profiles to isolate target environment history markers
    culture_score = 0.0
    for history_node in candidate_record.get("career_history", []):
        if isinstance(history_node, dict):
            vertical_market = history_node.get("industry", "").lower()
            if any(
                keyword in vertical_market
                for keyword in ["software", "internet", "tech"]
            ):
                culture_score = 1.0
                break

    # Summing individual weighted vectors into the unified baseline score
    aggregated_score = (
        (semantic_metric * w_s)
        + (experience_score * w_e)
        + (behavioral_score * w_b)
        + (culture_score * w_c)
    )

    return aggregated_score, culture_score


def stringify_reasoning_metadata(candidate_record, target_culture_score):
    # Builds a human-readable summary string outlining primary profile criteria for tracking references.
    profile_summary = candidate_record.get("profile", {})
    job_title = profile_summary.get("current_title", "Unknown")
    experience_years = profile_summary.get("years_of_experience", 0)

    # Parsing core stack listing array tags
    associated_skills = candidate_record.get("skills", [])
    extracted_skill_names = [
        item.get("name", "")
        for item in associated_skills
        if isinstance(item, dict)
    ][:3]
    skills_flattened = (
        ", ".join(extracted_skill_names) if extracted_skill_names else "N/A"
    )

    narrative_tokens = [
        f"{job_title} with {experience_years:.1f} years exp",
        f"skills: {skills_flattened}",
    ]

    if target_culture_score > 0:
        narrative_tokens.append("product company background")

    return "; ".join(narrative_tokens) + "."


def main():
    print("Initializing final scoring engine lifecycle...")

    # Load serialized matrix components out of binary store configurations
    with open(vector_pool_pkl, "rb") as f:
        precomputed_embeddings_payload = pickle.load(f)
    with open(jd_vector_pkl, "rb") as f:
        jd_vector_baseline = pickle.load(f)

    # Import generated cleaning lists to run an inclusion shield pass
    with open(honeypot_blacklist_path, "r", encoding="utf-8") as f:
        flagged_honeypots_registry = json.load(f)

    # Convert mapping keys into a set structure to guarantee O(1) runtime lookup checks
    exclusion_set = {
        item["candidate_id"] for item in flagged_honeypots_registry
    }
    print(f"[Log] Exclusion filter armed. Blacklisted IDs cached: {len(exclusion_set)}")

    processing_timer_start = datetime.now()

    # Calculate structural spatial distances against the target Job Description embedding profile
    print("Running cosine proximity calculations over candidate vector arrays...")
    jd_matrix_row = jd_vector_baseline.reshape(1, -1)
    raw_spatial_similarities = cosine_similarity(
        jd_matrix_row, precomputed_embeddings_payload["embeddings"]
    )[0]

    # Normalize similarity vector outputs safely inside a 0.0 to 1.0 boundary envelope
    normalized_semantic_scores = (raw_spatial_similarities + 1) / 2

    # --- Iterative Compilation Phase ---
    print("Beginning pipeline stream evaluation across full 100k repository...")
    evaluation_records_accumulator = []
    skipped_honeypot_counter = 0

    with open(candidate_pool_path, "r", encoding="utf-8") as stream:
        for loop_idx, text_line in enumerate(stream):
            profile_data = json.loads(text_line)
            profile_id = profile_data.get("candidate_id", "")

            # Dropping synthetic profile rows matching blacklist identifiers
            if profile_id in exclusion_set:
                skipped_honeypot_counter += 1
                continue

            semantic_distance_factor = normalized_semantic_scores[loop_idx]
            composite_score, startup_alignment_flag = evaluate_candidate_metrics(
                profile_data, semantic_distance_factor
            )

            evaluation_records_accumulator.append(
                {
                    "candidate_id": profile_id,
                    "final_score": composite_score,
                    "culture_score": startup_alignment_flag,
                    "candidate_raw_object": profile_data,
                }
            )

            if (loop_idx + 1) % 20000 == 0:
                print(f" -> Processed check-count: {loop_idx + 1}/100000...")

    print(f"[Shield] Successfully blocked {skipped_honeypot_counter} trap records.")

    # --- Sorting and Shortlist Compilation ---
    print("Running fast array sort routines to isolate highest tier candidates.")
    analysis_frame = pd.DataFrame(evaluation_records_accumulator)
    sorted_rank_frame = analysis_frame.sort_values(
        "final_score", ascending=False
    ).reset_index(drop=True)

    # Capturing top 100 best-fit profiles for submission payload packaging
    shortlisted_top_100 = sorted_rank_frame.head(100).copy()
    shortlisted_top_100["rank"] = range(1, 101)

    print("Compiling qualitative reasoning tokens for target shortlist entries...")
    shortlisted_top_100["reasoning"] = shortlisted_top_100.apply(
        lambda row: stringify_reasoning_metadata(
            row["candidate_raw_object"], row["culture_score"]
        ),
        axis=1,
    )

    # Structuring clean target columns matching hackathon format specification constraints
    final_export_dataframe = shortlisted_top_100[
        ["candidate_id", "rank", "final_score", "reasoning"]
    ]
    final_export_dataframe.columns = ["candidate_id", "rank", "score", "reasoning"]

    # Ensure output folder directory exists before saving
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exporting out final clean results matrix to designated output directory configuration
    final_export_dataframe.to_csv(output_submission_csv, index=False)

    total_execution_seconds = (
        datetime.now() - processing_timer_start
    ).total_seconds()

    print("Success: valid packaged submission dispatched to local disk")
    print(f"Target csv compiled at: {output_submission_csv}")
    print(f"Pipeline verification time: {total_execution_seconds:.2f} seconds")
    print(f"Maximum rank baseline score resolved: {shortlisted_top_100.iloc[0]['final_score']:.4f}")


if __name__ == "__main__":
    main()