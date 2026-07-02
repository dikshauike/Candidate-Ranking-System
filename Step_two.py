import json
from datetime import datetime
from pathlib import Path

# Tracking down data and output folders locally
data_path = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\data")
output_path = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\output")

full_pool_file = data_path / "candidates.jsonl"
results_output = output_path / "honeypots_detected.json"


def check_for_traps(cand_data):
    # Sifts through a candidate payload to spot contradictory data signals.
    issues_tracked = []

    profile_info = cand_data.get("profile", {})
    skills_list = cand_data.get("skills", [])
    work_history = cand_data.get("career_history", [])
    activity_signals = cand_data.get("redrob_signals", {})

    stated_exp = profile_info.get("years_of_experience", 0)

    # Flagging unearned expert credentials (claims expertise but 0 months logged)
    suspicious_experts = 0
    for skill_obj in skills_list:
        if isinstance(skill_obj, dict):
            is_expert = skill_obj.get("proficiency", "").lower() == "expert"
            zero_duration = skill_obj.get("duration_months", -1) <= 0
            if is_expert and zero_duration:
                suspicious_experts += 1

    if suspicious_experts >= 3:
        issues_tracked.append("Expert status listed on zero duration skills")

    # Cross-referencing declared timeline against the actual career nodes
    aggregated_months = sum(
        job_node.get("duration_months", 0)
        for job_node in work_history
        if isinstance(job_node, dict)
    )
    derived_years = aggregated_months / 12.0

    # Flagging major delta between overall summary and math breakdown
    if abs(stated_exp - derived_years) > 5:
        issues_tracked.append(
            f"Experience discrepancy: claims {stated_exp}y but timeline yields {derived_years:.1f}y"
        )

    # Looking for behavior anomalies (high platform visibility but zero outreach loop)
    if isinstance(activity_signals, dict):
        total_views = activity_signals.get("search_appearance_30d", 0)
        apps_filed = activity_signals.get("applications_submitted_30d", 0)

        if total_views > 500 and apps_filed == 0:
            issues_tracked.append(
                "High passive search appearance metric with zero conversion"
            )

    return issues_tracked


def main():
    print("Running Honeypot Filtering Pipeline ")
    flagged_pool = []

    # Iterating raw lines out of unzipped jsonl dump
    with open(full_pool_file, "r", encoding="utf-8") as stream:
        for idx, text_line in enumerate(stream):
            record = json.loads(text_line)
            anomalies = check_for_traps(record)

            if anomalies:
                flagged_pool.append(
                    {
                        "candidate_id": record.get("candidate_id", ""),
                        "flags": anomalies,
                    }
                )

            # Simple heartbeat to make sure processing hasn't locked up
            if (idx + 1) % 20000 == 0:
                print(f"[Log] Records evaluated: {idx + 1}/100000...")

    # Automatically create the output directory if it doesn't exist yet
    output_path.mkdir(parents=True, exist_ok=True)

    # Writing out bad candidate IDs to exclude them from the ranking baseline later
    with open(results_output, "w", encoding="utf-8") as out_stream:
        json.dump(flagged_pool, out_stream, indent=2)

    print(f"\nProcessing complete. Found {len(flagged_pool)} flagged profiles.")
    print(f"Bad profile metadata cached to: {results_output}")


if __name__ == "__main__":
    main()