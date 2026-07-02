import gzip
import json
from pathlib import Path

# hardcoding my local path for now, need to change if moving to sandbox
base_path = Path(r"C:\Users\Diksha Uike\Downloads\Redrob_Hackathon\data")


def main():
    print("Quick Data Sanity Check")

    # checking the jd file first
    jd_path = base_path / "job_description.md"
    with open(jd_path, "r", encoding="utf-8") as f:
        job_desc = f.read()
    print(f"JD loaded up fine. Total characters: {len(job_desc)}")

    # parsing the small sample json to map out fields
    sample_path = base_path / "sample_candidates.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        candidates_sample = json.load(f)
    print(f"Loaded sample candidate array size: {len(candidates_sample)}")

    # let's peek at the first profile block to confirm schema
    print("\n[INFO] Printing head of first sample profile:")
    # truncating output so terminal doesn't overflow
    raw_preview = json.dumps(candidates_sample[0], indent=2)
    print(raw_preview[:1500])

    # critical step: extracting redrob_signals to catch honeypot keys later
    first_record = candidates_sample[0]
    if "redrob_signals" in first_record:
        print("\nAvailable keys inside redrob_signals dictionary:")
        for signal_key in first_record["redrob_signals"].keys():
            print(f"  > {signal_key}")


if __name__ == "__main__":
    main()