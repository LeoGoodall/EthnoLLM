# run_batch_and_save.py
import os
import time
import json
import csv
from pathlib import Path
from openai import OpenAI

INPUT_JSONL = os.getenv("BATCH_INPUT", "batch_tasks.jsonl")
OUTPUT_JSONL = os.getenv("BATCH_OUTPUT", "batch_results.jsonl")
ERROR_JSONL  = os.getenv("BATCH_ERROR",  "batch_errors.jsonl")
PARSED_CSV   = os.getenv("BATCH_PARSED_CSV", "batch_results_parsed.csv")
POLL_SECONDS = int(os.getenv("BATCH_POLL_SECONDS", "10"))

def main():
    client = OpenAI()  # requires OPENAI_API_KEY in env

    # Upload input file with purpose='batch'
    with open(INPUT_JSONL, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"Uploaded: {uploaded.id}")

    # Create batch for Chat Completions
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        # metadata is optional; omit or add as needed
        # metadata={"description": "classification job"},
    )
    print(f"Batch created: {batch.id} status={batch.status}")

    # Poll until terminal state
    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"[{time.strftime('%H:%M:%S')}] {batch.id} status={batch.status} "
              f"counts={getattr(batch, 'request_counts', None)}")
        if batch.status in terminal:
            break
        time.sleep(POLL_SECONDS)

    # Save errors (if any)
    if getattr(batch, "error_file_id", None):
        err_bytes = client.files.content(batch.error_file_id).read()
        Path(ERROR_JSONL).write_bytes(err_bytes)
        print(f"Saved errors -> {ERROR_JSONL}")

    if batch.status != "completed":
        raise SystemExit(f"Batch ended with status={batch.status}")

    # Download output JSONL
    out_bytes = client.files.content(batch.output_file_id).read()
    Path(OUTPUT_JSONL).write_bytes(out_bytes)
    print(f"Saved output -> {OUTPUT_JSONL}")

    # Parse JSONL to CSV: custom_id, content
    rows = []
    for line in out_bytes.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id") or obj.get("id")
        # Body mirrors Chat Completions; extract assistant text
        body = (obj.get("response") or {}).get("body") or {}
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps(body, ensure_ascii=False)
        rows.append({"custom_id": cid, "content": content})

    with open(PARSED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["custom_id", "content"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved parsed CSV -> {PARSED_CSV}")

if __name__ == "__main__":
    main()