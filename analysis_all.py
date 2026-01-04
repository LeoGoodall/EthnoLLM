import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from utils import models
import time
import threading
from pathlib import Path

force_cpu = os.environ.get("FORCE_CPU", "").lower() == "true"
if force_cpu:
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
    })

os.environ.update({
    "TOKENIZERS_PARALLELISM": "false", 
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
})



def main(model, condition):

    print(f"BASELINE: Processing {model} with {condition} condition...")
    model_name = models[model]

    # Load data
    rituals_codes = pd.read_csv("data/rituals_codes.csv")
    features = pd.read_csv("data/features_all.csv")
    exclude = set(pd.read_csv("data/exclude.csv")["exclude"].tolist())

    # Pre-index for fast lookups (avoid pandas filtering inside workers)
    rituals_idx = rituals_codes.set_index("ritual_number")
    features_idx = features.set_index("feature_variable")

    # Create results dataframe with human and LLM columns - build all columns at once to avoid fragmentation
    feature_vars = features["feature_variable"].values.tolist()
    
    # Create dictionary of all columns
    columns_dict = {"ritual_number": rituals_codes["ritual_number"]}
    for feature_var in feature_vars:
        # Copy over the human-coded column
        columns_dict[feature_var] = rituals_codes[feature_var]
        # Use object dtype (string) for LLM columns to accept string values
        columns_dict[f"{feature_var}_llm"] = pd.Series([""] * len(rituals_codes), dtype=object)
        columns_dict[f"{feature_var}_llm_certainty"] = pd.Series([0.0] * len(rituals_codes), dtype=float)
    
    # Create DataFrame with all columns at once
    results = pd.DataFrame(columns_dict)
    
    # Fast row index for assignment (like mtp_main)
    row_index_by_ritual = {rn: i for i, rn in enumerate(results["ritual_number"])}

    # Determine output file path
    out_file = f"all/results_{model_name}{f'_{condition}' if condition != 'single' else ''}.csv"
    
    # Load existing CSV if it exists to resume
    processed_combinations = set()
    if Path(out_file).exists():
        print(f"Loading existing results from {out_file} to resume...")
        existing_results = pd.read_csv(out_file)
        # Ensure all required columns exist (in case previous run was interrupted)
        for feature_var in feature_vars:
            if feature_var not in existing_results.columns:
                existing_results[feature_var] = results[feature_var]
            if f"{feature_var}_llm" not in existing_results.columns:
                existing_results[f"{feature_var}_llm"] = ""
                existing_results[f"{feature_var}_llm"] = existing_results[f"{feature_var}_llm"].astype(object)
            if f"{feature_var}_llm_certainty" not in existing_results.columns:
                existing_results[f"{feature_var}_llm_certainty"] = 0.0
        # Ensure proper dtypes for LLM columns
        for feature_var in feature_vars:
            if f"{feature_var}_llm" in existing_results.columns:
                existing_results[f"{feature_var}_llm"] = existing_results[f"{feature_var}_llm"].astype(object)
            if f"{feature_var}_llm_certainty" in existing_results.columns:
                existing_results[f"{feature_var}_llm_certainty"] = existing_results[f"{feature_var}_llm_certainty"].astype(float)
        # Track which (ritual_number, feature_variable) combinations are already processed
        for _, row in existing_results.iterrows():
            ritual_number = row["ritual_number"]
            for feature_var in feature_vars:
                llm_col = f"{feature_var}_llm"
                if llm_col in existing_results.columns and pd.notna(row.get(llm_col)) and str(row[llm_col]).strip() != "":
                    processed_combinations.add((ritual_number, feature_var))
        # Update results with existing data
        results = existing_results.copy()
        # Rebuild row_index_by_ritual after loading existing results
        row_index_by_ritual = {rn: i for i, rn in enumerate(results["ritual_number"])}
        print(f"Found {len(processed_combinations)} already processed combinations")
    
    # Thread lock for safe CSV writing
    csv_lock = threading.Lock()
    
    def write_incremental_csv():
        """Write current results to CSV in a thread-safe manner"""
        with csv_lock:
            results.to_csv(out_file, index=False)

    def allow_ritual(rn: str) -> bool:
        return rn not in exclude

    def process_ritual_feature(args):
        # Import inside worker keeps this file self-contained
        from llm_inference import annotate_text, annotate_text_ensemble

        ritual_number, feature_variable = args
        if not allow_ritual(ritual_number):
            return None
        
        # Skip if already processed
        if (ritual_number, feature_variable) in processed_combinations:
            return None

        rrow = rituals_idx.loc[ritual_number]
        frow = features_idx.loc[feature_variable]

        ritual_name = rrow["ritual_name"]
        feature_name = feature_variable  # Use feature_variable as feature_name
        feature_description = frow["feature_description"]
        feature_options = frow["feature_options"]
        text = rrow["text"]
        if isinstance(text, pd.Series):
            if text.empty or text.isna().all() or text.str.len().le(3).all() or (text == "-").all():
                print(f"Skipping {ritual_number} {feature_variable} because text is missing or invalid")
                return None
            ethnographic_excerpt = text.iloc[0]
        else:
            if pd.isna(text) or len(str(text)) <= 3 or text == "-":
                print(f"Skipping {ritual_number} {feature_variable} because text is missing or invalid")
                return None
            ethnographic_excerpt = text

        # Call the model with retries on 502 errors
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if condition == "ensemble":
                    raw = annotate_text_ensemble(
                        ritual_name,
                        feature_name, 
                        feature_description,
                        feature_options,
                        ethnographic_excerpt,
                        model_name=model,
                        iterations=10,
                        temperature=0.2
                    )
                elif condition == "single":
                    raw = annotate_text(
                        ritual_name,
                        feature_name, 
                        feature_description,
                        feature_options,
                        ethnographic_excerpt,
                        model_name=model
                    )
                else:
                    raise ValueError(f"Invalid condition: {condition}")
                break
            except Exception as e:
                if "502" in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raw = [] 
                break

        feature_var = feature_variable
        column_name = f"{feature_var}_llm"

        if condition == "single":
            # Extract full numeric value (model returns a single number, possibly multi-digit)
            s = str(raw).strip()
            # Extract numeric value (handles cases like "5", "10", "-2", etc.)
            import re
            match = re.match(r'-?\d+', s) if s else None
            value = match.group(0) if match else ""
            certainty = 100 if value else 0
            return (ritual_number, column_name, value, certainty)

        elif condition == "ensemble":
            # raw: ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
            import statistics
            if not raw:  # Handle empty list case
                value = ""
                certainty = 0
            else:
                value = statistics.mode(raw)
                certainty = raw.count(value) / len(raw) * 100
            return (ritual_number, column_name, value, certainty)

    # Build combinations
    ritual_numbers = rituals_codes["ritual_number"].tolist()
    feature_names = features["feature_variable"].tolist()
    total_combinations = len(ritual_numbers) * len(feature_names)

    def combo_iter():
        for r in ritual_numbers:
            for f in feature_names:
                yield (r, f)

    print(f"Processing {total_combinations} ritual-feature combinations...")

    # Threads suit network/API calls; keep concurrency modest by default
    default_workers = 8
    max_workers = int(os.environ.get("OLLAMA_MAX_WORKERS", default_workers))
    max_workers = max(1, max_workers)

    outputs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(ex.map(process_ritual_feature, combo_iter()),
                        total=total_combinations, smoothing=0.1):
            if res is not None:
                ritual_number, column_name, value, certainty = res
                # Mark as processed
                feature_var = column_name.replace("_llm", "")
                processed_combinations.add((ritual_number, feature_var))
                
                # Update results immediately using .at for efficient, non-fragmenting assignment
                row_ix = row_index_by_ritual[ritual_number]
                results.at[row_ix, column_name] = str(value) if value else ""
                results.at[row_ix, column_name + "_certainty"] = float(certainty)
                
                # Write incrementally
                write_incremental_csv()
                outputs.append(res)

    print("Processing complete! Final save...")
    write_incremental_csv()
    print(f"Wrote {out_file}")



def mtp_main(model, condition):
    print(f"MTP: Processing {model} with {condition} condition...")
    model_name = models[model]

    from llm_inference import annotate_text_mtp, annotate_text_ensemble_mtp

    # Load data
    rituals_codes = pd.read_csv("data/rituals_codes.csv")
    features = pd.read_csv("data/features_all.csv")
    exclude = set(pd.read_csv("data/exclude.csv")["exclude"].tolist())

    # Pre-index
    rituals_idx = rituals_codes.set_index("ritual_number")

    # Results frame - build all columns at once to avoid fragmentation
    feature_vars = features["feature_variable"].tolist()
    
    # Create dictionary of all columns
    columns_dict = {"ritual_number": rituals_codes["ritual_number"]}
    for feature_var in feature_vars:
        columns_dict[feature_var] = rituals_codes[feature_var]
        # Use object dtype (string) for LLM columns to accept string values
        columns_dict[f"{feature_var}_llm"] = pd.Series([""] * len(rituals_codes), dtype=object)
        columns_dict[f"{feature_var}_llm_certainty"] = pd.Series([0.0] * len(rituals_codes), dtype=float)
    
    # Create DataFrame with all columns at once
    results = pd.DataFrame(columns_dict)

    # Group features by category, excluding "duration"
    categories = features.groupby("feature_category")
    categories = {name: group for name, group in categories if name != "duration"} # exclude duration since it only has one ritual feature.
    print(f"Found {len(categories)} feature categories (excluding duration): {list(categories.keys())}")

    # Fast row index for assignment
    row_index_by_ritual = {rn: i for i, rn in enumerate(results["ritual_number"])}

    # Determine output file path
    out_name = f"all/results_{model_name}{f'_{condition}' if condition != 'single' else ''}_mtp.csv"
    
    # Load existing CSV if it exists to resume
    processed_combinations = set()
    if Path(out_name).exists():
        print(f"Loading existing results from {out_name} to resume...")
        existing_results = pd.read_csv(out_name)
        # Ensure all required columns exist (in case previous run was interrupted)
        for feature_var in feature_vars:
            if feature_var not in existing_results.columns:
                existing_results[feature_var] = results[feature_var]
            if f"{feature_var}_llm" not in existing_results.columns:
                existing_results[f"{feature_var}_llm"] = ""
                existing_results[f"{feature_var}_llm"] = existing_results[f"{feature_var}_llm"].astype(object)
            if f"{feature_var}_llm_certainty" not in existing_results.columns:
                existing_results[f"{feature_var}_llm_certainty"] = 0.0
        # Ensure proper dtypes for LLM columns
        for feature_var in feature_vars:
            if f"{feature_var}_llm" in existing_results.columns:
                existing_results[f"{feature_var}_llm"] = existing_results[f"{feature_var}_llm"].astype(object)
            if f"{feature_var}_llm_certainty" in existing_results.columns:
                existing_results[f"{feature_var}_llm_certainty"] = existing_results[f"{feature_var}_llm_certainty"].astype(float)
        # Track which (ritual_number, category_name) combinations are already processed
        for _, row in existing_results.iterrows():
            ritual_number = row["ritual_number"]
            for category_name, category_features in categories.items():
                # Use the actual feature_category value (same as in combo_iter)
                actual_category = category_features.iloc[0].feature_category
                # Check if all features in this category are already processed
                all_processed = True
                for _, feat_row in category_features.iterrows():
                    feat_var = feat_row["feature_variable"]
                    llm_col = f"{feat_var}_llm"
                    if llm_col not in existing_results.columns or pd.isna(row.get(llm_col)) or str(row.get(llm_col, "")).strip() == "":
                        all_processed = False
                        break
                if all_processed:
                    processed_combinations.add((ritual_number, actual_category))
        # Update results with existing data
        results = existing_results.copy()
        # Rebuild row_index_by_ritual after loading existing results
        row_index_by_ritual = {rn: i for i, rn in enumerate(results["ritual_number"])}
        print(f"Found {len(processed_combinations)} already processed combinations")
    
    # Thread lock for safe CSV writing
    csv_lock = threading.Lock()
    
    def write_incremental_csv():
        """Write current results to CSV in a thread-safe manner"""
        with csv_lock:
            results.to_csv(out_name, index=False)

    def allow_ritual(rn: str) -> bool:
        return rn not in exclude

    def parse_llm_csv(raw: str, expected_len: int) -> list[str]:
        if raw is None:
            vals = []
        elif isinstance(raw, list):
            vals = [str(x).strip() for x in raw]
        else:
            s = str(raw).strip()
            if s:
                # Parse comma-separated values
                vals = [v.strip() for v in s.split(',') if v.strip()]
            else:
                vals = []
        if len(vals) < expected_len:
            vals = vals + [""] * (expected_len - len(vals))
        elif len(vals) > expected_len:
            vals = vals[:expected_len]
        return vals

    def process_ritual_category(args):
        ritual_number, category_name, category_features = args
        if not allow_ritual(ritual_number):
            return None
        
        # Skip if already processed
        if (ritual_number, category_name) in processed_combinations:
            return None

        rrow = rituals_idx.loc[ritual_number]
        ritual_name = rrow["ritual_name"]
        text = rrow["text"]

        # validate text
        if isinstance(text, pd.Series):
            if text.empty or text.isna().all() or text.str.len().le(3).all() or (text == "-").all():
                print(f"Skipping {ritual_number} because text is missing or invalid")
                return None
            ethnographic_excerpt = text.iloc[0]
        else:
            if pd.isna(text) or len(str(text)) <= 3 or text == "-":
                print(f"Skipping {ritual_number} because text is missing or invalid")
                return None
            ethnographic_excerpt = text

        # Create ordered features for this category
        ordered_features = [
            {
                "feature_variable": row.feature_variable,
                "feature_name": row.feature_variable,
                "feature_description": row.feature_description,
                "feature_options": row.feature_options
            }
            for _, row in category_features.iterrows()
        ]

        # mtp payload expected by get_mtp_prompt
        category_features_payload = {
            "feature_name": [f["feature_variable"] for f in ordered_features],
            "feature_description": [f["feature_description"] for f in ordered_features],
            "feature_options": [f["feature_options"] for f in ordered_features]
        }

        max_retries = 5
        retry_delay = 1

        if condition == "single":
            raw = ""
            for attempt in range(max_retries):
                try:
                    raw = annotate_text_mtp(
                        ritual_name,
                        category_name,
                        category_features_payload,
                        ethnographic_excerpt,
                        model_name=model,
                        temperature=0.0
                    )
                    break
                except Exception as e:
                    if "502" in str(e) and attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raw = ""
                    break

            parsed = parse_llm_csv(raw, expected_len=len(ordered_features))
            assignments = []
            for feat, val in zip(ordered_features, parsed):
                llm_col = f"{feat["feature_variable"]}_llm"
                # Use full value (now supports multi-digit numbers from comma-separated input)
                assignments.append((llm_col, (val if val else ""), 100 if val else 0))
            return ritual_number, category_name, assignments

        elif condition == "ensemble":
            mtp_runs = []
            for attempt in range(max_retries):
                try:
                    mtp_runs = annotate_text_ensemble_mtp(
                        ritual_name,
                        category_name,
                        category_features_payload,
                        ethnographic_excerpt,
                        model_name=model,
                        iterations=10,
                        temperature=0.2
                    )
                    break
                except Exception as e:
                    if "502" in str(e) and attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    mtp_runs = []
                    break

            runs_parsed = []
            if isinstance(mtp_runs, list) and mtp_runs and isinstance(mtp_runs[0], str):
                for s in mtp_runs:
                    runs_parsed.append(parse_llm_csv(s, expected_len=len(ordered_features)))
            elif isinstance(mtp_runs, list) and mtp_runs and isinstance(mtp_runs[0], list):
                for lst in mtp_runs:
                    # Join with commas to match comma-separated format
                    s = ",".join(str(x).strip() for x in lst)
                    runs_parsed.append(parse_llm_csv(s, expected_len=len(ordered_features)))
            else:
                runs_parsed = [[""] * len(ordered_features)]

            num_features = len(ordered_features)
            votes_per_feature = [[] for _ in range(num_features)]
            for run in runs_parsed:
                if len(run) != num_features:
                    run = parse_llm_csv(run, expected_len=num_features)
                for i in range(num_features):
                    # Use full value (now supports multi-digit numbers from comma-separated input)
                    votes_per_feature[i].append(str(run[i]) if run[i] else "")

            import statistics
            assignments = []
            for i, feat in enumerate(ordered_features):
                votes = [v for v in votes_per_feature[i] if v != ""]
                if not votes:
                    value = ""
                    certainty = 0
                else:
                    try:
                        value = statistics.mode(votes)
                    except statistics.StatisticsError:
                        value = votes[0]
                    certainty = votes.count(value) / len(votes_per_feature[i]) * 100 if votes_per_feature[i] else 0
                llm_col = f"{feat["feature_variable"]}_llm"
                assignments.append((llm_col, value, certainty))
            return ritual_number, category_name, assignments

        else:
            return None

    # Build combinations of rituals and categories
    ritual_numbers = rituals_codes["ritual_number"].tolist()
    allowed = [rn for rn in ritual_numbers if allow_ritual(rn)]
    
    def combo_iter():
        for rn in allowed:
            for category_name, category_features in categories.items():
                # Use the actual feature_category value from the first row of the group
                actual_category = category_features.iloc[0].feature_category
                yield (rn, actual_category, category_features)
    
    total_combinations = len(allowed) * len(categories)
    print(f"Processing {total_combinations} ritual-category combinations...")

    default_workers = 16
    max_workers = int(os.environ.get("OLLAMA_MAX_WORKERS", default_workers))
    max_workers = max(1, max_workers)

    outputs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(ex.map(process_ritual_category, combo_iter()),
                        total=total_combinations, smoothing=0.1):
            if res is not None:
                ritual_number, category_name, assignments = res
                # Mark as processed
                processed_combinations.add((ritual_number, category_name))
                
                # Update results immediately using .at for efficient, non-fragmenting assignment
                row_ix = row_index_by_ritual[ritual_number]
                for col, value, certainty in assignments:
                    results.at[row_ix, col] = str(value) if value else ""
                    results.at[row_ix, col + "_certainty"] = float(certainty)
                
                # Write incrementally
                write_incremental_csv()
                outputs.append(res)

    print("Processing complete! Final save...")
    write_incremental_csv()
    print(f"Wrote {out_name}")
    return results


if __name__ == "__main__":
    for model in ["gpt-oss:120b"]:
        for condition in ["ensemble"]:
            main(model, condition)
