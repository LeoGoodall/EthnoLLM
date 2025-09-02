import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from utils import models
import time
# CPU-only / threading hygiene
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "TOKENIZERS_PARALLELISM": "false", 
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
})


def main(model, condition):

    print(f"Processing {model} with {condition} condition...")
    model_name = models[model]

    # Load data
    rituals_codes = pd.read_csv("data/rituals_codes.csv")
    features = pd.read_csv("data/features.csv")
    exclude = set(pd.read_csv("data/exclude.csv")["exclude"].tolist())

    # Pre-index for fast lookups (avoid pandas filtering inside workers)
    rituals_idx = rituals_codes.set_index("ritual_number")
    features_idx = features.set_index("feature_name")

    # Create results dataframe with human and LLM columns
    results = pd.DataFrame()
    results["ritual_number"] = rituals_codes["ritual_number"]

    feature_vars = features["feature_variable"].values.tolist()
    for feature_var in feature_vars:
        # Copy over the human-coded column
        results[feature_var] = rituals_codes[feature_var]

        parts = feature_var.split("_")
        base_name = "_".join(parts[:-1])
        results[f"{base_name}_llm"] = None

    def allow_ritual(rn: str) -> bool:
        if rn in exclude:
            return False
        try:
            return int(str(rn).replace("Ritual", "")) <= 399
        except Exception:
            return False

    def process_ritual_feature(args):
        # Import inside worker keeps this file self-contained
        from llm_inference import annotate_text, annotate_text_ensemble

        ritual_number, feature_name = args
        if not allow_ritual(ritual_number):
            return None

        rrow = rituals_idx.loc[ritual_number]
        frow = features_idx.loc[feature_name]

        ritual_name = rrow["ritual_name"]
        feature_description = frow["feature_description"]
        feature_options = frow["feature_options"]
        text = rrow["text"]
        if isinstance(text, pd.Series):
            if text.empty or text.isna().all() or text.str.len().le(3).all() or (text == "-").all():
                print(f"Skipping {ritual_number} {feature_name} because text is missing or invalid")
                return None
            ethnographic_excerpt = text.iloc[0]
        else:
            if pd.isna(text) or len(str(text)) <= 3 or text == "-":
                print(f"Skipping {ritual_number} {feature_name} because text is missing or invalid")
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
                raw = []  # Initialize as empty list instead of empty string
                break

        parts = frow["feature_variable"].split("_")
        feature_var = "_".join(parts[:-1])
        column_name = f"{feature_var}_llm"

        if condition == "single":
            # Force to first character (model should return a single number)
            s = str(raw).strip()
            value = s[0] if s else ""
            certainty = 100 if s else 0
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
    feature_names = features["feature_name"].tolist()
    total_combinations = len(ritual_numbers) * len(feature_names)

    def combo_iter():
        for r in ritual_numbers:
            for f in feature_names:
                yield (r, f)

    print(f"Processing {total_combinations} ritual-feature combinations...")

    # Threads suit network/API calls; keep concurrency modest by default
    default_workers = 16
    max_workers = int(os.environ.get("OLLAMA_MAX_WORKERS", default_workers))
    max_workers = max(1, max_workers)

    outputs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(ex.map(process_ritual_feature, combo_iter()),
                        total=total_combinations, smoothing=0.1):
            if res is not None:
                outputs.append(res)

    print("Processing complete! Consolidating...")

    # Assign results in a single pass
    for ritual_number, column_name, value, certainty in outputs:
        results.loc[results["ritual_number"] == ritual_number, column_name] = value
        results.loc[results["ritual_number"] == ritual_number, column_name + "_certainty"] = certainty

    # Save results to CSV
    results.to_csv(f"results_{model_name}{f"_{condition}" if condition != "single" else ""}.csv", index=False)
    print("Wrote results.csv")


def mtp_main(model, condition):
    print(f"Processing {model} with {condition} condition...")
    model_name = models[model]

    from llm_inference import annotate_text_mtp, annotate_text_ensemble_mtp

    # Load data
    rituals_codes = pd.read_csv("data/rituals_codes.csv")
    features = pd.read_csv("data/features.csv")
    exclude = set(pd.read_csv("data/exclude.csv")["exclude"].tolist())

    # Pre-index
    rituals_idx = rituals_codes.set_index("ritual_number")

    # Results frame
    results = pd.DataFrame()
    results["ritual_number"] = rituals_codes["ritual_number"]

    feature_vars = features["feature_variable"].tolist()
    for feature_var in feature_vars:
        results[feature_var] = rituals_codes[feature_var]
        base_name = "_".join(feature_var.split("_")[:-1])
        results[f"{base_name}_llm"] = None

    # Stable feature order, carry feature_variable for column mapping
    ordered_features = [
        {
            "feature_variable": row.feature_variable,
            "feature_name": row.feature_name,
            "feature_description": row.feature_description,
        }
        for _, row in features.iterrows()
    ]

    # mtp payload expected by get_mtp_prompt
    all_features_payload = {
        "feature_name": [f["feature_name"] for f in ordered_features],
        "feature_description": [f["feature_description"] for f in ordered_features],
    }

    # Fast row index for assignment
    row_index_by_ritual = {rn: i for i, rn in enumerate(results["ritual_number"])}

    def allow_ritual(rn: str) -> bool:
        if rn in exclude:
            return False
        try:
            return int(str(rn).replace("Ritual", "")) <= 399
        except Exception:
            return False

    def parse_llm_csv(raw: str, expected_len: int) -> list[str]:
        if raw is None:
            vals = []
        elif isinstance(raw, list):
            vals = [str(x) for x in raw]
        else:
            s = str(raw).strip().strip(",")
            vals = s.split(",") if s else []
            vals = [v.strip() for v in vals]
        if len(vals) < expected_len:
            vals = vals + [""] * (expected_len - len(vals))
        elif len(vals) > expected_len:
            vals = vals[:expected_len]
        return vals

    def process_ritual(ritual_number):
        if not allow_ritual(ritual_number):
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

        max_retries = 5
        retry_delay = 1

        if condition == "single":
            raw = ""
            for attempt in range(max_retries):
                try:
                    raw = annotate_text_mtp(
                        ritual_name,
                        all_features_payload,
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
                base = "_".join(feat["feature_variable"].split("_")[:-1])
                llm_col = f"{base}_llm"
                assignments.append((llm_col, (val[:1] if val else ""), 100 if val else 0))
            return ritual_number, assignments

        elif condition == "ensemble":
            mtp_runs = []
            for attempt in range(max_retries):
                try:
                    mtp_runs = annotate_text_ensemble_mtp(
                        ritual_name,
                        all_features_payload,
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
                    votes_per_feature[i].append(str(run[i])[:1] if run[i] else "")

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
                base = "_".join(feat["feature_variable"].split("_")[:-1])
                llm_col = f"{base}_llm"
                assignments.append((llm_col, value, certainty))
            return ritual_number, assignments

        else:
            return None

    # plan work set and pool size
    ritual_numbers = rituals_codes["ritual_number"].tolist()
    allowed = [rn for rn in ritual_numbers if allow_ritual(rn)]
    print(f"Processing {len(allowed)} rituals in mtp mode...")

    default_workers = 16
    max_workers = int(os.environ.get("OLLAMA_MAX_WORKERS", default_workers))
    max_workers = max(1, max_workers)

    outputs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(ex.map(process_ritual, allowed), total=len(allowed), smoothing=0.1):
            if res is not None:
                outputs.append(res)

    print("Processing complete! Consolidating...")

    for ritual_number, assignments in outputs:
        row_ix = row_index_by_ritual[ritual_number]
        for col, value, certainty in assignments:
            results.at[row_ix, col] = value
            results.at[row_ix, col + "_certainty"] = certainty

    out_name = f"results_{model_name}{f'_{condition}' if condition != 'single' else ''}_mtp.csv"
    results.to_csv(out_name, index=False)
    print(f"Wrote {out_name}")

    return results

if __name__ == "__main__":
    for model in ["gpt-oss:20b", "gpt-oss:120b", "deepseek-v3.1:671b"]:
        for condition in ["ensemble"]:
            # main(model, condition)
            mtp_main(model, condition)

