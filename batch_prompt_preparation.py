import os
import json
import pandas as pd
from utils import get_prompt  # uses your existing prompt builder

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_INSTRUCTION = (
    "You are an expert in anthropology, ethnographic text analysis, and text classification. "
    "Your objective is to accurately annotate ethnographic texts according to specified ritual features. "
    "Respond only with the required output."
)

INPUT_JSONL = "batch_tasks.jsonl"  # this is what you'll upload with purpose='batch'

def allow_ritual(rn: str, exclude: set) -> bool:
    if rn in exclude:
        return False
    try:
        return int(str(rn).replace("Ritual", "")) <= 399
    except Exception:
        return False

def main():
    rituals_codes = pd.read_csv("data/rituals_codes.csv")
    features = pd.read_csv("data/features.csv")
    exclude = set(pd.read_csv("data/exclude.csv")["exclude"].tolist())

    rituals_idx = rituals_codes.set_index("ritual_number")
    features_idx = features.set_index("feature_name")

    ritual_numbers = rituals_codes["ritual_number"].tolist()
    feature_names = features["feature_name"].tolist()

    with open(INPUT_JSONL, "w", encoding="utf-8") as f:
        for ritual_number in ritual_numbers:
            if not allow_ritual(ritual_number, exclude):
                continue

            for feature_name in feature_names:
                rrow = rituals_idx.loc[ritual_number]
                frow = features_idx.loc[feature_name]

                ritual_name = rrow["ritual_name"]
                text = rrow["text"]
                if isinstance(text, pd.Series):
                    if text.empty or text.isna().all() or text.str.len().le(3).all() or (text == "-").all():
                        print(f"Skipping {ritual_number} {feature_name} because text is missing or invalid")
                        continue
                    ethnographic_excerpt = text.iloc[0]
                else:
                    if pd.isna(text) or len(str(text)) <= 3 or text == "-":
                        print(f"Skipping {ritual_number} {feature_name} because text is missing or invalid")
                        continue
                    ethnographic_excerpt = text
                feature_description = frow["feature_description"]
                feature_options = frow["feature_options"]

                # Build your user prompt with your existing helper
                user_prompt = get_prompt(
                    ritual_name,
                    feature_name,
                    feature_description,
                    feature_options,
                    ethnographic_excerpt
                )

                task = {
                    "custom_id": f"{ritual_number}__{feature_name}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": SYSTEM_INSTRUCTION},
                            {"role": "user", "content": user_prompt}
                        ]
                    }
                }

                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Wrote batch file: {INPUT_JSONL}")

if __name__ == "__main__":
    main()