from ollama import Client
import re
from utils import get_prompt, get_mtp_prompt
import os


def call_llm(prompt, model_name, temperature):
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': os.getenv("OLLAMA_API_KEY")}
        if os.getenv("OLLAMA_API_KEY") else None
    )

    resp = client.chat(
        model=model_name,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are an expert in anthropology, ethnographic text analysis, and text classification. '
                    'Your primary objective is to accurately annotate ethnographic texts according to specified ritual features.'
                )
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        stream=False,
        options={'temperature': temperature}
    )

    return resp.message.content.strip()


_single_token_re = re.compile(r'^\s*-?\d+\s*$')
_mtp_csv_re     = re.compile(r'^\d+(,\d+)*$')

def is_valid_single_token(s: str) -> bool:
    return bool(_single_token_re.match(s or ""))

def is_valid_mtp_csv(s: str, expected_len: int) -> bool:
    if not s or not _mtp_csv_re.match(s):
        return False
    return len(s.split(",")) == expected_len


def ask_until_valid(prompt, model_name, temperature, validate_fn, max_tries=10):
    last = ""
    for _ in range(max_tries):
        try:
            out = call_llm(prompt, model_name, temperature)
        except Exception:
            out = ""
        last = (out or "").strip()
        if validate_fn(last):
            return last
    return last  # return last even if invalid





def annotate_text(ritual_name,
                  feature_name,
                  feature_description,
                  feature_options,
                  ethnographic_excerpt,
                  model_name,
                  temperature=0.0):

    prompt = get_prompt(ritual_name, feature_name, feature_description, feature_options, ethnographic_excerpt)
    resp = ask_until_valid(prompt, model_name, temperature, is_valid_single_token)
    return resp[-1] if resp else ""

def annotate_text_ensemble(ritual_name,
                           feature_name,
                           feature_description,
                           feature_options,
                           ethnographic_excerpt,
                           model_name,
                           iterations=10,
                           temperature=0.2):

    prompt = get_prompt(ritual_name, feature_name, feature_description, feature_options, ethnographic_excerpt)
    results = []
    for _ in range(iterations):
        resp = ask_until_valid(prompt, model_name, temperature, is_valid_single_token)
        results.append(resp[-1] if resp else "")
    return results

def annotate_text_mtp(ritual_name,
                       all_features,
                       ethnographic_excerpt,
                       model_name,
                       temperature):

    n_features = len(all_features["feature_name"])
    prompt = get_mtp_prompt(ritual_name, all_features, ethnographic_excerpt)
    resp = ask_until_valid(
        prompt, model_name, temperature,
        lambda s: is_valid_mtp_csv(s, expected_len=n_features)
    )
    return resp

def annotate_text_ensemble_mtp(ritual_name,
                                all_features,
                                ethnographic_excerpt,
                                model_name,
                                iterations,
                                temperature):

    n_features = len(all_features["feature_name"])
    prompt = get_mtp_prompt(ritual_name, all_features, ethnographic_excerpt)
    results = []
    for _ in range(iterations):
        resp = ask_until_valid(
            prompt, model_name, temperature,
            lambda s: is_valid_mtp_csv(s, expected_len=n_features)
        )
        results.append(resp)
    return results