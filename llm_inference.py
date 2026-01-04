import re
import os
import threading
from utils import get_prompt, get_mtp_prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

# Global cache for model and tokenizer to avoid reloading
_model_cache = {}
_cache_lock = threading.Lock()

HF_MODEL_MAP = {
    "llama3:3b": "meta-llama/Llama-3.2-3B-Instruct",
}

QWEN_MODEL_MAP = {
    "qwen3": "Qwen/Qwen3-4B-Instruct-2507"
}



def get_or_load_model(model_name):
    """
    Load model and tokenizer with caching for HPC efficiency.
    Thread-safe singleton pattern to avoid loading multiple times.
    """
    # Map model name to HuggingFace ID if needed
    hf_model_id = HF_MODEL_MAP.get(model_name, model_name)
    
    with _cache_lock:
        if hf_model_id not in _model_cache:
            token = os.environ.get("HF_TOKEN")
            print(f"Loading model {hf_model_id} (first time)...")
            
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                token=token,
                dtype=torch.float16,  # Use FP16 for efficiency
                device_map="auto",  # Automatically distribute model across available devices
                low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
            )
            
            _model_cache[hf_model_id] = (tokenizer, model)
            print(f"Model {hf_model_id} loaded successfully.")
        
        return _model_cache[hf_model_id]


def get_or_load_qwen_model(model_name):
    """
    Load Qwen model and tokenizer with caching for HPC efficiency.
    Thread-safe singleton pattern to avoid loading multiple times.
    """
    # Map model name to HuggingFace ID if needed
    hf_model_id = QWEN_MODEL_MAP.get(model_name, model_name)
    
    with _cache_lock:
        if hf_model_id not in _model_cache:
            token = os.environ.get("HF_TOKEN")
            print(f"Loading Qwen model {hf_model_id} (first time)...")
            
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                token=token,
                dtype="auto",
                device_map="auto"
            )
            
            _model_cache[hf_model_id] = (tokenizer, model)
            print(f"Qwen model {hf_model_id} loaded successfully.")
        
        return _model_cache[hf_model_id]


def call_llm_llama(prompt, ethnographic_excerpt, model_name, temperature):
    """
    Call Llama model using cached HuggingFace transformers.
    Uses model caching for efficiency in HPC environments.
    """
    tokenizer, model = get_or_load_model(model_name)
    
    # Format messages in chat format
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": ethnographic_excerpt}
    ]
    
    # Apply chat template and tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens (excluding input)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up input tensors to free GPU memory
    del inputs
    del outputs
    del generated_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response.strip()


def call_llm_qwen(prompt, ethnographic_excerpt, model_name, temperature):
    """
    Call Qwen model using cached HuggingFace transformers.
    Uses model caching for efficiency in HPC environments.
    """
    tokenizer, model = get_or_load_qwen_model(model_name)
    
    # Format messages using the user role (Qwen expects user messages)
    messages = [
        {"role": "user", "content": f"{prompt}\n\n{ethnographic_excerpt}"}
    ]
    
    # Apply chat template and tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens (excluding input)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # Clean up input tensors to free GPU memory
    del model_inputs
    del generated_ids
    del output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response.strip()


def cleanup_models():
    """
    Clean up cached models and free GPU memory.
    Call this at the end of processing for HPC politeness.
    """
    global _model_cache
    with _cache_lock:
        for model_id, model_or_tuple in _model_cache.items():
            print(f"Cleaning up model {model_id}...")
            # Handle both tuple (tokenizer, model) and single model cases
            if isinstance(model_or_tuple, tuple):
                tokenizer, model = model_or_tuple
                del model
                del tokenizer
            else:
                del model_or_tuple
        _model_cache.clear()
    
    # Force garbage collection and clear GPU cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU memory cleaned up.")

def call_llm(prompt, ethnographic_excerpt, model_name, temperature):
    from ollama import Client  # import here to avoid requiring ollama if unused
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'dbc56a448a8a48d39c6982bf59d5c731.qhdWodXgEf_tLKbJAEwqlgkY'}
    )

    resp = client.chat(
        model=model_name,
        messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': ethnographic_excerpt
            }
        ],
        stream=False,
        options={'temperature': temperature}
    )

    return resp.message.content.strip()

_single_token_re = re.compile(r'^\s*-?\d+\s*$')
_mtp_csv_re      = re.compile(r'^\s*-?\d+\s*(?:,\s*-?\d+\s*)*\s*$')

def is_valid_single_token(s: str) -> bool:
    return bool(_single_token_re.match(s or ""))

def is_valid_mtp(s: str, expected_len: int) -> bool:
    if not s:
        return False
    # Reject immediately if contains any newlines (do not try to parse)
    if '\n' in s or '\r' in s:
        return False
    # Strip whitespace
    normalised = s.strip()
    # Reject if contains any letters (catches text like "Games:", "TabooSmoke:", etc.)
    if re.search(r'[a-zA-Z]', normalised):
        return False
    # Reject if contains any punctuation other than commas and minus signs
    # This catches colons, periods, semicolons, etc.
    if re.search(r'[^\d\s,\-]', normalised):
        return False
    # Remove all whitespace to ensure strict comma-separated format
    normalised = re.sub(r'\s+', '', normalised)
    # Check if it matches the pattern (comma-separated integers, no spaces)
    if not re.match(r'^-?\d+(?:,-?\d+)*$', normalised):
        return False
    # Parse comma-separated values and verify each is numeric
    values = [v for v in normalised.split(',') if v]
    if len(values) != expected_len:
        return False
    # Double-check: ensure each value is actually numeric (catches edge cases)
    for v in values:
        try:
            int(v)  # Try to parse as integer
        except ValueError:
            return False
    return True


def ask_until_valid(prompt, ethnographic_excerpt, model_name, temperature, validate_fn, max_tries=100):
    last = ""
    
    for attempt in range(max_tries):
        try:
            if model_name in HF_MODEL_MAP:
                out = call_llm_llama(prompt, ethnographic_excerpt, model_name, temperature)
            elif model_name in QWEN_MODEL_MAP:
                out = call_llm_qwen(prompt, ethnographic_excerpt, model_name, temperature)
            else:
                out = call_llm(prompt, ethnographic_excerpt, model_name, temperature)
        except Exception as e:
            if attempt == 0:  # Log error on first attempt only
                print(f"ERROR in ask_until_valid (model={model_name}, attempt={attempt}): {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
            out = ""
        last = (out or "").strip()
        if validate_fn(last):
            return last
    
    # Max retries exhausted - log warning and return empty string to indicate failure
    print(f"WARNING: Validation failed after {max_tries} attempts. Last invalid response: {last[:100]}...")
    return ""  # return empty string instead of invalid data



def annotate_text(ritual_name,
                  feature_name,
                  feature_description,
                  feature_options,
                  ethnographic_excerpt,
                  model_name,
                  temperature=0.0):

    prompt = get_prompt(ritual_name, feature_name, feature_description, feature_options)
    resp = ask_until_valid(prompt, ethnographic_excerpt, model_name, temperature, is_valid_single_token)
    return resp.strip() if resp else ""

def annotate_text_ensemble(ritual_name,
                           feature_name,
                           feature_description,
                           feature_options,
                           ethnographic_excerpt,
                           model_name,
                           iterations=10,
                           temperature=0.2):

    prompt = get_prompt(ritual_name, feature_name, feature_description, feature_options)
    results = []
    for _ in range(iterations):
        resp = ask_until_valid(prompt, ethnographic_excerpt, model_name, temperature, is_valid_single_token)
        results.append(resp.strip() if resp else "")
    return results

def annotate_text_mtp(ritual_name,
                       category_name,
                       all_features,
                       ethnographic_excerpt,
                       model_name,
                       temperature):

    n_features = len(all_features["feature_name"])
    prompt = get_mtp_prompt(ritual_name, category_name, all_features)
    resp = ask_until_valid(
        prompt, ethnographic_excerpt, model_name, temperature,
        lambda s: is_valid_mtp(s, expected_len=n_features)
    )
    return resp

def annotate_text_ensemble_mtp(ritual_name,
                                category_name,
                                all_features,
                                ethnographic_excerpt,
                                model_name,
                                iterations,
                                temperature):

    n_features = len(all_features["feature_name"])
    prompt = get_mtp_prompt(ritual_name, category_name, all_features)
    results = []
    for _ in range(iterations):
        resp = ask_until_valid(
            prompt, ethnographic_excerpt, model_name, temperature,
            lambda s: is_valid_mtp(s, expected_len=n_features)
        )
        results.append(resp)
    return results
