models = {
    "gpt-oss:120b": "gptoss120b",
    "deepseek-v3.1:671b": "deepseekv31671b",
    "gpt5nano": "gpt5nano",
    "llama3:3b": "llama33b",
    "qwen3": "qwen3",
    "claude45sonnet": "claude45sonnet",
    "perplexity": "perplexity"
}


def get_prompt(ritual_name, feature_name, feature_description, feature_options):
    return f"""
# Instructions
You are given an ethnographic excerpt to classify. 
- Carefully read the excerpt in full. 
- Use the feature name, its description, and the list of options to guide your decision. 
- Select the ONE option that best matches the excerpt. 
- Do not summarise, explain, or add text. 
- Return only the numeric label of the chosen option.

# Context
Ritual: {ritual_name}
Feature: {feature_name}
Definition of feature: {feature_description}
Available options (numeric labels only): {feature_options}

# Output Format
Return the numeric label of the selected option.
No words, no punctuation, no extra output.
"""


def get_mtp_prompt(ritual_name, category_name, all_features):

    n_features = len(all_features["feature_name"])

    feature_descriptions_str = ""
    for i, (name, desc, options) in enumerate(
        zip(all_features["feature_name"], all_features["feature_description"], all_features["feature_options"]), 1
    ):
        feature_descriptions_str += f"{i}. {name}: {desc}; {options}\n"
    feature_descriptions_str = feature_descriptions_str.rstrip()

    return f"""
# Instructions
You are given an ethnographic excerpt to classify across {n_features} features in the following category: {category_name}. 
- Carefully read the excerpt in full. 
- Use each feature name, its description, and the list of options to guide your decision. 
- For each feature, select the ONE option that best matches the excerpt. 
- Do not summarise, explain, or add text. 
- Return only the numeric label of the chosen option for each feature as a comma-separated list.

# Context
Ritual: {ritual_name}
Category: {category_name}
{feature_descriptions_str}

# Output Format
Return a comma-separated list of exactly {n_features} numeric values. Each value represents the chosen option for the corresponding feature in order (Example: "0,1,0,0,1,1"). 
"""



def get_perplexity_prompt(ritual_name, author, date, ethnography_title, culture, feature_name, feature_description, feature_options):
    """
    Perplexity-specific prompt that uses web search instead of ethnographic text.
    """
    return f"""Based on the ritual "{ritual_name}" from the {culture} culture (documented by {author} in {date} in "{ethnography_title}"), does this ritual have the following feature?

Feature: {feature_name}
Definition: {feature_description}
Available options: {feature_options}

Use your web search capabilities to find information about this ritual and culture. Return only the numeric label of the option that best matches. Do not include any explanation or additional text."""


def get_perplexity_mtp_prompt(ritual_name, author, date, ethnography_title, culture, category_name, all_features):
    """
    Perplexity-specific multi-task prompt that uses web search instead of ethnographic text.
    """
    n_features = len(all_features["feature_name"])

    feature_descriptions_str = ""
    for i, (name, desc, options) in enumerate(
        zip(all_features["feature_name"], all_features["feature_description"], all_features["feature_options"]), 1
    ):
        feature_descriptions_str += f"{i}. {name}: {desc}; {options}\n"
    feature_descriptions_str = feature_descriptions_str.rstrip()

    return f"""Based on the ritual "{ritual_name}" from the {culture} culture (documented by {author} in {date} in "{ethnography_title}"), classify this ritual across {n_features} features in the category "{category_name}".

{feature_descriptions_str}

Use your web search capabilities to find information about this ritual and culture. For each feature, select the ONE option that best matches. Return only a comma-separated list of exactly {n_features} numeric values (Example: "0,1,0,0,1,1"). Do not include any explanation or additional text."""
