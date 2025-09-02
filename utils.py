models = {
    "gpt-oss:20b": "gptoss20b",
    "gpt-oss:120b": "gptoss120b",
    "deepseek-v3.1:671b": "deepseekv31671b",
    "llama3.1:405b": "llama31405b",
    "GPT-5": "gpt5",
    "Claude 3.5 Sonnet": "claude35sonnet",
    "BERT": "bert",
}


def get_prompt(ritual_name, feature_name, feature_description, feature_options, ethnographic_excerpt):
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
Definition of feature:
{feature_description}
Available options (numeric labels only):
{feature_options}

Excerpt to classify:
{ethnographic_excerpt}

# Output Format
Return a single integer corresponding to the selected option.
No words, no punctuation, no extra output.
"""


def get_mtp_prompt(ritual_name, all_features, ethnographic_excerpt):
    # THIS ONLY WORKS FOR SYNCHRONY CONDITION [we will need to change the available options later]

    n_features = len(all_features["feature_name"])

    feature_descriptions_str = ""
    for i, (name, desc) in enumerate(
        zip(all_features["feature_name"], all_features["feature_description"]), 1
    ):
        feature_descriptions_str += f"{i}. {name}: {desc}\n"
    feature_descriptions_str = feature_descriptions_str.rstrip()

    return f"""
# Instructions
You are given an ethnographic excerpt to classify across {n_features} features. 
- Carefully read the excerpt in full. 
- Use each feature name, its description, and the list of options to guide your decision. 
- For each feature, select the ONE option that best matches the excerpt. 
- Do not summarise, explain, or add text. 
- Return only the numeric label of the chosen option for each feature separated by a comma.

# Context
Ritual: {ritual_name}
{feature_descriptions_str}
Available options (numeric labels only): 0=absent, 1=present.
Excerpt to classify:
{ethnographic_excerpt}

# Output Format
Return a string with the numeric label of the chosen option for each feature separated by a comma and no spaces (Example: "0,1,0,0,1,0"). 
"""