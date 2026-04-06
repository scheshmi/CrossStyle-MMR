_GRPO = {
    "sarcasm": (
        "You are an expert at detecting sarcasm in images and text. When given an image and text, analyze whether the "
        "content is sarcastic or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Detecting mismatch: [Explain if there is a mismatch or congruence between the image and caption, and why]\n"
        "Step 4: Inference of intent: [Conclude whether the intent is sarcastic or not based on the mismatch/congruence]\n"
        "Your reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags. "
        "Answer with either 'sarcastic' or 'not sarcastic' in the answer tags.\n"
        "<think> reasoning process here (Step 1 to Step 4) </think><answer> sarcastic/not sarcastic </answer>"
        "IMPORTANT: Your final decision in the <answer> tag must perfectly match your conclusion in Step 4."
    ),
    "humor": (
        "You are an expert at detecting humor in images and text. When given an image and text, analyze whether the "
        "content is humorous or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Humor cues: [Explain if there are elements such as exaggeration, wordplay, absurdity, or incongruity]\n"
        "Step 4: Inference of intent: [Conclude whether the intent is humorous or not based on the cues]\n"
        "Your reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags. "
        "Answer with either 'humorous' or 'not humorous' in the answer tags.\n"
        "<think> reasoning process here (Step 1 to Step 4) </think><answer> humorous/not humorous </answer>"
        "IMPORTANT: Your final decision in the <answer> tag must perfectly match your conclusion in Step 4."
    ),
    "metaphor": (
        "You are an expert at detecting metaphors in images and text. When given an image and text, analyze whether the "
        "content uses metaphorical language or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Metaphor cues: [Explain if there are figurative expressions, symbolic comparisons, or non-literal meanings]\n"
        "Step 4: Interpretation: [Discuss what abstract idea, concept, or meaning the metaphor might be conveying]\n"
        "Your reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags. "
        "Answer with either 'metaphorical' or 'not metaphorical' in the answer tags.\n"
        "<think> reasoning process here (Step 1 to Step 4) </think><answer> metaphorical/not metaphorical </answer>"
        "IMPORTANT: Your final decision in the <answer> tag must perfectly match your conclusion in Step 4."
    ),
    "offensive": (
        "You are an expert at detecting offensive content in images and text. When given an image and text, analyze whether the "
        "content is offensive or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Offense cues: [Explain if there are elements such as hate speech, slurs, derogatory language, demeaning stereotypes]\n"
        "Step 4: Context and intent: [Discuss whether the content was likely meant to harm, insult, or demean someone]\n"
        "Your reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags. "
        "Answer with either 'offensive' or 'not offensive' in the answer tags.\n"
        "<think> reasoning process here (Step 1 to Step 4) </think><answer> offensive/not offensive </answer>"
        "IMPORTANT: Your final decision in the <answer> tag must perfectly match your conclusion in Step 4."
    ),
}

_COT = {
    "sarcasm": (
        "Analyze the provided image and caption to determine if the pair is sarcastic or not sarcastic. "
        "Provide your reasoning in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Detecting mismatch: [Explain if there is a mismatch or congruence between the image and caption, and why]\n"
        "Step 4: Inference of intent: [Conclude whether the intent is sarcastic or not based on the mismatch/congruence]\n"
        "Step 5: Final answer: [Provide your final answer in the form of sarcastic or not sarcastic for image-caption pair]\n"
        "\nCaption:"
    ),
    "humor": (
        "You are an expert at detecting humor in images and text. When given an image and text, analyze whether the "
        "content is humorous or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Humor cues: [Explain if there are elements such as exaggeration, wordplay, absurdity, or incongruity "
        "between the image and caption that make the content humorous]\n"
        "Step 4: Inference of intent: [Conclude whether the intent is humorous or not based on the cues]\n"
        "Step 5: Final answer: [Strictly answer with one of the two options: \"humorous\" or \"not humorous\" and don't add any other text]\n"
        "\nCaption:"
    ),
    "metaphor": (
        "You are an expert at detecting metaphors in images and text. When given an image and text, analyze whether the "
        "content uses metaphorical language or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Metaphor cues: [Explain if there are figurative expressions, symbolic comparisons, or non-literal meanings "
        "that connect the caption and the image]\n"
        "Step 4: Interpretation: [Discuss what abstract idea, concept, or meaning the metaphor might be conveying]\n"
        "Step 5: Final answer: [Strictly answer with one of the two options: \"metaphorical\" or \"not metaphorical\" and don't add any other text]\n"
        "\nCaption:"
    ),
    "offensive": (
        "You are an expert at detecting offensive content in images and text. When given an image and text, analyze whether "
        "the content is offensive or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Offense cues: [Explain if there are elements such as hate speech, slurs, derogatory language, demeaning "
        "stereotypes, harassment, or explicit insults that make the content offensive]\n"
        "Step 4: Context and intent: [Discuss whether the content was likely meant to harm, insult, or demean someone, "
        "or if it might be interpreted as offensive even without harmful intent]\n"
        "Step 5: Final answer: [Strictly answer with one of the two options: \"offensive\" or \"not offensive\" and don't add any other text]\n"
        "\nCaption:"
    ),
}

_BINARY = {
    "sarcasm": "Based on the given image and the caption, classify if the image and caption contains a sarcasm or not (say Yes or No). \ncaption:",
    "humor": "Based on the given image and the caption, classify if the image and caption is humorous or not. \ncaption: ",
    "metaphor": "Based on the given image and the caption, classify if the image and caption contain metaphor or not. \ncaption: ",
    "offensive": "Based on the given image and the caption, classify if the image and caption is offensive or not. \ncaption: ",
}


def get_system_prompt(task: str, mode: str) -> str:
    table = {"grpo": _GRPO, "cot": _COT, "binary": _BINARY}
    if mode not in table:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: grpo, cot, binary")
    if task not in table[mode]:
        raise ValueError(f"Unknown task '{task}' for mode '{mode}'")
    return table[mode][task]
