import re
from datasets import load_dataset, interleave_datasets

def sentence_aware_truncate(text, word_limit=150):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    words = text.split()
    if len(words) <= word_limit:
        return text
    
    truncated_words = words[:word_limit]
    remaining_text = " ".join(words[word_limit:])
    
    match = re.search(r'[.!?]', remaining_text)
    if match:
        end_idx = match.end()
        final_text = " ".join(truncated_words) + " " + remaining_text[:end_idx]
    else:
        final_text = " ".join(truncated_words) + "."
        
    return final_text

def flatten_hc3(examples):
    texts, labels, sources = [], [], []
    for human_answers, ai_answers in zip(examples['human_answers'], examples['chatgpt_answers']):
        for h_text in human_answers:
            clean_text = sentence_aware_truncate(h_text)
            if clean_text:
                texts.append(clean_text)
                labels.append(0)
                sources.append("HC3")
        for a_text in ai_answers:
            clean_text = sentence_aware_truncate(a_text)
            if clean_text:
                texts.append(clean_text)
                labels.append(1)
                sources.append("HC3")
    return {"text": texts, "label": labels, "source": sources}

def smart_harmonize(example, source_name):
    raw_text = example.get("text", example.get("article", example.get("content", "")))
    clean_text = sentence_aware_truncate(raw_text)
    
    is_ai = False
    
    if "label" in example:
        val = str(example["label"]).lower()
        is_ai = val in ["1", "true", "ai", "machine"]
    elif "is_ai" in example:
        val = str(example["is_ai"]).lower()
        is_ai = val in ["1", "true"]
    elif "source" in example:
        is_ai = "human" not in str(example["source"]).lower()
    elif "model" in example:
        is_ai = "human" not in str(example["model"]).lower()
        
    return {"text": clean_text, "label": 1 if is_ai else 0, "source": source_name}

def build_megacorpus():
    print("☁️ Loading datasets directly from Hugging Face...")
    
    print("Loading HC3...")
    hc3 = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
    hc3 = hc3.map(flatten_hc3, batched=True, remove_columns=hc3.column_names)
    
    print("Loading artem9k (multi-domain)...")
    artem = load_dataset("artem9k/ai-text-detection-pile", split="train")
    artem = artem.map(lambda x: smart_harmonize(x, "artem9k"), remove_columns=artem.column_names)
    
    try:
        print("Loading MAGE...")
        mage = load_dataset("yaful/MAGE", split="train")
        mage = mage.map(lambda x: smart_harmonize(x, "MAGE"), remove_columns=mage.column_names)
    except Exception as e:
        print(f"⚠️ Skipping MAGE ({e})")
        mage = None
    
    print("Loading RAID (adversarial data)...")
    raid = load_dataset("liamdugan/raid", split="train")
    raid = raid.map(lambda x: smart_harmonize(x, "RAID"), remove_columns=raid.column_names)

    datasets_to_interleave = [d for d in [hc3, artem, mage, raid] if d is not None]
    probs = [1.0 / len(datasets_to_interleave)] * len(datasets_to_interleave)

    print(f"✅ Harmonization complete. Mixing {len(datasets_to_interleave)} datasets...")
    
    mega_corpus = interleave_datasets(
        datasets_to_interleave, 
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted"
    )
    
    return mega_corpus