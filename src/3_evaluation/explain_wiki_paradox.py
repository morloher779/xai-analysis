import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import os
import sys
from datasets import load_dataset

def plot_custom_shap_bar(shap_values_1d, tokens, title, output_path, max_display=15):
    data = []
    for val, tok in zip(shap_values_1d, tokens):
        clean_tok = tok.strip()
        if clean_tok and len(clean_tok) > 1: 
            data.append((val, clean_tok))
            
    data.sort(key=lambda x: x[0], reverse=True)
    top_data = data[:max_display]
    top_data.sort(key=lambda x: x[0]) 
    
    vals = [x[0] for x in top_data]
    labels = [x[1] for x in top_data]
    
    colors = ['#d62728' for _ in vals]
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals, color=colors)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("SHAP Value (Impact pushing towards 'AI' classification)", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model_path = "models/roberta_artem9k"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading the RoBERTa (artem9k) model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.id2label = {0: "Human", 1: "AI"}
    model.config.label2id = {"Human": 0, "AI": 1}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Loading human Wikipedia texts from Hugging Face...")
    try:
        dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    df = dataset.to_pandas()
    human_texts = df['wiki_intro'].head(500).tolist() 
    
    print("Searching for the 'Wikipedia-Paradoxon' (Human Text misclassified as AI)...")
    
    found_text = None
    with torch.no_grad():
        for text in human_texts:
            inputs = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            pred = torch.argmax(model(**inputs).logits, dim=-1).item()
            if pred == 1:
                found_text = text
                break
                
    if not found_text:
        print("No False Positive found in this sample.")
        sys.exit(0)

    print("\n[!] Wikipedia-Paradoxon found!")
    print("Calculating SHAP values...")
    
    pred_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, top_k=None 
    )
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer([found_text])

    plot_path = os.path.join(output_dir, "shap_wiki_paradox.png")
    
    plot_custom_shap_bar(
        shap_values[0, :, "AI"].values, shap_values[0, :, "AI"].data, 
        "The Wikipedia Paradox: Tokens that tricked RoBERTa into predicting 'AI'", 
        plot_path
    )
    
    print("\n" + "="*70)
    print("THE TEXT OF THE WIKIPEDIA-PARADOXON (Real Human, misclassified as AI by the model):")
    print("="*70)
    print(found_text)
    print("="*70)
    
    print("\nTop 5 Words that tricked the model (Impact on 'AI'):")
    tokens = shap_values[0, :, "AI"].data
    vals = shap_values[0, :, "AI"].values
    word_impacts = [(tok.strip(), val) for tok, val in zip(tokens, vals) if tok.strip() and len(tok.strip()) > 1]
    word_impacts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (word, impact) in enumerate(word_impacts[:5]):
        print(f"{i+1}. '{word}' (+{impact:.4f})")
        
    print(f"\n[OK] SHAP Image saved under: {plot_path}")

if __name__ == "__main__":
    main()