import transformers
import shap
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from tqdm import tqdm

def main():
    print("Start global SHAP analysis...")
    
    model_path = "/models/roberta_artem9k" 
    news_path = "/data/processed/FINAL_local_news_validation.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    if not hasattr(model.config, 'id2label') or model.config.id2label[0] == "LABEL_0":
        model.config.id2label = {0: "Human", 1: "AI"}
        model.config.label2id = {"Human": 0, "AI": 1}

    pred_pipeline = transformers.pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device.index if device.type == 'cuda' else -1,
        top_k=None 
    )
    explainer = shap.Explainer(pred_pipeline)

    df = pd.read_csv(news_path)
    ai_texts = df[df['label'] == 1]['text'].tolist()[:50]
    
    print("Calculate SHAP scores for 50 texts (this may take a few minutes)...")
    shap_values = explainer(ai_texts)
    
    token_impact = defaultdict(list)
    
    for i in range(len(ai_texts)):
        tokens = shap_values.data[i]
        scores_ai = shap_values.values[i][:, 1]
        
        for tok, score in zip(tokens, scores_ai):
            clean_tok = tok.replace('Ġ', '').strip()
            
            if len(clean_tok) > 2 and clean_tok.lower() not in ['the', 'and', 'for', 'that', 'with', 'this']:
                token_impact[clean_tok].append(score)

    mean_impact = {}
    for tok, scores in token_impact.items():
        if len(scores) >= 5: 
            mean_impact[tok] = np.mean(scores)
            
    sorted_impact = sorted(mean_impact.items(), key=lambda x: x[1], reverse=True)[:20]
    
    tokens_to_plot = [x[0] for x[0] in sorted_impact]
    scores_to_plot = [x[1] for x[1] in sorted_impact]
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(tokens_to_plot))
    
    plt.barh(y_pos, scores_to_plot, align='center', color='#d62728', edgecolor='black', linewidth=0.5)
    plt.yticks(y_pos, tokens_to_plot)
    plt.gca().invert_yaxis()
    
    plt.xlabel("Mean SHAP Value (Impact pushing towards 'AI' classification)")
    plt.title("Global SHAP Signature: Top 20 Tokens Driving 'AI' Classification (n=50)")
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = "/results/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shap_global_signature_ai_REAL.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ DONE! Real plot saved under: {output_path}")

if __name__ == "__main__":
    main()