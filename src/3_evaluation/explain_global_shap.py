import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

def main():
    model_path = "models/roberta_artem9k"
    data_path = "data/processed/OOD_Test_Dataset_Synced.csv" # Passe an, falls nötig
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.id2label = {0: "Human", 1: "AI"}
    model.config.label2id = {"Human": 0, "AI": 1}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loading OOD dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Searching for True Positives (successfully identified AI texts)...")
    df_ai = df[df['label'] == 1].copy()
    texts_ai = df_ai['text'].tolist()
    
    predictions = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(texts_ai), batch_size)):
            batch_texts = texts_ai[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            preds = torch.argmax(model(**inputs).logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            
    df_ai['pred'] = predictions
    true_positives = df_ai[df_ai['pred'] == 1]['text'].tolist()
    
    print(f"Found True Positives: {len(true_positives)}")
    
    sample_size = min(50, len(true_positives))
    texts_to_explain = true_positives[:sample_size]
    
    print(f"Calculating SHAP values for {sample_size} texts. This will take a moment...")
    pred_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, top_k=None 
    )
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer(texts_to_explain)

    print("Aggregating SHAP values for the global signature...")
    token_importance = defaultdict(list)
    
    for i in range(len(texts_to_explain)):
        values = shap_values[i, :, "AI"].values
        data = shap_values[i, :, "AI"].data
        
        for val, tok in zip(values, data):
            clean_tok = tok.strip()
            if len(clean_tok) > 1 and clean_tok.isalpha():
                token_importance[clean_tok].append(val)

    global_importance = {}
    for tok, vals in token_importance.items():
        if len(vals) >= 3: 
            global_importance[tok] = np.mean(vals)
            
    top_tokens = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    labels = [x[0] for x in top_tokens]
    vals = [x[1] for x in top_tokens]

    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], vals[::-1], color='#d62728')
    plt.title("Global SHAP Signature: Top 20 Tokens Driving 'AI' Classification", fontsize=16, pad=20)
    plt.xlabel("Mean SHAP Value (Impact on Model Output towards 'AI')", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(output_dir, "shap_global_signature_ai.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("TOP 10 AI VOCABULARY (according to RoBERTa):")
    for i, (tok, val) in enumerate(top_tokens[:10]):
        print(f"{i+1}. '{tok}' (Impact: +{val:.4f})")
    print("="*50)
    print(f"\n[OK] Global SHAP-Plot successfully saved under: {plot_path}")

if __name__ == "__main__":
    main()