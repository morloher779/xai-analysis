import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

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
    
    colors = ['#1f77b4' for _ in vals]
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals, color=colors)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("SHAP Value (Impact pushing towards 'Human' classification)", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model_path = "models/roberta_artem9k"
    data_path = "data/processed/OOD_Test_Dataset_Synced.csv"
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

    print("Loading the dataset...")
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    print("Scanning all texts to find 'Sneaky AI' (False Negatives)...")
    predictions = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            preds = torch.argmax(model(**inputs).logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            
    df['pred'] = predictions

    false_negatives = df[(df['label'] == 1) & (df['pred'] == 0)]
    
    if len(false_negatives) == 0:
        print("No False Negatives found!")
        sys.exit(0)

    print(f"\n[!] {len(false_negatives)} Sneaky AIs found!")
    
    sneaky_text = false_negatives.iloc[0]['text']
    
    print("Calculating SHAP values for this tricked text...")
    
    pred_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, top_k=None 
    )
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer([sneaky_text])

    plot_path = os.path.join(output_dir, "shap_sneaky_ai.png")
    
    plot_custom_shap_bar(
        shap_values[0, :, "Human"].values, shap_values[0, :, "Human"].data, 
        "The 'Sneaky AI': Tokens that tricked RoBERTa into predicting 'Human'", 
        plot_path
    )
    
    print("\n" + "="*70)
    print("THE TEXT OF THE SNEAKY AI (Real AI, misclassified as Human by the model):")
    print("="*70)
    print(sneaky_text)
    print("="*70)
    
    print("\nTop 5 Words that tricked the model (Impact on 'Human'):")
    tokens = shap_values[0, :, "Human"].data
    vals = shap_values[0, :, "Human"].values
    word_impacts = [(tok.strip(), val) for tok, val in zip(tokens, vals) if tok.strip() and len(tok.strip()) > 1]
    word_impacts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (word, impact) in enumerate(word_impacts[:5]):
        print(f"{i+1}. '{word}' (+{impact:.4f})")
        
    print(f"\n[OK] SHAP Image saved under: {plot_path}")

if __name__ == "__main__":
    main()