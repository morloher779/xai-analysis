import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

def plot_custom_shap_bar(shap_values_1d, tokens, title, output_path, max_display=12):
    data = []
    for val, tok in zip(shap_values_1d, tokens):
        clean_tok = tok.strip()
        if clean_tok: 
            data.append((val, clean_tok))
            
    data.sort(key=lambda x: abs(x[0]), reverse=True)
    top_data = data[:max_display]
    top_data.sort(key=lambda x: x[0]) 
    
    vals = [x[0] for x in top_data]
    labels = [x[1] for x in top_data]
    
    colors = ['#1f77b4' if v < 0 else '#d62728' for v in vals]
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals, color=colors)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.axvline(0, color='black', linewidth=1)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Pushes towards AI'),
        Patch(facecolor='#1f77b4', label='Pushes towards Human')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model_path = "models/roberta_artem9k"
    data_path = "data/processed/OOD_Test_Dataset_Synced.csv"
    output_dir = "results/"
    
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
    
    print("Scanning all texts to find the 'Unicorn' (False Positive)...")
    predictions = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            preds = torch.argmax(model(**inputs).logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            
    df['pred'] = predictions

    false_positives = df[(df['label'] == 0) & (df['pred'] == 1)]
    
    if len(false_positives) == 0:
        print("No False Positive found. The model is perfect here!")
        sys.exit(0)

    fp_text = false_positives.iloc[0]['text']
    
    print(f"\n[!] UNICORN FOUND! Text length: {len(fp_text.split())} words.")
    print("Calculating SHAP values for this special text...")
    
    pred_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, top_k=None 
    )
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer([fp_text])

    plot_path = os.path.join(output_dir, "shap_roberta_artem9k_unicorn.png")
    
    plot_custom_shap_bar(
        shap_values[0, :, "AI"].values, shap_values[0, :, "AI"].data, 
        "The 'Unicorn' False Positive: RoBERTa (artem9k)", 
        plot_path
    )
    
    print("\n" + "="*50)
    print("THE TEXT OF THE UNICORN:")
    print("="*50)
    print(fp_text)
    print("="*50)
    
    print(f"\n[OK] SHAP Image saved under: {plot_path}")

if __name__ == "__main__":
    main()