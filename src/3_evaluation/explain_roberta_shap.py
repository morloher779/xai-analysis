import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import os
import sys

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
    model_path = "models/roberta_hc3"
    data_path = "data/processed/OOD_Test_Dataset_Synced.csv"
    output_dir = "results/"
    
    print("Loading the RoBERTa model and tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.config.id2label = {0: "Human", 1: "AI"}
    model.config.label2id = {"Human": 0, "AI": 1}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pred_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, top_k=None 
    )

    df = pd.read_csv(data_path)
    sample_df = df.sample(min(2000, len(df)), random_state=42).copy()
    texts = sample_df['text'].tolist()
    
    print("Searching for a False Positive (Real Human, but Model says AI)...")
    predictions = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
            pred = torch.argmax(model(**inputs).logits, dim=-1).item()
            predictions.append(pred)
            
    sample_df['pred'] = predictions

    # FILTERN: Label 0 (Mensch) aber Prediction 1 (KI)
    false_positives = sample_df[(sample_df['label'] == 0) & (sample_df['pred'] == 1)]
    
    if len(false_positives) == 0:
        print("No False Positive found in the sample.")
        sys.exit(1)

    fp_text = false_positives.iloc[0]['text']
    
    print(f"False Positive found! Calculating SHAP... (Text length: {len(fp_text.split())} words)")
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer([fp_text])

    # Plot erstellen
    fp_vals = shap_values[0, :, "AI"].values
    fp_tokens = shap_values[0, :, "AI"].data
    plot_path = os.path.join(output_dir, "shap_roberta_false_positive.png")
    
    plot_custom_shap_bar(
        fp_vals, fp_tokens, 
        "False Positive (RoBERTa): Human Text Misclassified as AI", 
        plot_path
    )
    print(f"[OK] Image saved under: {plot_path}")

if __name__ == "__main__":
    main()