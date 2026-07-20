import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import os
import sys

def plot_custom_shap_bar(shap_values_1d, tokens, title, output_path, max_display=12):
    """Creates a clean, error-free bar plot for SHAP text values in English."""
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
    
    # Colors: Red for AI (>0), Blue for Human (<0)
    colors = ['#1f77b4' if v < 0 else '#d62728' for v in vals]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, vals, color=colors)
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
    model_path = "models/distilbert_hc3"
    data_path = "data/processed/OOD_Test_Dataset_Synced.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Model or Data path does not exist.")
        sys.exit(1)

    print("Loading model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    model.config.id2label = {0: "Human", 1: "AI"}
    model.config.label2id = {"Human": 0, "AI": 1}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Initializing Hugging Face Pipeline...")
    pred_pipeline = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, 
        top_k=None 
    )

    print("Loading OOD dataset and searching for case studies...")
    df = pd.read_csv(data_path)
    sample_df = df.sample(min(1000, len(df)), random_state=42).copy()
    texts = sample_df['text'].tolist()
    
    predictions = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
            
    sample_df['pred'] = predictions

    true_positives = sample_df[(sample_df['label'] == 1) & (sample_df['pred'] == 1)]
    false_negatives = sample_df[(sample_df['label'] == 1) & (sample_df['pred'] == 0)]
    
    if len(true_positives) == 0 or len(false_negatives) == 0:
        print("Could not find both cases in the sample.")
        sys.exit(1)

    tp_text = true_positives.iloc[0]['text']
    fn_text = false_negatives.iloc[0]['text']
    texts_to_explain = [tp_text, fn_text]

    print("\nInitializing SHAP Explainer...")
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer(texts_to_explain)

    print("\nGenerating Outputs...")

    # Figure 1: True Positive
    tp_vals = shap_values[0, :, "AI"].values
    tp_tokens = shap_values[0, :, "AI"].data
    plot_custom_shap_bar(
        tp_vals, tp_tokens, 
        "True Positive: Top Features Identifying AI-Generated Text", 
        os.path.join(output_dir, "shap_distilbert_true_positive.png")
    )
    print("   [OK] True Positive PNG saved.")

    # Figure 2: False Negative
    fn_vals = shap_values[1, :, "AI"].values
    fn_tokens = shap_values[1, :, "AI"].data
    plot_custom_shap_bar(
        fn_vals, fn_tokens, 
        "False Negative: Top Features Deceiving the Model", 
        os.path.join(output_dir, "shap_distilbert_false_negative.png")
    )
    print("   [OK] False Negative PNG saved.")

    # HTML
    html_path = os.path.join(output_dir, "shap__distilbert_interactive_dashboard.html")
    html_content = shap.plots.text(shap_values, display=False)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"   [OK] HTML saved.")

if __name__ == "__main__":
    main()