import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import os

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

    print("Loading the dataset and selecting an interesting sample...")
    df = pd.read_csv(data_path)
    
    df_human = df[df['label'] == 0].head(10)
    df_ai = df[df['label'] == 1].head(10)
    df_sample = pd.concat([df_human, df_ai]).sample(frac=1, random_state=42) # Durchmischen
    texts = df_sample['text'].tolist()
    
    print("Calculating SHAP values (this takes a moment)...")
    pred_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1, top_k=None 
    )
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer(texts)

    print("Generating interactive HTML dashboard...")
    html_content = shap.plots.text(shap_values, display=False)

    full_html = f"""
    <html>
    <head><title>RoBERTa SHAP Dashboard</title></head>
    <body style="font-family: sans-serif; padding: 20px;">
        <h2>Interactive SHAP Explanations: RoBERTa (artem9k)</h2>
        <p>Hover over the text to see how each token pushes the prediction towards 'Human' or 'AI'.</p>
        {html_content}
    </body>
    </html>
    """
    
    output_file = os.path.join(output_dir, "shap_roberta_interactive_dashboard.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"\n[OK] RoBERTa Dashboard successfully saved under: {output_file}")

if __name__ == "__main__":
    main()