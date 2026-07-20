import transformers
import shap
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_html_heatmap(tokens, shap_scores, title):
    max_abs_score = max(np.max(shap_scores), abs(np.min(shap_scores)))
    if max_abs_score == 0: max_abs_score = 1e-9 

    html = f"<div style='font-family: \"Times New Roman\", serif; margin-bottom: 40px;'>"
    html += f"<h3 style='color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px;'>{title}</h3>"
    html += "<div style='line-height: 2.2; font-size: 16px; padding: 20px; border: 1px solid #eee; background-color: #fafafa; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);'>"

    for token, score in zip(tokens, shap_scores):
        clean_token = token.replace('Ġ', ' ').replace(' ', ' ')
        alpha = abs(score) / max_abs_score
        alpha = min(alpha * 1.5, 0.8) 

        if score > 0:
            color = f"rgba(255, 99, 71, {alpha})" # Rot = KI
        else:
            color = f"rgba(100, 149, 237, {alpha})" # Blau = Mensch

        html += f"<span style='background-color: {color}; padding: 2px 0px; border-radius: 3px;'>{clean_token}</span>"

    html += "</div></div>"
    return html

def find_extreme_true_positives(model_path, df, device, top_n=3):
    print(f"\nFind the {top_n} most extreme true positives using RoBERTa...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    ai_df = df[df['label'] == 1].copy()
    texts = ai_df['text'].tolist()
    
    ai_probabilities = []
    batch_size = 32
    
    print("Rate AI-generated texts...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            ai_probabilities.extend(probs)
            
    ai_df['ai_prob'] = ai_probabilities

    extreme_tps = ai_df.sort_values(by='ai_prob', ascending=False).head(top_n)
    
    return extreme_tps

def analyze_model_shap(model_path, text, device):
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
    shap_values = explainer([text])
    
    return shap_values.data[0], shap_values.values[0][:, 1] 

def main():
    distilbert_path = "models/distilbert_hc3" 
    roberta_path = "models/roberta_artem9k" 
    news_path = "data/processed/FINAL_local_news_validation.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(news_path)
    
    extreme_cases = find_extreme_true_positives(roberta_path, df, device, top_n=3)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Top 3 Extreme True Positives</title>
        <style>
            body { padding: 40px; max-width: 1000px; margin: 0 auto; color: #000; }
            .legend { margin-top: 20px; font-family: 'Times New Roman', serif; font-size: 14px; padding: 15px; border: 1px solid #ccc; background: #fff; display: inline-block; }
            .box { display: inline-block; width: 15px; height: 15px; margin-right: 5px; vertical-align: middle; border: 1px solid #999; }
            .red-box { background-color: rgba(255, 99, 71, 0.6); }
            .blue-box { background-color: rgba(100, 149, 237, 0.6); }
            .case-container { margin-bottom: 80px; padding-bottom: 40px; border-bottom: 3px dashed #bbb; }
        </style>
    </head>
    <body>
        <h1 style="font-family: 'Times New Roman', serif; text-align: center;">SHAP Architectural Comparison: Extreme True Positives</h1>
        <div class="legend" style="margin-bottom: 40px;">
            <strong>Legend:</strong><br><br>
            <div><span class="box red-box"></span> Pushes prediction toward <b>AI</b> (High impact)</div>
            <div style="margin-top: 8px;"><span class="box blue-box"></span> Pushes prediction toward <b>Human</b> (Low/Negative impact)</div>
        </div>
    """

    for i, (index, row) in enumerate(extreme_cases.iterrows()):
        text = row['text']
        ai_prob = row['ai_prob']
        print(f"\n[{i+1}/3] Analyzing Text (AI Probability: {ai_prob:.4f})...")
        
        tokens_db, scores_db = analyze_model_shap(distilbert_path, text, device)
        tokens_rob, scores_rob = analyze_model_shap(roberta_path, text, device)
        
        html_content += f"<div class='case-container'>"
        html_content += f"<h2 style='font-family: \"Times New Roman\", serif; color: #006400;'>True Positive #{i+1} (Model Confidence that this is AI: {ai_prob*100:.2f}%)</h2>"
        html_content += create_html_heatmap(tokens_db, scores_db, "DistilBERT (6-Layer) Analysis")
        html_content += create_html_heatmap(tokens_rob, scores_rob, "RoBERTa (12-Layer) Analysis")
        html_content += "</div>"

    html_content += "</body></html>"

    output_dir = "results/"
    output_path = os.path.join(output_dir, "shap_top3_true_positives.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ DONE! HTML saved under: {output_path}")

if __name__ == "__main__":
    main()