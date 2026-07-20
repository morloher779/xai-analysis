import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
from tqdm import tqdm
import sys

def main():
    model_path = "models/roberta_artem9k"
    
    print("Loading Wikipedia dataset (cross-domain) from Hugging Face...")
    try:
        dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    df_raw = dataset.to_pandas()
    print(f"Dataset loaded! Found pairs: {len(df_raw)}")
    
    df_sample = df_raw.sample(2000, random_state=42).copy()
    
    print("Formatting data (Human = 0, AI = 1)...")
    df_human = pd.DataFrame({'text': df_sample['wiki_intro'], 'label': 0})
    df_ai = pd.DataFrame({'text': df_sample['generated_intro'], 'label': 1})
    
    df_test = pd.concat([df_human, df_ai]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    texts = df_test['text'].tolist()
    true_labels = df_test['label'].tolist()

    print("Loading trained RoBERTa model (artem9k)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    print(f"Model running on: {device}")

    batch_size = 16
    predictions = []
    
    print("Starting predictions on the academic Wikipedia dataset...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    print("\n" + "="*50)
    print("RESULTS: RoBERTa (artem9k) ON WIKIPEDIA")
    print("="*50)

    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\nConfusion Matrix:")
    print(f"Real human texts (0): {tn} correctly identified (TN) | {fp} wrongly marked as AI (FP)")
    print(f"Real AI texts (1):    {fn} wrongly marked as human (FN) | {tp} correctly identified (TP)")
    
    print(f"\n-> False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
    
    print("\nDetailed classification report:")
    print(classification_report(true_labels, predictions, target_names=["Human (0)", "AI (1)"]))

if __name__ == "__main__":
    main()