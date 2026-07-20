import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def main():
    # 1. Define paths
    model_path = "models/distilbert_hc3"
    data_path = "data/processed/OOD_Test_Dataset_Synced.csv"

    print(f"Loading test data from: {data_path}")
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    models_used = df['model_used'].tolist()

    # 2. Load model and tokenizer
    print(f"Loading trained model from: {model_path}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # 3. Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model running on: {device}")

    # 4. Make predictions
    batch_size = 16
    predictions = []
    
    print("Starting predictions on the OOD dataset...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    # 5. Evaluate and compute metrics
    print("\n" + "="*50)
    print("RESULTS ON THE HYPER-LOCAL OOD DATA")
    print("="*50)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # FPR = False Positives / (False Positives + True Negatives)
    fpr = fp / (fp + tn)

    print("\nConfusion Matrix:")
    print(f"Real human texts (0): {tn} correctly identified (TN) | {fp} wrongly marked as AI (FP)")
    print(f"Real AI texts (1):    {fn} wrongly marked as human (FN) | {tp} correctly identified (TP)")
    
    print(f"\n-> False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
    if fpr > 0.05:
        print("⚠️ Warning: The false positive rate is above 5%. The model is struggling with human texts.")
    
    print("\nDetailed classification report:")
    print(classification_report(true_labels, predictions, target_names=["Human (0)", "AI (1)"]))

if __name__ == "__main__":
    main()