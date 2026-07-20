import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": acc, "f1": f1}

def prepare_hc3_dataset():
    print("Loading raw HC3 data from the Hugging Face Hub...")
    raw_dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
    
    texts = []
    labels = []
    
    print("Extracting human and AI texts and assigning labels...")
    for row in raw_dataset:
        # 0 = Human
        for human_text in row['human_answers']:
            if human_text.strip():
                texts.append(human_text)
                labels.append(0)
                
        # 1 = AI (ChatGPT)
        for ai_text in row['chatgpt_answers']:
            if ai_text.strip():
                texts.append(ai_text)
                labels.append(1)
                
    # Convert to pandas DataFrame for an easy split
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Reduce dataset size
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    
    processed_dataset = Dataset.from_pandas(df)
    
    # 90% training, 10% evaluation (validation)
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset

def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load and prepare data
    dataset = prepare_hc3_dataset()

    # 2. Load tokenizer
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 3. Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Apply tokenization
    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Load model
    print("Initializing DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # 5. Training parameters
    output_dir = "models/distilbert_hc3"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=100,
    )

    # 6. Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], 
        compute_metrics=compute_metrics,
    )

    # 7. Start training
    print("Starting training...")
    trainer.train() 
    
    print(f"Training finished! Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved successfully!")

if __name__ == "__main__":
    main()