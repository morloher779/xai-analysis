import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    """Compute F1 score and accuracy during training."""
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": acc, "f1": f1}

def prepare_hc3_dataset():
    """Convert the HC3 dataset into a clean format (text, label)."""
    print("Loading raw HC3 data from the Hugging Face Hub...")
    raw_dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
    
    texts = []
    labels = []
    
    print("Extracting human and AI texts and assigning labels...")
    for row in raw_dataset:
        for human_text in row['human_answers']:
            if human_text.strip():
                texts.append(human_text)
                labels.append(0)
                
        for ai_text in row['chatgpt_answers']:
            if ai_text.strip():
                texts.append(ai_text)
                labels.append(1)
                
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    
    processed_dataset = Dataset.from_pandas(df)
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset

def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    dataset = prepare_hc3_dataset()

    # 1. RoBERTa tokenizer
    print("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 2. RoBERTa model
    print("Initializing RoBERTa model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', 
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1}
    )

    # 3. Training parameters
    output_dir = "models/roberta_hc3"
    
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], 
        compute_metrics=compute_metrics,
    )

    print("Starting RoBERTa training...")
    trainer.train()
    
    print(f"Training finished! Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Successfully saved!")

if __name__ == "__main__":
    main()