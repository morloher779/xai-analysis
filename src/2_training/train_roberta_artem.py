import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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

def prepare_artem9k_dataset():
    """Load the artem9k dataset directly from the local Hugging Face cache."""
    print("Loading artem9k data from cache...")
    
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")
    
    print(f"Dataset loaded. Original columns: {dataset.column_names}")

    def map_labels(example):
        if 'source' in example:
            example['label'] = 0 if example['source'] == 'human' else 1
        return example

    print("Formatting labels (0 = Human, 1 = AI)...")
    dataset = dataset.map(map_labels)
    
    dataset = dataset.select_columns(['text', 'label'])
    
    # random sample of 20,000 texts
    print("Sampling data for training...")
    dataset = dataset.shuffle(seed=42).select(range(20000))
    
    # Train (90%) Test (10%)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset

def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
    dataset = prepare_artem9k_dataset()

    print("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing artem9k data (this may take a moment)...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("Initializing fresh RoBERTa model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', 
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1}
    )

    output_dir = "models/roberta_artem9k"
    
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

    print("Starting RoBERTa training on artem9k... please buckle up!")
    trainer.train() 
    
    print(f"Training finished! Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved successfully!")

if __name__ == "__main__":
    main()