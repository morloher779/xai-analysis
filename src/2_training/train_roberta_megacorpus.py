import sys
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_gen_dir = os.path.join(os.path.dirname(current_dir), '1_data_generation')
print(f"current dir:{current_dir} ")

if data_gen_dir not in sys.path:
    sys.path.append(data_gen_dir)

from data_pipeline import build_megacorpus

def main():
    print("Starting mega-corpus training for RoBERTa...")

    mega_dataset = build_megacorpus()

    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length" ,
            truncation=True,
            max_length=256
        )
    
    print("Tokenizing the mega-corpus (this may take a moment)...")
    tokenized_datasets = mega_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text", "source"] 
    )

    print("Loading RoBERTa-base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1}
    )

    output_model_dir = "models/roberta_megacorpus"
    
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="epoch",
        save_total_limit=2,
        
        fp16=True,
        dataloader_num_workers=4,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("Starting training loop...")

    trainer.train(resume_from_checkpoint=True)
    
    # Save final model
    print(f"✅ Training finished. Saving model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("All done")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ WARNING: No GPU found.")
    else:
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        
    main()