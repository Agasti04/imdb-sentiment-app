# src/train.py

import os
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    print("Loading IMDb dataset from Hugging Face hub...")
    dataset = load_dataset("imdb")

    # --- Optional: Subsample for faster CPU training (comment out to use full data) ---
    # train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    # test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))
    # dataset = {"train": train_dataset, "test": test_dataset}

    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Tokenizing dataset...")
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        # Change max_length=512 if you want full length but slower training

    tokenized_datasets = dataset.map(tokenize, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir="../models/imdb-distilbert",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=100,
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()4

    save_path = os.path.abspath("../models/imdb-distilbert-trained")
    print(f"Saving model to {save_path}...")
    trainer.save_model(save_path)

if __name__ == "__main__":
    main()
