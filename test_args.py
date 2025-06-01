from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="test_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
)

print("All good! Your TrainingArguments is working.")
