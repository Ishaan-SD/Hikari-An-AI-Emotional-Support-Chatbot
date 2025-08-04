# %pip install evaluate
# %pip install --upgrade transformers==4.40.1
# %pip install peft==0.10.0 sentence-transformers==2.2.2

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import os

def train_transformer_model():

    input_filename = r'/content/cleaned_merged_dataset.csv'
    model_name = "distilbert-base-uncased"
    output_dir = "./transformer_model_merged"

    # --- 1. Load and Prepare Data ---
    print(f"Loading cleaned data from '{input_filename}'...")
    df = pd.read_csv(input_filename)
    df.dropna(subset=['Cleaned_Text', 'emotion'], inplace=True)

    # Create a numerical label column
    unique_emotions = sorted(df['emotion'].unique()) # Sort for consistency
    label2id = {label: i for i, label in enumerate(unique_emotions)}
    id2label = {i: label for i, label in enumerate(unique_emotions)}
    df['label'] = df['emotion'].map(label2id)

    num_labels = len(unique_emotions)
    print(f"Found {num_labels} unique emotion classes.")

    # Convert pandas DataFrame to Hugging Face Dataset object
    hf_dataset = Dataset.from_pandas(df)

    # FIX: Cast the 'label' column to a ClassLabel type for stratification.
    # This is required by the `train_test_split` function's `stratify_by_column` argument.
    class_label_feature = ClassLabel(names=unique_emotions)
    hf_dataset = hf_dataset.cast_column("label", class_label_feature)

    # --- 2. Load Tokenizer and Preprocess ---
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        # The tokenizer will handle padding and truncation for us.
        return tokenizer(examples["Cleaned_Text"], truncation=True, padding=True, max_length=256)

    print("Tokenizing the dataset...")
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)

    # Split the dataset into training and testing sets
    train_test_split_dict = tokenized_dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    train_dataset = train_test_split_dict["train"]
    test_dataset = train_test_split_dict["test"]

    # --- 3. Load Pre-trained Model ---
    print(f"Loading pre-trained model '{model_name}' for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Set up Trainer ---
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, # 3 epochs is often a good balance for fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch", # Changed from evaluation_strategy to eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- 5. Start Fine-Tuning ---
    print("\n--- Starting Transformer Fine-Tuning ---")
    trainer.train()
    print("Fine-tuning complete.")

    # --- 6. Evaluate and Save ---
    print("\n--- Final Model Evaluation ---")
    eval_results = trainer.evaluate()
    print(f"Accuracy on the test set: {eval_results['eval_accuracy']:.4f}")

    print(f"\nSaving the best model to '{output_dir}'...")
    trainer.save_model(output_dir)

    print("\n--- Transformer Training Process Finished Successfully ---")
    print("Your state-of-the-art model is now saved.")

if __name__ == '__main__':
    train_transformer_model()
    # print(TrainingArguments.__module__)