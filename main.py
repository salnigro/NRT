import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

import random
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
torch.manual_seed(42)

def generate_risk_label(row):
    """
    Heuristically generate risk labels and explainable target text based on 
    mentions of substance abuse, withdrawal, or emotional distress.
    """
    review = str(row['review']).lower()
    condition = str(row['condition']).lower()
    
    risk_keywords = ['addict', 'withdrawal', 'relapse', 'suicide', 'abuse', 'overdose', 'dependence', 'craving']
    found_keywords = [kw for kw in risk_keywords if kw in review or kw in condition]
    
    if len(found_keywords) > 0:
        words_str = ", ".join(found_keywords)
        return f"Yes, the user mentions {words_str} indicating distress or abuse potential."
    else:
        return "No, there are no immediate substance abuse risk signals."

def prepare_data(csv_path="data/drugsComTrain_raw.csv", sample_size=1000):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop NAs in review
    df = df.dropna(subset=['review', 'condition'])
    
    # Apply heuristic labeling
    df['target_text'] = df.apply(generate_risk_label, axis=1)
    
    # Add an instruction prompt
    instruction = "Task: Read the following patient review and identify if there is evidence of emotional distress, substance abuse, or relapse. Explain your reasoning. Review: "
    df['input_text'] = instruction + df['review']
    
    # We want to balance the dataset a bit for fine-tuning
    df_risk = df[df['target_text'].str.startswith("Yes")]
    df_safe = df[df['target_text'].str.startswith("No")]
    
    # Take a sample size
    half_size = sample_size // 2
    df_risk_sample = df_risk.sample(n=min(half_size, len(df_risk)), random_state=42)
    df_safe_sample = df_safe.sample(n=min(half_size, len(df_safe)), random_state=42)
    
    df_final = pd.concat([df_risk_sample, df_safe_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_final[['input_text', 'target_text']]

def main():
    model_name = "google/flan-t5-base"
    print(f"Loading tokenizer {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    
    print(f"Loading model {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Prepare data
    df_train = prepare_data("data/drugsComTrain_raw.csv", sample_size=1000)
    df_test = prepare_data("data/drugsComTest_raw.csv", sample_size=200)
    
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
        
        # Replace padding token id with -100 so it's ignored by the loss
        labels = targets["input_ids"]
        labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels
        ]
        
        inputs["labels"] = labels
        return inputs
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # changed from evaluation_strategy for newer HF compatibility
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",  # Don't save intermediate checkpoints to save space/time
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    # Inference / Explainability Demo
    print("\n\n--- Inference and Explainability Demo ---")
    
    test_reviews = [
        "I've been taking this for pain. At first it worked, but now I'm starting to get cravings and if I don't take it I have bad withdrawal symptoms and feel like I'm addicted.",
        "This antibiotic cleared up my infection in about a week. No major side effects noticed."
    ]
    
    for rev in test_reviews:
        instruction = "Task: Read the following patient review and identify if there is evidence of emotional distress, substance abuse, or relapse. Explain your reasoning. Review: "
        input_text = instruction + rev
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate with the fine-tuned model
        outputs = model.generate(
            **inputs, 
            max_length=128, 
            num_beams=4,
            early_stopping=True
        )
        
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nReview: {rev}")
        print(f"AI Explainability Output: {explanation}")

if __name__ == "__main__":
    main()