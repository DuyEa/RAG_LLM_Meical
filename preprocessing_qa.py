from datasets import load_dataset, Dataset
import pandas as pd


dataset_full = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
print(f"Full dataset size (en config): {len(dataset_full)}")

# quick testing
dataset = dataset_full.select(range(2000))
print(f"Subset size: {len(dataset)}")

# Convert to DataFrame
df = dataset.to_pandas()

print("Dataset columns:", df.columns.tolist())

df_clean = df.dropna(subset=['Question', 'Complex_CoT', 'Response']).drop_duplicates(subset=['Question'])
print(f"Cleaned unique size: {len(df_clean)}")  #

# Format prompt for fine-tuning: combine Question + CoT + Response (full reasoning chain)
def format_prompt(row):
    return f"Question: {row['Question']}\nComplex CoT: {row['Complex_CoT']}\nResponse: {row['Response']}"

df_clean['text'] = df_clean.apply(format_prompt, axis=1)

# Add metadata (original question) into dataset for traceability
df_with_meta = df_clean[['text', 'Question']].rename(columns={'Question': 'orig_question'})

# Save as Dataset + split into train/eval (90/10)
train_dataset = Dataset.from_pandas(df_with_meta)
train_dataset = train_dataset.train_test_split(test_size=0.1)
train_dataset.save_to_disk("processed_medical_qa_fixed_en")

# Preview a short sample of formatted data (first entry)
print("Sample formatted data (first entry):")
print(train_dataset['train'][0]['text'][:500] + "...")
