from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_from_disk
import torch
from huggingface_hub import login
import os

# Login to Hugging Face Hub
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    login()

# Load dataset
dataset = load_from_disk("processed_medical_qa_fixed_en")
train_ds = dataset['train']
eval_ds = dataset['test']
print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

# Quantization in 4-bit with fp16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Max memory configuration for RTX 3060
max_memory = {0: "5GiB", "cpu": "20GiB"}

# Load model with fp16
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory=max_memory,
    offload_folder=None,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("Model loaded successfully! Ready for training.")

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

train_ds = train_ds.map(tokenize_function, batched=True)
eval_ds = eval_ds.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training arguments with overwrite_output_dir=True
training_args = TrainingArguments(
    output_dir="./medical_llama3_finetuned_en",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    push_to_hub=True,
    hub_model_id="duy0204/medical-llama3-8b-sft-en",
    overwrite_output_dir=True,  # Fix: Overwrite old folder to save disk space
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)


print("Starting fine-tuning (3 epochs)...")
trainer.train()


trainer.save_model("./medical_llama3_finetuned_en_final")
trainer.push_to_hub()


print("Fine-tuning completed! Quick test:")
inputs = tokenizer("Question: A patient has sudden chest pain. Complex CoT: ", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
