import numpy as np
from glob import glob
import os
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset as HFDataset
from langchain.document_loaders import PyPDFLoader
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
import pickle

# Load later
with open('chunks.pkl', 'rb') as f:
    load_chunks = pickle.load(f)
chunks = load_chunks
# 4. Masukkan ke Dataset Hugging Face
dataset = HFDataset.from_dict({"text": chunks})

model_name = "google/gemma-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )
    # <== Ini WAJIB untuk CausalLM!
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


if not os.path.exists('mapped_chunks_causal.pkl'):
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"])
    with open('mapped_chunks_causal.pkl', 'wb') as f:
        pickle.dump(tokenized_dataset, f)
else:
    with open('mapped_chunks_causal.pkl', 'rb') as f:
        tokenized_dataset = pickle.load(f)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# Load model dengan quantization (hemat VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True  # atau load_in_8bit=True
)

# Siapkan model untuk training 4-bit
model = prepare_model_for_kbit_training(model)

# Konfigurasi LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Sesuaikan, ini aman untuk model transformer
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Untuk cek berapa parameter yang ditraining

# Setup Trainer
training_args = TrainingArguments(
    output_dir="./gemma3-pdf-indo-lora",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=200,
    learning_rate=5e-5,
    logging_steps=10,
    report_to="tensorboard",
    gradient_accumulation_steps=2,  # Sesuaikan jika perlu hemat VRAM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=True)
