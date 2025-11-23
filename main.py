import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Load dataset
# ==========================================
df = pd.read_csv("dataset.csv")
dataset = Dataset.from_pandas(df)

# ==========================================
# 2. Load base model and tokenizer
# ==========================================
model_name = "facebook/bart-large"
print("Downloading / Loading BART model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ==========================================
# 3. Apply LoRA
# ==========================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(base_model, lora_config)
print("LoRA adapter added!")

# ==========================================
# 4. Tokenization function
# ==========================================
def preprocess(example):
    prompt = "Summarize this SYSMON event as a CTI report:\n" + example["sysmon_log"]
    model_inputs = tokenizer(prompt, max_length=512, truncation=True)

    labels = tokenizer(
        example["cti_summary"],
        max_length=128,
        truncation=True
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(preprocess)

# 🔥 remove columns that cannot be converted to tensors
dataset = dataset.remove_columns(["sysmon_log", "cti_summary", "id"])

# ==========================================
# 5. Data Collator (fixes padding)
# ==========================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# ==========================================
# 6. Training arguments
# ==========================================
training_args = TrainingArguments(
    output_dir="lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False
)

# ==========================================
# 7. Trainer
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# ==========================================
# 8. Train model
# ==========================================
print("Training started...")
trainer.train()
print("Training finished!")

# ==========================================
# 9. Save LoRA adapter
# ==========================================
model.save_pretrained("lora_cti_adapter")
tokenizer.save_pretrained("lora_cti_adapter")
print("LoRA adapter saved to: lora_cti_adapter/")
