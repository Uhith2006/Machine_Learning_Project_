import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# ======== Load LoRA + Base BART =========
base_model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "lora_cti_adapter")
model.eval()

# ======== Load dataset containing SYSMON logs to generate summaries for ========
df = pd.read_csv("dataset.csv")   # or use a new file if you want

generated_summaries = []

for i, row in df.iterrows():
    prompt = "Summarize this SYSMON event as a CTI report:\n" + row["sysmon_log"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=4
        )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_summaries.append(summary)

df["generated_summary"] = generated_summaries
df.to_csv("generated_summaries.csv", index=False)

print("Summaries generated and saved to generated_summaries.csv")
