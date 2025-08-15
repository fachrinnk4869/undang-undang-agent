from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

base_model = "google/gemma-7b"
# ganti XXX dengan checkpoint terakhir
# adapter_dir = "./phi3-pdf-indo-lora/checkpoint-49830"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=False,
    device_map="auto",
    load_in_4bit=True
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=500,
    top_k=50,
    temperature=0.1,
)
# Load adapter (LoRA result)
llm = HuggingFacePipeline(pipeline=pipe)
print(llm.invoke("apa itu Hugging Face jawab pakai bahasa Indonesia!"))
# model = PeftModel.from_pretrained(model, adapter_dir)
# prompt = "Jelaskan isi Undang-Undang Nomor 51 Tahun 2009!"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
