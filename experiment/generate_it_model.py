from transformers import AutoTokenizer, AutoModelForCausalLM

# Load later
# with open('chunks.pkl', 'rb') as f:
#     load_chunks = pickle.load(f)
# chunks = load_chunks
# # 4. Masukkan ke Dataset Hugging Face
# dataset = HFDataset.from_dict({"text": chunks})

model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True  # atau load_in_8bit=True
)
