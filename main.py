#Cell1
# Install dependencies dulu
!pip install -q accelerate
!pip install -q transformers
!pip install -q bitsandbytes
!pip install -q torch torchvision torchaudio

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

# Setup 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model dan tokenizer
model_name = "mistralai/Mistral-7B-v0.1"  # atau "mistralai/Mistral-7B-Instruct-v0.1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

print("Model loaded successfully!")
print(f"Model size in memory: {model.get_memory_footprint() / 1024**3:.2f} GB")

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Test the model
prompt = "The future of AI is"
print("\nTesting the model:")
print("Prompt:", prompt)
print("\nGenerated text:")
result = pipe(prompt)
print(result[0]['generated_text'])

# Function untuk generate text dengan custom parameters
def generate_text(prompt, max_tokens=256, temperature=0.7):
    """
    Generate text using the loaded Mistral model
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Contoh penggunaan
print("\n" + "="*50)
print("Custom generation example:")
custom_prompt = "Explain quantum computing in simple terms:"
result = generate_text(custom_prompt, max_tokens=200)
print(result)


#cell2.optional
from google.colab import drive
drive.mount('/content/drive')


!mkdir -p /content/drive/MyDrive/models/mistral2
!cp -r /root/.cache/huggingface/hub/* /content/drive/MyDrive/models/mistral2/

custom_prompt = "apa perbedaan embedding, encoding, dan transformers dalam arsitektur ai:"
result = generate_text(custom_prompt, max_tokens=200)
print(result)