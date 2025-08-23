import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast


model_path = os.getenv("HF_MODEL_PATH", "models/t5-lora-domain")


model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5TokenizerFast.from_pretrained(model_path)


prompt = "Enhance defect: payment page error after clicking pay"
ids = tokenizer(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=300)
print(tokenizer.decode(out[0], skip_special_tokens=True))