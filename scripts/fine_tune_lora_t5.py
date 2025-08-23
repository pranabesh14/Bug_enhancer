import json
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


"""
Lightweight contextual training with LoRA on T5-base.
Input: vague summary; Target: structured description.
This is compute-friendly and good for a POC.
"""




def jsonl_to_hf(path):
	def gen():
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				if line.strip():
					j = json.loads(line)
					yield {"input": j["input"], "target": j["target"]}
	return load_dataset("json", data_files={"train": path}, split="train").map(lambda _: _)




if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("--train", required=True)
	ap.add_argument("--eval", required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--base", default="google/flan-t5-base")
	args = ap.parse_args()

	tokenizer = T5TokenizerFast.from_pretrained(args.base)
	model = T5ForConditionalGeneration.from_pretrained(args.base)

	lora_config = LoraConfig(
		task_type=TaskType.SEQ_2_SEQ_LM,
		r=8, lora_alpha=16, lora_dropout=0.1,
		target_modules=["q", "v"]
	)
	model = get_peft_model(model, lora_config)

	def preprocess(ex):
		x = tokenizer(
			[f"Enhance defect: {e}" for e in ex["input"]],
			max_length=256, truncation=True
		)
		y = tokenizer(text_target=ex["target"], max_length=512, truncation=True)
		x["labels"] = y["input_ids"]
		return x

	train_ds = load_dataset("json", data_files={"train": args.train})["train"].map(preprocess, batched=True, remove_columns=["input", "target"])
	eval_ds = load_dataset("json", data_files={"eval": args.eval})["eval"].map(preprocess, batched=True, remove_columns=["input", "target"])

	collator = DataCollatorForSeq2Seq(tokenizer, model=model)

	targs = TrainingArguments(
		output_dir=args.out,
		per_device_train_batch_size=4,
		per_device_eval_batch_size=4,
		gradient_accumulation_steps=4,
		learning_rate=2e-4,
		num_train_epochs=3,
		evaluation_strategy="epoch",
		save_strategy="epoch",
		logging_steps=50,
		fp16=True
	)

	trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
	trainer = Trainer(
		model=model,
		args=targs,
		train_dataset=train_ds,
		eval_dataset=eval_ds,
		data_collator=collator,
		predict_with_generate=True
	)
	trainer.train()
	trainer.save_model(args.out)
	tokenizer.save_pretrained(args.out)
	print("Saved LoRA model to", args.out)