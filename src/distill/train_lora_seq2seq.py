import argparse, os
from datasets import DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model, TaskType

def tokenize_batch(examples, tok, max_source=1024, max_target=256):
    model_inputs = tok(examples["input"], max_length=max_source, truncation=True)
    with tok.as_target_tokenizer():
        labels = tok(examples["target"], max_length=max_target, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto")
    lcfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=["q","v"]
    )
    model = get_peft_model(base, lcfg)
    ds: DatasetDict = DatasetDict.load_from_disk(args.data)
    tok_fn = lambda ex: tokenize_batch(ex, tok, args.max_source, args.max_target)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=list(ds["train"].features))
    collator = DataCollatorForSeq2Seq(tok, model=model)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_target,
        fp16=True,
        gradient_accumulation_steps=args.grad_accum,
        report_to="none",
        max_steps=args.max_steps if args.max_steps > 0 else None
    )

    trainer = Seq2SeqTrainer(
        model=model, args=train_args,
        train_dataset=ds_tok["train"], eval_dataset=ds_tok["dev"],
        data_collator=collator, tokenizer=tok
    )
    trainer.train()
    model.save_pretrained(os.path.join(args.out_dir, "lora_student"))
    tok.save_pretrained(os.path.join(args.out_dir, "lora_student"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--data", required=True, help="DatasetDict saved with .save_to_disk()")
    ap.add_argument("--out_dir", default="outputs/distill")
    ap.add_argument("--max_source", type=int, default=1024)
    ap.add_argument("--max_target", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="Override epochs with a tiny-run (e.g., 20)")
    args = ap.parse_args()
    main(args)
