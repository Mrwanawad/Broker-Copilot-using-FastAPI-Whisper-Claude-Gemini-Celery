"""Fine-tune Whisper on Egyptian-Arabic call audio (LoRA on the decoder/encoder attention).

Data: a folder of 16 kHz clips with a manifest.jsonl of {"audio": "clip.wav", "text": "..."}, built from
broker-audited call segments (consented recordings only). Hold out whole calls, not segments, for eval.

    uv pip install "transformers>=4.46" "datasets>=3.0" "peft>=0.13" "accelerate>=1.0" soundfile evaluate
    python scripts/finetune_whisper.py --data data/asr_train --base openai/whisper-small --out models/whisper-eg

Then convert for faster-whisper:
    ct2-transformers-converter --model models/whisper-eg/merged --output_dir models/whisper-eg-ct2 --quantization int8
and set BC_WHISPER_MODEL=models/whisper-eg-ct2. Compare WER/CER with scripts/eval_audio.py before switching.

On a 4 GB GPU use whisper-small with fp16 and batch 4 + gradient accumulation; medium/large-v3 need more VRAM.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


def load_split(data_dir: Path, eval_fraction: float = 0.1):
    from datasets import Audio, Dataset

    rows = [json.loads(line) for line in (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["audio"] = str(data_dir / r["audio"])
    # split by call id so segments of one call never land on both sides
    calls = sorted({r.get("call_id", r["audio"]) for r in rows})
    eval_calls = set(calls[: max(1, int(len(calls) * eval_fraction))])
    train = [r for r in rows if r.get("call_id", r["audio"]) not in eval_calls]
    test = [r for r in rows if r.get("call_id", r["audio"]) in eval_calls]
    cast = Audio(sampling_rate=16_000)
    return (Dataset.from_list(train).cast_column("audio", cast), Dataset.from_list(test).cast_column("audio", cast))


@dataclass
class Collator:
    processor: object

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", default="openai/whisper-small")
    parser.add_argument("--out", type=Path, default=Path("models/whisper-eg"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    import jiwer
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    from app.text.normalize import normalize_search

    processor = WhisperProcessor.from_pretrained(args.base, language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model.print_trainable_parameters()

    def prep(row):
        audio = row["audio"]
        row["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16_000).input_features[0]
        row["labels"] = processor.tokenizer(row["text"]).input_ids
        return row

    train, test = load_split(args.data)
    train = train.map(prep, remove_columns=train.column_names)
    test = test.map(prep, remove_columns=test.column_names)

    def metrics(pred):
        ids = pred.label_ids
        ids[ids == -100] = processor.tokenizer.pad_token_id
        hyp = [normalize_search(t) for t in processor.batch_decode(pred.predictions, skip_special_tokens=True)]
        ref = [normalize_search(t) for t in processor.batch_decode(ids, skip_special_tokens=True)]
        return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=0.1, fp16=True,
        eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True, generation_max_length=225,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=test,
                             data_collator=Collator(processor), compute_metrics=metrics)
    trainer.train()
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out / "merged")
    processor.save_pretrained(args.out / "merged")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
