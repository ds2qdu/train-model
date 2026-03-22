"""
HF Trainer + Kubeflow TrainJob 검증용 학습 스크립트
모델: distilbert-base-uncased
태스크: SST-2 감성 분류 (positive/negative)
"""

import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default="/mnt/output/checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = use full dataset, >0 = subset for quick test")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    args = parse_args()

    print(f"=== HF TrainJob Validation ===")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"GPU available: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'N/A')}")

    # --------------------------------------------------
    # 1. 데이터셋 로드 (SST-2 from GLUE)
    # --------------------------------------------------
    dataset = load_dataset("glue", "sst2")

    if args.max_samples > 0:
        dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(args.max_samples // 4, len(dataset["validation"]))))
        print(f"Using subset: train={len(dataset['train'])}, val={len(dataset['validation'])}")

    # --------------------------------------------------
    # 2. 토크나이저 & 모델
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])

    # --------------------------------------------------
    # 3. TrainingArguments
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",  # W&B 등 외부 연동 없이 순수 검증
        fp16=True,         # GPU 사용 시 mixed precision
        dataloader_num_workers=2,
    )

    # --------------------------------------------------
    # 4. Trainer 실행
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    train_result = trainer.train()

    # --------------------------------------------------
    # 5. 결과 저장 & 출력
    # --------------------------------------------------
    metrics = train_result.metrics
    trainer.save_model(os.path.join(args.output_dir, "final"))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Eval
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("\n=== Training Complete ===")
    print(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Eval accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Eval F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
    print(f"Model saved to: {args.output_dir}/final")


if __name__ == "__main__":
    main()
