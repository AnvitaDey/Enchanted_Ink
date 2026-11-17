import os
import time
import re
import torch
import pandas as pd

from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm import tqdm


# =================================================
# CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "trocr-iam"
)

DEVICE = torch.device("cpu")

BATCH_SIZE = 4
MAX_LENGTH = 128

OUTPUT_CSV = os.path.join(
    BASE_DIR, "evaluation", "iam_predictions.csv"
)


# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading TrOCR processor...")

processor = TrOCRProcessor.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

print("Loading fine-tuned model...")

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model loaded!")
print("Device:", DEVICE)


# ==================================================
# LOAD IAM TEST DATASET
# ==================================================

print("\nDownloading/loading IAM test dataset...")

dataset = load_dataset(
    "Teklia/IAM-line",
    split="test"
)

print("Test samples:", len(dataset))


# ==================================================
# DATA COLLATOR
# ==================================================

def collate_fn(batch):

    images = [
        item["image"].convert("RGB")
        for item in batch
    ]

    texts = [
        item["text"]
        for item in batch
    ]

    pixel_values = processor(
        images=images,
        return_tensors="pt"
    ).pixel_values

    return pixel_values, texts


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ==================================================
# EVALUATION
# ==================================================

references = []
predictions = []

total_time = 0
total_images = 0

print("\nStarting evaluation...\n")

with torch.inference_mode():

    for pixel_values, texts in tqdm(loader):

        pixel_values = pixel_values.to(DEVICE)

        start = time.perf_counter()

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=MAX_LENGTH,
            num_beams=1,
            do_sample=False
        )

        elapsed = time.perf_counter() - start

        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        references.extend(texts)
        predictions.extend(decoded)

        total_time += elapsed
        total_images += len(texts)


# ==================================================
# METRIC CALCULATION
# ==================================================

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


normalized_refs = [
    normalize(x) for x in references
]

normalized_preds = [
    normalize(x) for x in predictions
]

CER = cer(normalized_refs, normalized_preds)
WER = wer(normalized_refs, normalized_preds)

character_accuracy = max(0, 1 - CER) * 100
word_accuracy = max(0, 1 - WER) * 100

exact_matches = sum(
    r == p
    for r, p in zip(normalized_refs, normalized_preds)
)

exact_accuracy = (
    exact_matches / len(references)
) * 100

average_time = total_time / total_images


# ==================================================
# SAVE PREDICTIONS
# ==================================================

df = pd.DataFrame({
    "ground_truth": references,
    "prediction": predictions,
    "correct": [
        r == p
        for r, p in zip(normalized_refs, normalized_preds)
    ]
})

df.to_csv(OUTPUT_CSV, index=False)


# ==================================================
# FINAL REPORT
# ==================================================

print("\n")
print("=" * 55)
print("       TrOCR IAM TEST EVALUATION RESULTS")
print("=" * 55)

print(f"\nTotal test images: {total_images}")

print("\n--- RECOGNITION METRICS ---")

print(f"CER:                  {CER * 100:.2f}%")
print(f"WER:                  {WER * 100:.2f}%")

print(f"Character Accuracy:   {character_accuracy:.2f}%")
print(f"Word Accuracy:        {word_accuracy:.2f}%")

print(f"Exact Sentence Match: {exact_accuracy:.2f}%")

print("\n--- PERFORMANCE METRICS ---")

print(f"Total inference time:  {total_time / 60:.2f} minutes")
print(f"Average time/image:   {average_time:.3f} seconds")

print("\n--- OUTPUT ---")

print("Predictions saved to:")
print(OUTPUT_CSV)

print("\n" + "=" * 55)
print("             EVALUATION COMPLETE")
print("=" * 55)


# ==================================================
# SHOW 10 SAMPLE PREDICTIONS
# ==================================================

print("\nSAMPLE PREDICTIONS:\n")

for i in range(min(10, len(references))):

    print(f"Example {i + 1}")
    print("Actual:    ", references[i])
    print("Predicted: ", predictions[i])
    print("Correct:   ", df.iloc[i]["correct"])
    print("-" * 45)