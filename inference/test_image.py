import time
import csv
import zipfile
import io
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "trocr-iam"
ZIP_PATH = BASE_DIR / "test_images" / "handwriting_lines.zip"
OUTPUT_CSV = BASE_DIR / "test_images" / "predictions.csv"

DEVICE = torch.device("cpu")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# LOAD MODEL
print("Loading TrOCR processor...")
processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))

print("Loading fine-tuned TrOCR model...")
model = VisionEncoderDecoderModel.from_pretrained(
    str(MODEL_DIR),
    use_safetensors=True
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!\n")


def recognize(image):

    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    start = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    elapsed = time.time() - start

    text = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )[0]

    # Approximate sequence confidence
    token_probs = []
    sequence = outputs.sequences[0]

    for i, logits in enumerate(outputs.scores):
        token_id = sequence[i + 1]
        probs = torch.softmax(logits[0], dim=-1)
        token_probs.append(probs[token_id].item())

    if token_probs:
        confidence = (
            torch.tensor(token_probs)
            .clamp(min=1e-10)
            .log()
            .mean()
            .exp()
            .item() * 100
        )
    else:
        confidence = 0.0

    return text, confidence, elapsed


# PROCESS ZIP
if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

results = []

with zipfile.ZipFile("/home/anvita/Downloads/enchanted_ink/test_images/handwriting_lines.zip", "r") as archive:

    image_files = [
        f for f in archive.namelist()
        if f.lower().endswith(IMAGE_EXTENSIONS)
        and not Path(f).name.startswith(".")
    ]

    print(f"Found {len(image_files)} images.\n")

    for index, filename in enumerate(image_files, start=1):

        try:
            with archive.open(filename) as file:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")

            text, confidence, elapsed = recognize(image)

            results.append({
                "Image": filename,
                "Recognized Text": text,
                "Confidence (%)": round(confidence, 2),
                "Inference Time (s)": round(elapsed, 2)
            })

            print(f"\nIMAGE {index}/{len(image_files)}")
            print("-" * 50)
            print("File:", filename)
            print("Recognized:", text)
            print(f"Confidence: {confidence:.2f}%")
            print(f"Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


# SAVE RESULTS
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Image",
            "Recognized Text",
            "Confidence (%)",
            "Inference Time (s)"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 55)
print("HANDWRITING TEST COMPLETE")
print("=" * 55)
print(f"Images processed: {len(results)}")
print(f"Results saved to: {OUTPUT_CSV}")