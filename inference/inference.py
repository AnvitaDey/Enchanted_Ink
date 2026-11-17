import os
import sys
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "trocr-iam"
)

DEVICE = torch.device("cpu")


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading TrOCR processor...")

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

print("Loading fine-tuned TrOCR model...")

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)


# ==========================================
# HANDWRITING RECOGNITION
# ==========================================

def recognize_handwriting(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs.pixel_values.to(DEVICE)

    with torch.inference_mode():

        outputs = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
            
        ) #num_beams=5,no_repeat_ngram_size=3 , repetition_penalty=1.3, early_stopping=True, max_new_tokens=64

    generated_ids = outputs.sequences

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # Calculate sequence confidence from generated token probabilities
    transition_scores = model.compute_transition_scores(
        generated_ids,
        outputs.scores,
        normalize_logits=True
    )

    # Exclude the decoder start token, EOS and padding
    valid_scores = transition_scores[0]

    eos_id = model.config.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    generated_tokens = generated_ids[0, 1:]

    valid_mask = torch.ones_like(
        valid_scores, dtype=torch.bool
    )

    for token_id in [eos_id, pad_id]:
        if token_id is not None:
            valid_mask &= generated_tokens[:len(valid_scores)] != token_id

    valid_scores = valid_scores[valid_mask]

    if len(valid_scores) > 0:
        confidence = torch.exp(
            valid_scores.mean()
        ).item()
    else:
        confidence = 0.0

    return text, confidence


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python inference/trocr_inference.py "
            "test_images/handwriting.png"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    print("\nRecognizing handwriting...")

    text, confidence = recognize_handwriting(
        image_path
    )

    print("\n" + "=" * 45)
    print("RECOGNIZED TEXT")
    print("=" * 45)

    print(text)

    print("\nSequence confidence:", round(confidence * 100, 2), "%")

    print("=" * 45)