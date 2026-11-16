# Enchanted Ink — AI-Powered Infinite Whiteboard

**Enchanted Ink** is a modern, high-productivity AI whiteboard where users write, draw, think, and interact with artificial intelligence directly on an infinite canvas.

Instead of a conventional chatbot trapped in a sidebar, **the whiteboard is the product**. When you type or write an instruction containing `@board`, the AI interprets your request and canvas context, generating editable native whiteboard objects—block diagrams, interactive flowcharts, circuit schematics, mind maps, formula cards, and structured sticky notes—placed directly onto your board.

---

## Key Features

1. **Infinite Spatial Canvas (Fabric.js 5.3):**
   - Smooth viewport pan & zoom (15% to 400%) with mouse wheel, gestures, or Hand tool (`H` / Spacebar drag).
   - Dynamic dot-grid background synchronized with viewport transforms.
   - High-DPI crisp rendering on retina and 4K displays.
2. **Comprehensive Drawing & Visual Tools:**
   - **Select & Move (`V`):** Multi-object marquee selection, bounding boxes, 8 scale handles, rotation, and contextual styling toolbar.
   - **Pen / Freehand (`P`):** Smooth vector stroke drawing with pressure sensitivity and multi-color palette.
   - **Eraser (`E`):** Direct click or drag object removal.
   - **Text (`T`):** Inline editable typography with auto-wrapping.
   - **Sticky Notes (`S`):** Soft pastel cards with titles and formatted note text.
   - **Shapes (`R`, `O`):** Rounded rectangles and ellipses with clean borders and subtle drop shadows.
   - **Connectors & Arrows (`A`):** Directional lines with arrowheads and optional connector labels.
3. **The `@board` Native AI Engine:**
   - Floating command bar (`Ctrl+K` or bottom center) accepting natural language instructions.
   - **Context-Aware:** Analyzes existing notes, selected objects, and spatial viewport bounds.
   - **Native Object Synthesis:** Generates individual editable nodes, cards, and connectors—never a flat image or non-interactive bubble.
   - **Deterministic Spatial Layout:** Non-overlapping layouts for block diagrams, flowcharts, and mind maps.
   - **Undo AI Insert:** One-click removal of an entire AI insertion batch without disturbing earlier work.
4. **Uncertainty-Aware Handwriting (TrOCR):**
   - Diary and whiteboard handwriting are read locally by the configured TrOCR checkpoint, or Microsoft's pretrained handwritten baseline if no fine-tuned checkpoint is present.
   - Decoder token log-probabilities are exposed as raw scores. Only an Enchanted Ink checkpoint with a loaded validation calibration reports calibrated confidence.
   - Uncalibrated baseline or Google HWR text requires confirmation before it is used; no calibrated confidence is fabricated.
5. **Persistence & Export:**
   - Autosaves state to local storage and backend SQLite (`data/enchanted_diary.db`).
   - Multiple board management (create, switch, delete boards).
   - High-resolution **Export to PNG** and **Export to JSON**.
   - **Import JSON** to restore editable canvas boards.

---

## Quickstart & Setup

### 1. Requirements
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Safari, Edge)

### 2. Configure Environment
Copy the example environment file and configure your Gemini API Key:

```bash
cp .env.example .env
```

Edit `.env` and provide your Google Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, configure the API key directly in the web UI via the **AI Ready** settings modal).*

Diary and board Gemini models are configurable as `GEMINI_DIARY_MODEL` and `GEMINI_BOARD_MODEL`. The default is `gemini-3.8-flash`; board generation retries with `GEMINI_BOARD_FALLBACK_MODEL` (default `gemini-3.5-flash-lite`) if the primary model times out or is temporarily unavailable. Structured output uses JSON mode. For TrOCR, see [Training and deployment](#training-and-deployment). At startup, the service tries the configured Enchanted Ink checkpoint first, then the optional upstream handwritten baseline. The UI and status endpoint identify which model actually loaded.

### Use a local Ollama model

The diary reply and whiteboard generation features can use Ollama instead of Gemini. Install Ollama from [ollama.com](https://ollama.com/), start its local service, and download a model that fits your computer. For example:

```bash
ollama pull qwen2.5:7b
```

In the whiteboard's **AI Provider Settings**, select **Ollama (local)**, keep the server URL at `http://localhost:11434`, enter the exact installed model name (for example `qwen2.5:7b`), and save. The status indicator checks that the Ollama server is reachable and the selected model is installed. You can also configure `AI_PROVIDER=ollama`, `OLLAMA_BASE_URL`, and `OLLAMA_MODEL` in `.env`. Gemini remains the default provider unless changed.

Ollama runs response generation on your own machine and does not need a Gemini key. The handwriting reader is a separate component: handwritten input still needs TrOCR or the optional Google handwriting fallback; typed prompts work with Ollama alone. If you want all handwriting recognition to stay local, set `GEMINI_HWR_FALLBACK=false` and use the local TrOCR model or confirm the transcription manually. Model speed and quality depend on your hardware and the model you select.

### 3. Launch Application
Run the startup script:

```bash
./run_prototype.sh
```

Or run via the virtual environment directly:
```bash
./venv/bin/python backend/app.py
```

For an existing virtual environment, install dependencies with `./venv/bin/pip install -r requirements.txt`.

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## Example `@board` Interactions

Try typing these in the `@board` command bar:

- `@board Explain how a DC-DC buck converter works`
  *Generates a complete circuit block diagram (Input $\to$ Switch $\to$ Inductor $\to$ Capacitor $\to$ Load), key equations ($V_{out} \approx D \times V_{in}$), and operational phase sticky notes.*
- `@board create a flowchart for user login`
  *Generates sequential start/end ellipses, credential validation steps, decision branches, error retries, and connecting arrows.*
- `@board summarize these notes`
  *Extracts key takeaways from notes on your canvas and organizes them into categorized sticky notes.*
- `@board create a mind map for deep learning architectures`
  *Creates a central topic node with radiating branches for CNNs, Transformers, and RNNs.*
- `@board compare TrOCR vs CRNN`
  *Draws a structured comparison matrix with architecture, CER benchmarks, and trade-offs.*

---

## Keyboard Shortcuts

| Shortcut | Action |
|:---|:---|
| `V` | Select & Move Tool |
| `H` / `Space + Drag` | Pan / Hand Tool |
| `P` | Freehand Pen Drawing |
| `E` | Eraser Tool |
| `T` | Text Object |
| `S` | Sticky Note |
| `R` | Rectangle Shape |
| `O` | Ellipse Shape |
| `A` | Connector Arrow |
| `Ctrl + K` | Focus `@board` AI Command Bar |
| `Ctrl + Z` | Undo |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Redo |
| `Ctrl + D` | Duplicate Selected Objects |
| `Del` / `Backspace` | Delete Selected Objects |
| Mouse Wheel | Zoom in / Zoom out toward cursor |

---

## Architecture

```
Enchanted_Ink/
├── backend/
│   ├── app.py                      # FastAPI application & REST endpoints
│   ├── models/
│   │   ├── board_ai_service.py     # Provider abstraction & Gemini structured JSON generation
│   │   ├── board_storage.py        # SQLite persistent storage for named boards
│   │   ├── trocr_htr.py            # Local TrOCR inference and calibrated sequence confidence
│   │   ├── memory_engine.py        # Episodic memory engine
│   │   └── persona_engine.py       # Conversational persona engine
│   └── data/
│       └── enchanted_diary.db      # SQLite relational database
├── static/
│   ├── index.html                  # Modern whiteboard application UI
│   ├── css/
│   │   └── whiteboard.css          # Modern productivity design system & layout
│   ├── js/
│   │   └── whiteboard.js           # Fabric.js canvas controller, spatial engine & AI connector
│   └── vendor/
│       └── fabric.min.js           # Fabric.js 5.3 canvas library
├── training/trocr_iam.py           # IAM writer-split training/evaluation workflow
├── run_prototype.sh                # Application launch script
├── .env.example                    # Environment configuration template
└── README.md                       # Documentation
```

## Training and deployment

The repository includes TrOCR inference, training, calibration, and evaluation code. A local experimental checkpoint was trained from the Kaggle English Handwritten Line dataset; the weights and calibration are stored outside the repository at `$HOME/datasets/trocr-kaggle-lines`. That run measured a validation CER of 0.603 and test CER/WER of 0.614/0.716 on a 35-line test split. The dataset has no writer IDs, so the line-level split can leak writer style across splits; these are pipeline-experiment metrics, not research-quality or unseen-writer results. Do not commit downloaded data, model weights, or calibration files.

### Kaggle IAM handwritten forms

The Kaggle [IAM Handwritten Forms dataset](https://www.kaggle.com/datasets/naderabdelghany/iam-handwritten-forms-dataset) contains full-page scans with a printed reference paragraph and a handwritten copy. It is not the line-image archive used by the original `prepare` command. Use the `prepare-forms` command to crop handwritten rows and use Tesseract to read the printed reference. The OCR and estimated word-to-line alignment are **suggestions, not verified labels**; review every generated crop and transcript before training. The trainer refuses a manifest while any `alignment_review_required` field is `true`.

Install Tesseract on Ubuntu/Debian and then prepare the user's extracted Kaggle folder:

```bash
sudo apt install tesseract-ocr
cd ~/Documents/projects/Enchanted_Ink
./venv/bin/python training/trocr_iam.py prepare-forms \
  --forms-root "$HOME/Downloads/archive" \
  --output-dir "$HOME/datasets/iam-form-lines" \
  --manifest "$HOME/datasets/iam-forms.csv" \
  --limit 20
```

`--limit 20` makes a small pilot first. Open `iam-forms.csv` and its image paths in the output folder, correct any OCR text and line assignments, and set `alignment_review_required` to `false` for every reviewed row. If the pilot looks usable, rerun without `--limit 20` to prepare all forms, then review the resulting manifest before training.

For manifests prepared with an earlier version of this script, first make a corrected copy of the writer IDs (the IAM form filename `a01-000u.png` has writer ID `000u`, not form type `a01`):

```bash
./venv/bin/python training/trocr_iam.py repair-form-writers \
  --manifest "$HOME/datasets/iam-forms.csv" \
  --output "$HOME/datasets/iam-forms-corrected.csv"
```

Review and train using the corrected CSV. The training code also derives writer IDs from `source_form` when a form manifest is passed, to prevent stale writer IDs from leaking writers across splits.

Train only after all alignment flags have been reviewed:

```bash
./venv/bin/python training/trocr_iam.py train \
  --manifest "$HOME/datasets/iam-forms.csv" \
  --output "$HOME/datasets/trocr-output" \
  --base-model microsoft/trocr-small-stage1
```

### Kaggle English Handwritten Line dataset

The Kaggle [English handwritten line dataset](https://www.kaggle.com/datasets/sushant097/english-handwritten-line-dataset) contains line images under `self_lines/` and labels in `lines.txt`. Prepare it with the regular `prepare` command, not `prepare-forms`:

```bash
mkdir -p "$HOME/datasets"
./venv/bin/python training/trocr_iam.py prepare \
  --iam-root "$HOME/Downloads/archive" \
  --manifest "$HOME/datasets/kaggle-english-lines.csv"
```

This dataset does not identify writers in its line IDs. For a line-level experiment, train with the explicit `--allow-line-split` option:

```bash
./venv/bin/python training/trocr_iam.py train \
  --manifest "$HOME/datasets/kaggle-english-lines.csv" \
  --output "$HOME/datasets/trocr-kaggle-lines" \
  --base-model microsoft/trocr-small-stage1 \
  --allow-line-split
```

The script prints a warning when it uses this split. Since lines from the same unknown writer may appear in training, validation, and test sets, the resulting metrics do not measure performance on unseen writers. The small dataset is suitable for trying the pipeline, not for claiming general handwriting accuracy. The IAM `writer_id` column is set to `unknown` rather than inferred from the filenames.

An experimental run has already been completed locally using this dataset: the best validation CER was 0.603, test CER was 0.614, and test WER was 0.716 across 35 held-out lines. Temperature calibration was also generated from the 33-line validation split. The checkpoint and reports are at `$HOME/datasets/trocr-kaggle-lines` on the training machine and are not included in the repository. Treat both the scores and calibration as preliminary because of the small data and non-writer-disjoint split.

The training script loads the TrOCR tokenizer with `use_fast=False`. This uses the model's SentencePiece tokenizer directly and avoids a slow-tokenizer conversion that can fail when the installed `sentencepiece` generated protobuf bindings are newer than the environment's protobuf runtime.

### Original IAM line-image dataset

The original IAM training archives also work with `prepare`. They require an extracted IAM directory containing `lines.txt` and line images; `/content/iam` is a Google Colab path, not a local Linux path. On Linux, locate the annotation file after extraction:

```bash
find "$HOME/Downloads" "$HOME/datasets" -type f -name lines.txt -print 2>/dev/null
```

Set `IAM_ROOT` to the extracted directory containing that file, then run preparation and training with `&&` so training cannot start if preparation failed:

```bash
IAM_ROOT="$HOME/datasets/iam" # Replace with the actual extracted directory
./venv/bin/python training/trocr_iam.py prepare \
  --iam-root "$IAM_ROOT" \
  --manifest "$HOME/datasets/iam-lines.csv" &&
./venv/bin/python training/trocr_iam.py train \
  --manifest "$HOME/datasets/iam-lines.csv" \
  --output "$HOME/datasets/trocr-output" \
  --base-model microsoft/trocr-small-stage1
```

For Google Colab, upload or mount the project and dataset into the runtime, install a CUDA-compatible PyTorch build and `requirements-htr.txt`, and use `/content` paths only inside Colab. The manifest contains image path, transcript, and writer ID. Verify the sample count and images before training. The warning that `encoder.pooler.dense` weights were newly initialized is expected: that unused pooling layer is not part of TrOCR's image-to-text prediction path. The default `small-stage1` checkpoint avoids starting from Microsoft's `small-handwritten`, which is already fine-tuned on IAM and would contaminate writer-held-out evaluation. Splitting is writer-disjoint (80/10/10); the script evaluates a pretrained baseline, fine-tunes with AdamW, gradient accumulation, mixed precision when CUDA is available, checkpoint resume and validation CER early stopping. It fits sequence-score temperature on validation data, evaluates CER/WER/ECE and risk/coverage on the untouched test split, and writes evaluation plots plus `calibration.json`.

Copy the saved model directory and `calibration.json` to a local path outside Git, then set:

```ini
TROCR_CHECKPOINT=/absolute/path/to/trocr-output
TROCR_CALIBRATION=/absolute/path/to/trocr-output/calibration.json
HTR_ACCEPTANCE_THRESHOLD=0.72
```

Install CPU PyTorch and `requirements-htr.txt` in the same environment used to run the app:

```bash
./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements-htr.txt
```

At startup the service loads the configured local checkpoint first. If it is absent or fails to load and `TROCR_ALLOW_BASELINE_FALLBACK=true`, it downloads/caches `TROCR_BASELINE_MODEL` from Hugging Face (default `microsoft/trocr-small-handwritten`). The Hugging Face cache is reused on later starts. This is an upstream IAM-trained baseline, not an Enchanted Ink checkpoint; its raw token scores are not calibrated. With `GEMINI_HWR_FALLBACK=false`, handwriting images stay local and all baseline text—including `@board` commands—requires confirmation before use. Enabling `GEMINI_HWR_FALLBACK=true` permits external handwriting recognition through Google Gemini Vision using `GEMINI_API_KEY`; that service's output also requires confirmation. Gemini board generation remains available independently through `/api/board/command`. HTR status and the recognition panel identify the active recognizer. The local model is line-oriented and is not a research evaluation.

### Implemented status

- Implemented: local checkpoint loading, shared processor preprocessing, token log-probability sequence scoring, explicit empty/error states, validation calibration file loading, confidence threshold clarification, automatic debounced pen-stroke collection and crop rasterization, constrained `@board` trigger extraction, editable confirmation/retry UI, typed/handwritten command routing, visible Gemini/fallback status, and safeguards against dissolving unrelated selected objects.
- Implemented and exercised: IAM/Kaggle manifest preparation, writer-disjoint splits when writer IDs exist, explicit line-level splits when they do not, fine-tuning, validation temperature calibration, and CER/WER evaluation.
- Completed locally but not distributed: an experimental TrOCR checkpoint and calibration trained from the 336-line Kaggle English Handwritten Line dataset. Those weights are not tracked in this checkout; the default `./models/trocr-iam` checkpoint is absent, so the current setup uses Microsoft's pretrained baseline. The experiment's test CER/WER were 0.614/0.716 on a small, line-level split, so it is not considered production-quality.
- The separate Kaggle IAM forms manifest still needs transcript/alignment review before its rows can be used for training. When no local checkpoint loads, the upstream IAM-trained handwritten baseline may recognize pen ink and is clearly labeled. Explicit `@board` commands run automatically; other uncalibrated text asks for correction. The UI preserves strokes and offers manual transcription if no model loads. The recognizer receives a crop of recent nearby strokes as one image, so multi-line commands may need user confirmation. Offline diary replies are local templates and rule-based `@board` content is limited and labeled as fallback.

## API behavior

- `GET /api/v1/htr/status`: checkpoint/calibration availability and acceptance threshold.
- `POST /api/v1/process-turn`: TrOCR transcription, clarification gate, memory retrieval, persona reply and persistence. Low confidence never reaches memory or response generation until the user confirms/corrects.
- `POST /api/board/htr`: local TrOCR transcription with optional Google Gemini Vision fallback (disabled by default in `.env.example`) for a crop built from recent pen strokes or selected canvas handwriting; response includes request ID, confidence/calibration status, and extracted command fields. Uncalibrated results require confirmation before board execution.
- `POST /api/board/command`: provider-labelled editable canvas objects. Requests must begin with `@board`; the complete board context is supplied for the answer. Invalid structured output returns an HTTP error without inserting partial objects; offline fallback responses preserve the request rather than inserting an unrelated template.

Automatic whiteboard recognition waits for `HTR_INACTIVITY_MS` (default 12000 ms) after the latest pen stroke and will not start while the pen is down. It keeps strokes together across words and lines. If a pause comes immediately after `@board`, it holds that prefix for up to 90 seconds and combines it with the rest of the question. Ordinary handwriting never triggers a board answer; only a recognized `@board` request does. Handwritten questions use the complete board context so the answer can connect information across notes and diagrams. On successful generation, only the strokes in that recognized command group are removed; Undo AI Insert restores those strokes and removes only that generated batch.

### Try handwritten `@board` locally

Open `http://localhost:8000`, choose the pen, and write `@board create a flowchart for user login` on the canvas. Pause for the configured inactivity interval, then confirm the uncalibrated transcription. After confirmation, the same `/api/board/command` route used for typed commands generates editable objects. Ordinary notes do not trigger commands. If no model loads, the panel reports the cause, preserves the ink, and offers retry or manual transcription. If Gemini is unavailable for board generation, the rule-based response is visibly identified as fallback.
