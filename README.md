# Sentiment Analysis with GoEmotions (RoBERTa + OpenVINO)

A fine-tuned RoBERTa-based model for detecting up to 28 different emotions in text, using the [GoEmotions (simplified)](https://huggingface.co/datasets/go_emotions) dataset.  
This repository contains:

- **Training notebook** (`GoEmotions_Multiclass_Trainer.ipynb`): shows how the model was trained with Intel XPU acceleration and Intel Extension for PyTorch.
- **Inference notebook** (`GoEmotionsInference.ipynb`): exports the model to ONNX, compiles it with OpenVINO, and demonstrates real-time inference.
- **Model files**:
  - `goemotions_openvino_demo/checkpoints/best` — best-performing checkpoint  
  - `goemotions_multilabel.onnx` — optimized ONNX export for inference

## Download Trained Models

> **Note:** The ONNX and best model files are large (~476 MB each). They have been hosted on Google Drive—click to download and place into `goemotions_openvino_demo/checkpoints/best`.

| File                      | Description                                  | Google Drive Link                                      |
| ------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| `model.safetensors`       | Trained weights (PyTorch safetensors format) | [model.safetensors](https://drive.google.com/file/d/16Lc5TCFJvjaJO5kqDZssFPVJTkUvWOdL/view?usp=sharing)  |
| `goemotions_multilabel.onnx` | ONNX‐export of the same model             | [goemotions_multilabel.onnx](https://drive.google.com/file/d/1YeIhk-UTK7PoIN2qekKU6lNcU7jbbjoK/view?usp=sharing) |


## Requirements
### Recommended Python version: 3.8–3.10
```txt
transformers>=4.30.0
datasets
numpy
torch
safetensors
openvino
jupyterlab
```
## Live Demo

Give the model a spin in your browser—no install needed!  
👉 [SentIntel on Hugging Face Spaces](https://huggingface.co/spaces/Kaiyeee/SentIntel)


## Build & Run Instructions

### 1. Clone the repository
```bash
git clone https://github.com/KhushiRajurkar/SentIntel.git
cd SentIntel
```
### 2. Set up Python environment
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the training notebook
  1. Launch Jupyter Lab
  ```bash
  jupyter lab
  ```

  2. Open `GoEmotions_Multiclass_Trainer.ipynb`

  3. Adjust hyperparameters if needed and execute all cells to fine-tune your model.
     
### 5. Download prebuilt model files
  1. Download the following and place them under the paths shown:
     ```bash
     SentIntel/
     └── goemotions_openvino_demo/
         └── checkpoints/
             └── best/
                 ├── model.safetensors
                 └── goemotions_multilabel.onnx

     ```

### 6. Export & convert for inference

  1. Load the best checkpoint from `goemotions_openvino_demo/checkpoints/best/` and rebuild the model.

  2. Export to ONNX by running the ONNX export cell.

  3. Compile with OpenVINO (IR → compiled model) by running the Model Optimizer & compile cells.

### 7. Run inference demo locally
```bash
from transformers import AutoTokenizer
from openvino.runtime import Core
import numpy as np

# 1) Load tokenizer + compiled OpenVINO IR
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
core      = Core()
compiled  = core.compile_model("goemotions_multilabel.xml", device_name="CPU")

# 2) Prepare inputs
texts  = ["I am so happy!", "This is bad..."]
tokens = tokenizer(texts,
                   padding="max_length",
                   truncation=True,
                   max_length=128,
                   return_tensors="np")

# 3) Inference
outs   = compiled([tokens["input_ids"], tokens["attention_mask"]])
logits = outs[compiled.output(0)]
probs  = 1 / (1 + np.exp(-logits))

# 4) Threshold & map back to human-readable emotions
THRESHOLD = 0.3
preds     = (probs > THRESHOLD).astype(int)

emotion_labels = [
  "admiration","amusement","anger","annoyance","approval","caring",
  "confusion","curiosity","desire","disappointment","disapproval",
  "disgust","embarrassment","excitement","fear","gratitude","grief",
  "joy","love","nervousness","optimism","pride","realization","relief",
  "remorse","sadness","surprise","neutral"
]

for i, text in enumerate(texts):
    fired = [emotion_labels[j] for j, v in enumerate(preds[i]) if v]
    print(f"{text!r} → {fired}")
```
