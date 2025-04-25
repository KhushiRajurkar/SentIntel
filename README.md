# Sentiment Analysis with GoEmotions (RoBERTa + OpenVINO)

A fine-tuned RoBERTa-based model for detecting up to 28 different emotions in text, using the [GoEmotions (simplified)](https://huggingface.co/datasets/go_emotions) dataset.  
This repository contains:

- **Training notebook** (`GoEmotions_Multiclass_Trainer.ipynb`): shows how the model was trained with Intel XPU acceleration and Intel Extension for PyTorch.
- **Inference notebook** (`GoEmotionsInference.ipynb`): exports the model to ONNX, compiles it with OpenVINO, and demonstrates real-time inference.
- **Model files**:
  - `goemotions_openvino_demo/checkpoints/best` — best-performing checkpoint  
  - `goemotions_multilabel.onnx` — optimized ONNX export for inference

## Download Trained Models

> **Note:** The ONNX and best model files are large (~476 MB each). They have been hosted on Google Drive—click to download and place into `goemotions_multilabel_model/checkpoint-10854/`.

| File                      | Description                                  | Google Drive Link                                      |
| ------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| `model.safetensors`       | Trained weights (PyTorch safetensors format) | [model.safetensors](https://drive.google.com/file/d/16Lc5TCFJvjaJO5kqDZssFPVJTkUvWOdL/view?usp=sharing)  |
| `goemotions_multilabel.onnx` | ONNX‐export of the same model             | [goemotions_multilabel.onnx](https://drive.google.com/file/d/1YeIhk-UTK7PoIN2qekKU6lNcU7jbbjoK/view?usp=sharing) |


## Requirements
```txt
transformers>=4.30.0
datasets
numpy
torch
safetensors
openvino
jupyterlab
```
