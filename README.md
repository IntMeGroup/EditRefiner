# EditRefiner

📦 Installation

```bash
pip install 'ms-swift' -U
pip install diffusers==0.37.1 transformers==5.3.0
```

📥 Model Weights

You can download the model weights from the following link:
[EditRefiner](https://huggingface.co/TmpAccount/EditRefiner/tree/main)


⚡ Quick Start

```bash
python inference_main.py \
  --source_image "/path/to/source.jpg" \
  --edited_image "/path/to/edited.jpg" \
  --instruction "xxx" \
  --output_dir "./output" \
  --max_iters 4
```
