## Files

- `modeling_qwen3_vl_8b_lora_sal.py`: model wrapper and saliency head
- `train_qwen3_vl_8b_lora_dp.py`: DDP training entry
- `config.env`: editable runtime config (model/data/GPU/hyper-params)
- `launch_train.sh`: one-command launcher
- `requirements.txt`: python dependencies

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Launch training

```bash
bash launch_train.sh
```
