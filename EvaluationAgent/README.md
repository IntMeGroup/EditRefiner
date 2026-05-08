## Files

- `modeling_qwen3_vl_8b_lora_regression.py`: model wrapper and score head
- `train_qwen3_vl_8b_full_lora_dp.py`: DDP training entry
- `config.env`: editable runtime config (model/data/GPU/hyper-params)
- `launch_train.sh`: one-command launcher
- `requirements.txt`: python dependencies

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Edit runtime config

Update `config.env` with your local paths and GPU IDs:

- `GPU_IDS=0,1` for 2 GPUs
- `TRAIN_CSV`, `TRAIN_IMG_ROOT`
- `VAL_CSV`, `VAL_IMG_ROOT`
- `TEST_CSV`, `TEST_IMG_ROOT`
- `RUN_ROOT`

## 3) Launch training

```bash
bash launch_train.sh
```

That command will:

- Parse `GPU_IDS` to infer DDP world size
- Set `CUDA_VISIBLE_DEVICES`
- Run `torchrun --nproc_per_node=<num_gpus>`
- Save checkpoints and logs under `RUN_ROOT`

Dataset split behavior is standard:

- train set: optimization updates
- validation set: in-training model selection
- test set: final evaluation only