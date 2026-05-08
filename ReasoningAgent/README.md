## 1) Install dependencies

```bash
pip install 'ms-swift' -U
```

## 2) SFT
```bash
swift export \
    --model_type qwen3_vl \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset ./train.json \
    --val_dataset ./test.json
```

## 3) GRPO
```bash
swift rlhf \
    --rlhf_type grpo \
    --model_type qwen3_vl \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --external_plugins ./plugin/plugin.py \
    --reward_funcs external_code_reasoning_reward \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
   --dataset ./train.json \
   --val_dataset ./test.json
```

