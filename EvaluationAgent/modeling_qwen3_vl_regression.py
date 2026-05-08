import math
from typing import List

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3VLForConditionalGeneration


def load_backbone(model_name_or_path, torch_dtype):
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    return full_model.model


def enable_lora_gradient_flow(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is None:
        return

    def _make_inputs_require_grad(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
            return
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    item.requires_grad_(True)

    if not hasattr(input_embeddings, "_edit_hf_require_grad_hook"):
        hook_handle = input_embeddings.register_forward_hook(_make_inputs_require_grad)
        input_embeddings._edit_hf_require_grad_hook = hook_handle


class ScoreRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, dropout):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_dim2 = int(hidden_dim2)
        self.dropout = float(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim2),
            nn.SiLU(),
            nn.Dropout(self.dropout / 2.0),
            nn.Linear(self.hidden_dim2, 3),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.001)

    def forward(self, x):
        return self.mlp(x)

    def export_config(self):
        return {
            "hidden_size": self.input_dim,
            "head_hidden_size": self.hidden_dim,
            "head_hidden_size2": self.hidden_dim2,
            "dropout": self.dropout,
        }


class Qwen3VLLoRARegression(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        rank=64,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=None,
        precision="bf16",
        gradient_checkpointing=True,
    ):
        super().__init__()

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in dtype_map:
            raise ValueError(f"Unsupported precision: {precision}")

        base_model = load_backbone(model_name_or_path, torch_dtype=dtype_map[precision])

        if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        enable_lora_gradient_flow(base_model)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        self.vl_model = get_peft_model(base_model, lora_cfg)

        hidden_size = getattr(self.vl_model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.vl_model.config, "text_config"):
            hidden_size = getattr(self.vl_model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config")

        head_hidden_size = max(512, hidden_size // 2)
        head_hidden_size2 = max(128, int(math.ceil(hidden_size / 8.0)))
        self.score_head = ScoreRegressor(
            input_dim=hidden_size,
            hidden_dim=head_hidden_size,
            hidden_dim2=head_hidden_size2,
            dropout=0.10,
        )

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": False,
        }
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw

        outputs = self.vl_model(**model_inputs)
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_idx, seq_lens]
        pooled = pooled.to(self.score_head.mlp[0].weight.dtype)
        out = self.score_head(pooled).squeeze(-1)
        out = torch.sigmoid(out) * 100.0
        return out