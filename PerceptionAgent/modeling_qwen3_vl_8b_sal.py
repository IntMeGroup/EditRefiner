import math
from typing import List

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM
import torch.nn.functional as F
import math
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimpleDecoder2D(nn.Module):
    def __init__(
        self,
        in_dim=4096,          # image token dim
        global_dim=4096,      # global feature dim
        embed_dim=1024,
        dec_high=256,
        dec_mid=128,
        dec_low=64,
        out_channels=1,
        dropout=0.1,
        num_heads=8,
    ):
        super().__init__()

        # 🔹 projection
        self.img_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.global_proj = nn.Linear(global_dim, embed_dim)

        # 🔹 self-attention for fused tokens
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.dec1 = ConvBlock(embed_dim, dec_high, dropout=dropout)
        self.dec2 = ConvBlock(dec_high, dec_mid, dropout=dropout)
        self.dec3 = ConvBlock(dec_mid, dec_low, dropout=dropout)
        # 🔹 refine
        self.refine = ConvBlock(dec_low, dec_low, dropout=dropout)
        self.saliency_head = nn.Conv2d(dec_low, 1, kernel_size=1)

    def forward(self, v, x=None, image_size=None):
        """
        v: image tokens [B, N, C]
        x: global feature [B, Cg]
        """
        num_tokens = v.shape[1]
        H, W = image_size
        ratio = H / W

        # 初步估计 w
        w_est = math.sqrt(num_tokens / ratio)
        w = max(1, round(w_est))  # 先四舍五入为整数
        # print(w)

        # h 根据 w 计算
        h = num_tokens // w
        # print(h)

        # 如果整数乘积不够，则调整 w
        while h * w != num_tokens:
            w += 1
            h = num_tokens // w

        B, N, C = v.shape
        # 🔹 token → feature map
        v_map = v.view(B, h, w, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        v_map = self.img_proj(v_map)                     # [B, embed, H, W]

        # 🔹 flatten spatial tokens
        v_tokens = v_map.flatten(2).transpose(1, 2)      # [B, HW, embed]
        # 🔹 project global feature
        if x is not None:
            x_tokens = self.global_proj(x).unsqueeze(0) # [B,1,embed]
            fused_tokens = torch.cat([v_tokens, x_tokens], dim=1)  # [B, HW+1, embed]
        else:
            fused_tokens = v_tokens

        # 🔹 self-attention over fused tokens
        fused_tokens, _ = self.attn(fused_tokens, fused_tokens, fused_tokens)
        # 🔹 reshape back to feature map
        v_tokens = fused_tokens[:, :v_tokens.shape[1], :]  # keep only image tokens

        v_map = v_tokens.transpose(1, 2).reshape(B, -1, h, w)
        # 🔹 decode
        x = self.dec1(v_map)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)
        x = self.dec3(x)
        x = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)
        # 🔹 refine
        x = self.refine(x)
        x = torch.sigmoid(self.saliency_head(x))
        if image_size is not None:
            x = F.interpolate(x, size=image_size, mode="bilinear", align_corners=False)
        return x.squeeze(0)



class Qwen3VL8BLoRARegression(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        rank: int = 64,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: List[str] = None,
        num_labels: int = 3,
        precision: str = "bf16",
        gradient_checkpointing: bool = True,
        enable_lora: bool = True,   # 新增参数
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

        # 加载基础模型
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype_map[precision],
                trust_remote_code=True,
            )
        except ValueError:
            base_model = AutoModel.from_pretrained(
                model_name_or_path,
                dtype=dtype_map[precision],
                trust_remote_code=True,
            )

        if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        # 根据 enable_lora 决定是否加 LoRA
        if enable_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            self.vl_model = get_peft_model(base_model, lora_cfg)
        else:
            print("[INFO] LoRA disabled, using base model only")
            self.vl_model = base_model

        # 推理 decoder 仍然基于 hidden_size
        hidden_size = getattr(self.vl_model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.vl_model.config, "text_config"):
            hidden_size = getattr(self.vl_model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config")

        self.SalDecoder = SimpleDecoder2D().to(torch.bfloat16)

    def _init_saliency_head(self):
        nn.init.xavier_uniform_(self.saliency_head.weight)
        if self.saliency_head.bias is not None:
            nn.init.constant_(self.saliency_head.bias, 0.0)

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, image_size=None):
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
        # for i, h in enumerate(outputs.hidden_states):
        #     print(i, h.shape)
        # visual 27 层
        visual_hidden_states27 = outputs.hidden_states[26]
        visual_hidden_states1 = outputs.hidden_states[0]

        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        # seq_lens = attention_mask.sum(dim=1) - 1
        # seq_lens = torch.clamp(seq_lens, min=0)
        # batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        # pooled = hidden_states[batch_idx, seq_lens]
        # image_size = tuple(image_size[0].to(torch.int).tolist())  # -> (768, 768)
        # saliency = self.decoder(pooled, image_size)  # [B, 1, H, W]
        # saliency = saliency.squeeze(0)
        visual_hidden_states27 = visual_hidden_states27.squeeze(0)
        # print(visual_hidden_states27.shape)
        visual_hidden_states1 = visual_hidden_states1.squeeze(0)

        target_id = 151655

        # 创建 mask
        mask = (input_ids == target_id)  # shape: (B, seq_len), bool
        # 假设 hidden_states: (seq_len, 4096)
        # mask: (seq_len,)  1 表示有效，0 表示无效
        seq_idx = mask[0].nonzero(as_tuple=True)[0]  # tensor([0, 5, 7, ...])
        visual_features27 = visual_hidden_states27[seq_idx, :]
        print(visual_features27.shape)
        visual_features27 = visual_features27[visual_features27.shape[0] // 2:, :]

        # visual_features27 = visual_hidden_states27
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        fused_features = hidden_states[batch_idx, seq_lens]  # shape: (num_valid_tokens, 4096)

        visual_features27 = visual_features27.unsqueeze(0)
        # features = torch.cat([visual_features27, fused_features], dim=0)
        # seq_lens = attention_mask.sum(dim=1) - 1
        # seq_lens = torch.clamp(seq_lens, min=0)
        # batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        # pooled = hidden_states[batch_idx, seq_lens]
        image_size = tuple(image_size[0].to(torch.int).tolist())  # -> (768, 768)

        # print(image_size)
        hidden_states = hidden_states.squeeze(0)
        saliency = self.SalDecoder(visual_features27, hidden_states, image_size).to(torch.bfloat16)  # [B, 1, H, W]
        # saliency = saliency.squeeze(0)

        return saliency

