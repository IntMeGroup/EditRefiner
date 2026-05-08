import cv2
import argparse
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel, PeftConfig
from modeling_qwen3_vl_8b_sal import Qwen3VL8BLoRARegression
import json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InferConfig:
    model_dir: str                  # 训练保存的 checkpoint 目录
    model_dir2: str
    output_dir: str                 # 推理结果保存目录
    # --- single mode ---
    source_image: Optional[str]
    target_image: Optional[str]
    caption: Optional[str]
    # --- batch mode ---
    test_csv: Optional[str]
    test_img_root: Optional[str]
    # --- common ---
    batch_size: int
    num_workers: int
    prefetch_factor: int
    max_pixels_per_image: int
    max_length: int
    precision: str                  # fp32 / fp16 / bf16
    # --- LoRA (must match training) ---
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    # --- output options ---
    save_npy: bool                  # 保存原始 float32 numpy 数组
    save_heatmap: bool              # 保存伪彩色热力图 PNG
    save_overlay: bool              # 保存叠加在 target 上的热力图


def parse_args() -> InferConfig:
    p = argparse.ArgumentParser(description="Distortion Saliency Map Inference")

    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to first saved perception checkpoint directory")
    p.add_argument("--model_dir2", type=str, required=True,
                   help="Path to second saved perception checkpoint directory")

    p.add_argument("--output_dir", type=str, default="./infer_outputs")

    # single image mode
    p.add_argument("--source_image", type=str, default=None)
    p.add_argument("--target_image", type=str, default=None)
    p.add_argument("--caption", type=str, default="")

    # batch CSV mode
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--test_img_root", type=str, default="")

    # data
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--max_pixels_per_image", type=int, default=1048576)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")

    # LoRA (keep consistent with training)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # output options
    p.add_argument("--save_npy", action="store_true", default=True)
    p.add_argument("--save_heatmap", action="store_true", default=True)
    p.add_argument("--save_overlay", action="store_true", default=True)

    a = p.parse_args()

    # 验证模式
    single_mode = a.source_image is not None and a.target_image is not None
    batch_mode = a.test_csv is not None
    if not single_mode and not batch_mode:
        p.error("Must specify either (--source_image + --target_image) or --test_csv")

    return InferConfig(
        model_dir=a.model_dir,
        model_dir2=a.model_dir2,
        output_dir=a.output_dir,
        source_image=a.source_image,
        target_image=a.target_image,
        caption=a.caption,
        test_csv=a.test_csv,
        test_img_root=a.test_img_root,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        prefetch_factor=a.prefetch_factor,
        max_pixels_per_image=a.max_pixels_per_image,
        max_length=a.max_length,
        precision=a.precision,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        lora_target_modules=[x.strip() for x in a.lora_target_modules.split(",") if x.strip()],
        save_npy=a.save_npy,
        save_heatmap=a.save_heatmap,
        save_overlay=a.save_overlay,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gb18030", "latin1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise RuntimeError(f"Cannot decode CSV: {csv_path}")


def load_and_limit_image(image_path: str, max_pixels: int) -> Image.Image:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / float(w * h))
            new_w = max(16, (max(16, int(round(w * scale))) // 16) * 16)
            new_h = max(16, (max(16, int(round(h * scale))) // 16) * 16)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img.copy()   # copy() 确保 file handle 关闭后图像仍可用


def resolve_path(rel_path: str, root_dir: str) -> str:
    """与训练代码保持一致的路径解析逻辑"""
    direct = os.path.join(root_dir, rel_path)
    if os.path.exists(direct):
        return direct
    parts = str(rel_path).split("/")
    for idx, part in enumerate(parts[:-1]):
        for repl in (part.replace("_", "-"), part.replace("-", "_")):
            if repl == part:
                continue
            candidate = parts.copy()
            candidate[idx] = repl
            cand_path = os.path.join(root_dir, "/".join(candidate))
            if os.path.exists(cand_path):
                return cand_path
    raise FileNotFoundError(f"Image not found: {direct}")


def build_prompt(caption: str) -> str:
    return (
        "Task: identify and localize visual distortions in the Target image with respect to the Source image under the given editing instruction. "
        f"Editing instruction: '{caption}'. "
        "Source is the original reference image and Target is the edited result image. "
        "Analyze the Target image and detect regions where visual quality is degraded, including artifacts, unnatural textures, color inconsistency, or structural errors introduced by editing. "
        "Focus on highlighting distortion regions while ignoring well-preserved areas. "
        "Output a dense distortion saliency map aligned with the Target image, where higher values indicate more severe distortions. "
        "Distortion Saliency Map:"
    )
def build_prompt2(caption: str) -> str:
    return (
        "Task: find where misalign the editing instruction occur in the Target image compared to the Source image. "
        f"Editing instruction: '{caption}'. "
        "The Source image is the original reference and the Target image is the edited result. "
        "Identify regions in the Target image that not properly executed or regions incorrectly modified despite not requiring edits. "
        "Only focus on regions with distortions."
    )


# ---------------------------------------------------------------------------
# Colormap: apply jet-like pseudo-color heatmap
# ---------------------------------------------------------------------------

def apply_heatmap(sal_map: np.ndarray) -> Image.Image:
    """
    sal_map: float32 [H, W], values in [0,1]
    Returns: RGB PIL Image with jet colormap
    """
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap("jet")
        rgba = colormap(sal_map)          # [H, W, 4]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb)
    except ImportError:
        # fallback: grayscale if matplotlib not available
        gray = (sal_map * 255).astype(np.uint8)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        gray = (gray ** 0.5) * 255  # gamma 提亮暗部
        gray = gray.astype(np.uint8)
        print("gray:", gray)
        return Image.fromarray(gray, mode="L").convert("RGB")


def apply_overlay(target_img: Image.Image, sal_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay heatmap on target image.
    sal_map: float32 [H, W], will be resized to target_img size if needed
    """
    tw, th = target_img.size
    if sal_map.shape != (th, tw):
        sal_tensor = torch.from_numpy(sal_map).unsqueeze(0).unsqueeze(0)
        sal_tensor = F.interpolate(sal_tensor, size=(th, tw), mode="bilinear", align_corners=False)
        sal_map = sal_tensor.squeeze().numpy()

    heat_img = apply_heatmap(sal_map).resize((tw, th), Image.Resampling.BILINEAR)
    return Image.blend(target_img.convert("RGB"), heat_img, alpha=alpha)


# ---------------------------------------------------------------------------
# Dataset for batch inference
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Supports:
      1. CSV-based batch mode
      2. Single pair mode (pass a single-row DataFrame)
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, processor, cfg: InferConfig, sal_type: bool):
        self.data = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.processor = processor
        self.cfg = cfg
        self.sal_type = sal_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---------- paths ----------
        row_root = str(row.get("__root_dir", self.root_dir))
        image_rel = str(row["image_path"])
        source_path = resolve_path(image_rel, row_root)
        target_path = resolve_path(image_rel, row_root)
        caption = str(row.get("caption", "")).strip()
        sample_id = str(row.get("sample_id", idx))

        # ---------- images ----------
        source_image = load_and_limit_image(source_path, self.cfg.max_pixels_per_image)
        target_image = load_and_limit_image(target_path, self.cfg.max_pixels_per_image)
        orig_w, orig_h = target_image.size   # 原始尺寸，用于最终 resize 输出

        if self.sal_type:
            # ---------- tokenize ----------
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source_image},
                        # {"type": "image", "image": target_image},
                        {"type": "text", "text": build_prompt(caption)},
                    ],
                }
            ]
        else:
            # ---------- tokenize ----------
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source_image},
                        # {"type": "image", "image": target_image},
                        {"type": "text", "text": build_prompt2(caption)},
                    ],
                }
            ]


        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        out = {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            # target_size: [orig_H, orig_W] — 用于最终输出 resize
            "target_size":    torch.tensor([orig_h, orig_w], dtype=torch.int),
            "sample_id":      sample_id,
            "target_path":    target_path,
        }
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            out["image_grid_thw"] = inputs["image_grid_thw"]
        return out


def infer_collate_fn(pad_token_id: int):
    def _collate(batch):
        input_ids     = [b["input_ids"]      for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        out = {
            "input_ids":      pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "target_size":    torch.stack([b["target_size"]  for b in batch]),   # [B, 2]
            "sample_ids":     [b["sample_id"]   for b in batch],
            "target_paths":   [b["target_path"] for b in batch],
        }
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return out
    return _collate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_sal1(cfg: InferConfig, device: torch.device):
    """
    从 checkpoint 目录中加载 processor + model，并恢复 LoRA 适配器 + sal_head 权重。
    """
    print(f"[INFO] Loading processor from {cfg.model_dir}")
    processor = AutoProcessor.from_pretrained(cfg.model_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 检查是否存在 adapter_model.safetensors (未合并的 LoRA)
    adapter_path = os.path.join(cfg.model_dir, "adapter_model.safetensors")
    has_adapter = os.path.exists(adapter_path)

    if has_adapter:
        print(f"[INFO] Found LoRA adapter at {adapter_path}, loading with PEFT")

        # 首先需要确定基础模型路径
        # 检查是否有 peft_config.json 来获取基础模型路径
        peft_config_path = os.path.join(cfg.model_dir, "adapter_config.json")
        if os.path.exists(peft_config_path):
            peft_config = PeftConfig.from_pretrained(cfg.model_dir)
            base_model_path = peft_config.base_model_name_or_path
        else:
            # 如果没有配置文件，假设基础模型就在同一目录或使用原始路径
            base_model_path = cfg.model_dir

        print(f"[INFO] Loading base model from {base_model_path}")

        # 先创建基础模型（不带 LoRA）
        model = Qwen3VL8BLoRARegression(
            model_name_or_path=base_model_path,
            rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            num_labels=3,
            precision=cfg.precision,
            gradient_checkpointing=False,
            enable_lora=False,  # 先不启用 LoRA，后面手动加载
        )

        # 然后加载 LoRA 适配器
        print(f"[INFO] Loading LoRA adapter from {cfg.model_dir}")
        model.vl_model = PeftModel.from_pretrained(model.vl_model, cfg.model_dir)

    else:
        print(f"[INFO] Loading merged model (no adapter found) from {cfg.model_dir}")
        # 原有逻辑：假设已经合并
        model = Qwen3VL8BLoRARegression(
            model_name_or_path=cfg.model_dir,
            rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            num_labels=3,
            precision=cfg.precision,
            gradient_checkpointing=False,
        )

    # 加载 saliency head 权重
    sal_head_path = os.path.join(cfg.model_dir, "sal_head.pth")
    if os.path.exists(sal_head_path):
        print(f"[INFO] Loading sal_head from {sal_head_path}")
        sal_state = torch.load(sal_head_path, map_location="cpu")
        model.SalDecoder.load_state_dict(sal_state, strict=True)
    else:
        print(f"[WARN] sal_head.pth not found in {cfg.model_dir}, using randomly initialized head!")

    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully")
    return processor, model


def load_model_sal2(cfg: InferConfig, device: torch.device):
    """
    从 checkpoint 目录中加载 processor + model，并恢复 LoRA 适配器 + sal_head 权重。
    """
    print(f"[INFO] Loading processor from {cfg.model_dir2}")
    processor = AutoProcessor.from_pretrained(cfg.model_dir2, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 检查是否存在 adapter_model.safetensors (未合并的 LoRA)
    adapter_path = os.path.join(cfg.model_dir2, "adapter_model.safetensors")
    has_adapter = os.path.exists(adapter_path)

    if has_adapter:
        print(f"[INFO] Found LoRA adapter at {adapter_path}, loading with PEFT")

        # 首先需要确定基础模型路径
        # 检查是否有 peft_config.json 来获取基础模型路径
        peft_config_path = os.path.join(cfg.model_dir2, "adapter_config.json")
        if os.path.exists(peft_config_path):
            peft_config = PeftConfig.from_pretrained(cfg.model_dir2)
            base_model_path = peft_config.base_model_name_or_path
        else:
            # 如果没有配置文件，假设基础模型就在同一目录或使用原始路径
            base_model_path = cfg.model_dir2

        print(f"[INFO] Loading base model from {base_model_path}")

        # 先创建基础模型（不带 LoRA）
        model = Qwen3VL8BLoRARegression(
            model_name_or_path=base_model_path,
            rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            num_labels=3,
            precision=cfg.precision,
            gradient_checkpointing=False,
            enable_lora=False,  # 先不启用 LoRA，后面手动加载
        )

        # 然后加载 LoRA 适配器
        print(f"[INFO] Loading LoRA adapter from {cfg.model_dir2}")
        model.vl_model = PeftModel.from_pretrained(model.vl_model, cfg.model_dir2)

    else:
        print(f"[INFO] Loading merged model (no adapter found) from {cfg.model_dir2}")
        # 原有逻辑：假设已经合并
        model = Qwen3VL8BLoRARegression(
            model_name_or_path=cfg.model_dir2,
            rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            num_labels=3,
            precision=cfg.precision,
            gradient_checkpointing=False,
        )

    # 加载 saliency head 权重
    sal_head_path = os.path.join(cfg.model_dir2, "sal_head.pth")
    if os.path.exists(sal_head_path):
        print(f"[INFO] Loading sal_head from {sal_head_path}")
        sal_state = torch.load(sal_head_path, map_location="cpu")
        model.SalDecoder.load_state_dict(sal_state, strict=True)
    else:
        print(f"[WARN] sal_head.pth not found in {cfg.model_dir2}, using randomly initialized head!")

    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully")
    return processor, model


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    cfg: InferConfig,
    device: torch.device,
) -> List[dict]:
    """
    Returns list of dicts:
        {
            "sample_id": str,
            "target_path": str,
            "sal_map": np.ndarray [H, W] float32,   # resized to original target size
        }
    """
    use_amp = cfg.precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16

    results = []
    bar = tqdm(loader, desc="Inference")

    for batch in bar:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        target_sizes   = batch["target_size"].to(device, non_blocking=True)   # [B, 2]
        sample_ids     = batch["sample_ids"]
        target_paths   = batch["target_paths"]

        pixel_values   = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, non_blocking=True)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp_ctx:
            # logits: [B, H_feat, W_feat]  (model 内部会上采样至 image_size)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_size=target_sizes,
            )  # [B, H, W]

        # 归一化到 [0,1]
        logits_f = logits.float()                                  # [B, H, W]
        B = logits_f.shape[0]

        for i in range(B):
            sal = logits_f[i] / 255                                     # [H, W]
            orig_h, orig_w = int(target_sizes[i, 0]), int(target_sizes[i, 1])

            # 如果 model 输出尺寸与原始尺寸不同，再做一次精确 resize
            if sal.shape[0] != orig_h or sal.shape[1] != orig_w:
                sal = F.interpolate(
                    sal.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            # min-max normalize to [0,1] for visualization
            sal_min, sal_max = sal.min(), sal.max()
            if (sal_max - sal_min) > 1e-6:
                sal_norm = (sal - sal_min) / (sal_max - sal_min)
            else:
                sal_norm = torch.zeros_like(sal)

            results.append({
                "sample_id":   sample_ids[i],
                "target_path": target_paths[i],
                "sal_map":     sal_norm.cpu().numpy().astype(np.float32),
            })

    return results

def saliency_to_mask(sal, thresh=0.5):
    """
    sal: [H, W] float32 in [0,1]
    return: binary mask uint8 {0,1}
    """
    mask = (sal >= thresh).astype(np.uint8)

    # 可选：形态学去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def fuse_saliency(sal1, sal2, mode="union"):
    if mode == "union":
        fused = np.maximum(sal1, sal2)
    elif mode == "avg":
        fused = (sal1 + sal2) / 2
    elif mode == "intersect":
        fused = np.minimum(sal1, sal2)
    else:
        raise ValueError
    return fused

def mask_to_bboxes(mask, min_area=50):
    """
    mask: [H, W] {0,1}
    return: list of (x1, y1, x2, y2)
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        bboxes.append((x, y, x + w, y + h))

    return bboxes

def merge_saliency_to_regions(sal1, sal2, thresh=0.5):
    # 1. fuse
    fused = fuse_saliency(sal1, sal2, mode="union")

    # 2. threshold → mask
    mask = saliency_to_mask(fused, thresh)

    # 3. bbox
    bboxes = mask_to_bboxes(mask)

    return fused, mask, bboxes

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")


    # -------- load model --------
    processor, model = load_model_sal1(cfg, device)
    collate_fn = infer_collate_fn(processor.tokenizer.pad_token_id)

    # -------- build dataset --------
    if cfg.test_csv is not None:
        # Batch CSV mode
        print(f"[INFO] Batch mode: loading CSV {cfg.test_csv}")
        df = read_csv_with_fallback(cfg.test_csv)
        df["__root_dir"] = cfg.test_img_root
        # 过滤不存在的文件
        valid_mask = df["image_path"].astype(str).map(
            lambda p: os.path.exists(os.path.join(cfg.test_img_root, p))
        )
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            print(f"[WARN] Dropped {dropped} samples with missing images")
        df = df.loc[valid_mask].reset_index(drop=True)
        # 添加 sample_id
        if "sample_id" not in df.columns:
            df["sample_id"] = df.index.astype(str)
        print(f"[INFO] Total inference samples: {len(df)}")

        dataset = InferenceDataset(df, cfg.test_img_root, processor, cfg, sal_type=1)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )

    else:
        # Single image mode
        print(f"[INFO] Single mode: {cfg.target_image}")
        single_df = pd.DataFrame([{
            "image_path":  os.path.relpath(cfg.target_image, cfg.test_img_root or "/"),
            "caption":     cfg.caption or "",
            "sample_id":   "0",
            "__root_dir":  os.path.dirname(os.path.abspath(cfg.target_image)),
        }])
        # 对 single mode 直接用绝对路径覆盖 image_path
        single_df["image_path"] = os.path.basename(cfg.target_image)
        single_df["__root_dir"] = os.path.dirname(os.path.abspath(cfg.target_image))

        dataset = InferenceDataset(single_df, single_df["__root_dir"].iloc[0], processor, cfg, sal_type=1)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # -------- run inference --------
    results = run_inference(model, loader, cfg, device)

    # -------- free model --------
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

 # -------- load model --------
    processor, model = load_model_sal2(cfg, device)
    collate_fn = infer_collate_fn(processor.tokenizer.pad_token_id)

    # -------- build dataset --------
    if cfg.test_csv is not None:
        # Batch CSV mode
        print(f"[INFO] Batch mode: loading CSV {cfg.test_csv}")
        df = read_csv_with_fallback(cfg.test_csv)
        df["__root_dir"] = cfg.test_img_root
        # 过滤不存在的文件
        valid_mask = df["image_path"].astype(str).map(
            lambda p: os.path.exists(os.path.join(cfg.test_img_root, p))
        )
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            print(f"[WARN] Dropped {dropped} samples with missing images")
        df = df.loc[valid_mask].reset_index(drop=True)
        # 添加 sample_id
        if "sample_id" not in df.columns:
            df["sample_id"] = df.index.astype(str)
        print(f"[INFO] Total inference samples: {len(df)}")

        dataset = InferenceDataset(df, cfg.test_img_root, processor, cfg, sal_type=0)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )

    else:
        # Single image mode
        print(f"[INFO] Single mode: {cfg.target_image}")
        single_df = pd.DataFrame([{
            "image_path":  os.path.relpath(cfg.target_image, cfg.test_img_root or "/"),
            "caption":     cfg.caption or "",
            "sample_id":   "0",
            "__root_dir":  os.path.dirname(os.path.abspath(cfg.target_image)),
        }])
        # 对 single mode 直接用绝对路径覆盖 image_path
        single_df["image_path"] = os.path.basename(cfg.target_image)
        single_df["__root_dir"] = os.path.dirname(os.path.abspath(cfg.target_image))

        dataset = InferenceDataset(single_df, single_df["__root_dir"].iloc[0], processor, cfg, sal_type=0)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # -------- run inference --------
    results2 = run_inference(model, loader, cfg, device)
    # -------- free model --------
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    mask_dir = os.path.join(cfg.output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    jsonl_path = os.path.join(cfg.output_dir, "flaw_regions.json")
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    for r1, r2 in zip(results, results2):

        sal1 = r1["sal_map"]
        sal2 = r2["sal_map"]

        fused, mask, bboxes = merge_saliency_to_regions(sal1, sal2, thresh=0.5)

        target_path = r1["target_path"]

        # -----------------------------
        # 1. save mask image
        # -----------------------------
        mask_img = (mask * 255).astype(np.uint8)
        filename = os.path.basename(target_path)  # e.g. Change_text_202.jpg
        name, _ = os.path.splitext(filename)  # -> Change_text_202
        mask_save_path = os.path.join(mask_dir, f"{name}.png")

        cv2.imwrite(mask_save_path, mask_img)

        # -----------------------------
        # 2. build jsonl entry
        # -----------------------------
        instruction = cfg.caption if cfg.caption else ""
        bbox_str = "; ".join([f"({x1},{y1},{x2},{y2})" for (x1, y1, x2, y2) in bboxes])
        query_text = (
            f"With the image editing prompt [{instruction}], "
            f"the source image <image> is edited to generate the target image <image>. "
            f"The detected flaw regions (bounding boxes) are: [{bbox_str}]. "
            f"Please specify the flaw type and textual description.\n"
            f"Answer with the following format:\n"
            f"[\n"
            f"  {{\n"
            f"    \"flaw_type\": \"Flaw type\",\n"
            f"    \"description\": \"Description of the flaw.\"\n"
            f"  }}\n"
            f"]"
            f"\n Each bounding box should correspond to a separate answer."
        )

        json_obj = {
            "query": query_text,
            "response": "",
            "images": [
                cfg.source_image if cfg.source_image else "",
                target_path
            ]
        }

        jsonl_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")






if __name__ == "__main__":
    main()

