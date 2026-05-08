import argparse
import gc
import math
import os
import random
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor
import torch.nn.functional as F
from modeling_qwen3_vl_8b_sal import Qwen3VL8BLoRARegression

@dataclass
class TrainConfig:
    model_id: str
    run_root: str
    train_csv: str
    train_img_root: str
    val_csv: str
    val_img_root: str
    test_csv: str
    test_img_root: str
    seed: int
    max_length: int
    max_pixels_per_image: int
    batch_size: int
    grad_accum_steps: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    max_grad_norm: float
    warmup_ratio: float
    min_lr_ratio: float
    train_num_workers: int
    val_num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    precision: str
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    lambda_huber: float
    huber_delta: float
    lambda_rank: float
    lambda_plcc: float
    final_stage_epochs: int
    final_stage_rank_multiplier: float
    final_stage_plcc_multiplier: float
    rank_queue_size: int
    rank_label_margin: float
    plcc_queue_size: int
    plcc_eps: float
    plcc_min_std: float


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="/mnt/sdb/xzt/LLM/Qwen3VL")
    p.add_argument("--run_root", type=str, default="/mnt/sdb/xzt/NIPS/qwen3_vl_8b_full_lora_dp")

    p.add_argument("--train_csv", type=str, default="/mnt/sdb/xzt/NIPS/alignment.csv")
    p.add_argument("--train_img_root", type=str, default="/mnt/sdb/xzt/NIPS")
    p.add_argument("--val_csv", type=str, default="/mnt/sdb/xzt/NIPS/alignment.csv")
    p.add_argument("--val_img_root", type=str, default="/mnt/sdb/xzt/NIPS")
    p.add_argument("--test_csv", type=str, default="/mnt/sdb/xzt/NIPS/alignment.csv")
    p.add_argument("--test_img_root", type=str, default="/mnt/sdb/xzt/NIPS")

    p.add_argument("--seed", type=int, default=20260319)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_pixels_per_image", type=int, default=1048576)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=12)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--warmup_ratio", type=float, default=0)
    p.add_argument("--min_lr_ratio", type=float, default=0.4)

    p.add_argument("--train_num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true", default=True)

    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--lambda_huber", type=float, default=4.0)
    p.add_argument("--huber_delta", type=float, default=0.1)
    p.add_argument("--lambda_rank", type=float, default=0.05)
    p.add_argument("--lambda_plcc", type=float, default=0.05)
    p.add_argument("--final_stage_epochs", type=int, default=2)
    p.add_argument("--final_stage_rank_multiplier", type=float, default=1.5)
    p.add_argument("--final_stage_plcc_multiplier", type=float, default=1.5)
    p.add_argument("--rank_queue_size", type=int, default=4096)
    p.add_argument("--rank_label_margin", type=float, default=0.03)
    p.add_argument("--plcc_queue_size", type=int, default=256)
    p.add_argument("--plcc_eps", type=float, default=1e-6)
    p.add_argument("--plcc_min_std", type=float, default=0.02)

    a = p.parse_args()
    return TrainConfig(
        model_id=a.model_id,
        run_root=a.run_root,
        train_csv=a.train_csv,
        train_img_root=a.train_img_root,
        val_csv=a.val_csv,
        val_img_root=a.val_img_root,
        test_csv=a.test_csv,
        test_img_root=a.test_img_root,
        seed=a.seed,
        max_length=a.max_length,
        max_pixels_per_image=a.max_pixels_per_image,
        batch_size=a.batch_size,
        grad_accum_steps=a.grad_accum_steps,
        num_epochs=a.num_epochs,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        adam_beta1=a.adam_beta1,
        adam_beta2=a.adam_beta2,
        adam_eps=a.adam_eps,
        max_grad_norm=a.max_grad_norm,
        warmup_ratio=a.warmup_ratio,
        min_lr_ratio=a.min_lr_ratio,
        train_num_workers=a.train_num_workers,
        val_num_workers=a.val_num_workers,
        prefetch_factor=a.prefetch_factor,
        persistent_workers=a.persistent_workers,
        precision=a.precision,
        gradient_checkpointing=a.gradient_checkpointing,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        lora_target_modules=[x.strip() for x in a.lora_target_modules.split(",") if x.strip()],
        lambda_huber=a.lambda_huber,
        huber_delta=a.huber_delta,
        lambda_rank=a.lambda_rank,
        lambda_plcc=a.lambda_plcc,
        final_stage_epochs=a.final_stage_epochs,
        final_stage_rank_multiplier=a.final_stage_rank_multiplier,
        final_stage_plcc_multiplier=a.final_stage_plcc_multiplier,
        rank_queue_size=a.rank_queue_size,
        rank_label_margin=a.rank_label_margin,
        plcc_queue_size=a.plcc_queue_size,
        plcc_eps=a.plcc_eps,
        plcc_min_std=a.plcc_min_std,
    )


def dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if dist_is_ready() else 1


def rank_id() -> int:
    return dist.get_rank() if dist_is_ready() else 0


def is_main_process() -> bool:
    return rank_id() == 0


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Failed to decode {csv_path}: {last_exc}")


class Qwen3VLIQADataset(Dataset):
    def __init__(self, data_source, root_dir, processor, cfg: TrainConfig, mode="train"):
        if isinstance(data_source, str):
            self.data = read_csv_with_fallback(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source.reset_index(drop=True)
        else:
            raise ValueError("data_source must be path or DataFrame")

        self.root_dir = root_dir
        self.processor = processor
        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def _load_and_limit_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            pixels = w * h
            if pixels > self.cfg.max_pixels_per_image:
                scale = math.sqrt(self.cfg.max_pixels_per_image / float(pixels))
                new_w = max(16, int(round(w * scale)))
                new_h = max(16, int(round(h * scale)))
                new_w = max(16, (new_w // 16) * 16)
                new_h = max(16, (new_h // 16) * 16)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return img

    def _resolve_existing_path(self, image_rel_path, root_dir=None):
        active_root = root_dir or self.root_dir
        direct_path = os.path.join(active_root, image_rel_path)
        if os.path.exists(direct_path):
            return direct_path

        parts = str(image_rel_path).split("/")
        for idx, part in enumerate(parts[:-1]):
            for repl in (part.replace("_", "-"), part.replace("-", "_")):
                if repl == part:
                    continue
                candidate = parts.copy()
                candidate[idx] = repl
                candidate_path = os.path.join(active_root, "/".join(candidate))
                if os.path.exists(candidate_path):
                    return candidate_path

        raise FileNotFoundError(f"Image not found: {direct_path}")

    def _build_prompt(self, row):
        row_root_dir = str(row.get("__root_dir", self.root_dir))
        source_rel = str(row["source"])
        edited_rel = str(row["edited"])
        gray_rel = str(row["gray"])
        source_path = self._resolve_existing_path(source_rel, root_dir=row_root_dir)
        target_path = self._resolve_existing_path(edited_rel, root_dir=row_root_dir)
        gray_path = self._resolve_existing_path(gray_rel, root_dir=row_root_dir)
        instruction = str(row.get("prompt", "")).strip()

        prompt = (
            "Task: find where misalign the editing instruction occur in the Target image compared to the Source image. "
            f"Editing instruction: '{instruction}'. "
            "The Source image is the original reference and the Target image is the edited result. "
            "Identify regions in the Target image that not properly executed or regions incorrectly modified despite not requiring edits. "
            "Only focus on regions with distortions."
        )

        return source_path, target_path, gray_path, prompt,

    def _load_gray(self, path: Path) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("L")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source_path, target_path, gray_path, prompt_text = self._build_prompt(row)
        source_image = self._load_and_limit_image(source_path)
        target_image = self._load_and_limit_image(target_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": target_image},
                    {"type": "text", "text": prompt_text},
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
        # print(inputs_clip)

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        #
        # target_path_obj = Path(target_path)
        # label_path = str(target_path_obj).replace("images", "labels/artifact")
        # label_img = self._load_gray(label_path)
        # # 转 tensor 并归一化到 0~1
        # label = torch.from_numpy(np.array(label_img)).float()
        # # 加 channel 维度  -> [1, H, W]
        # label = label.unsqueeze(0)
        # print(label)

        if self.mode != "test":
            label_img = self._load_gray(gray_path)

            label = torch.from_numpy(np.array(label_img)).float()
            # label = label.unsqueeze(0)
            # resize 到 target 的 spatial 尺寸
            H_target, W_target = target_image.size[1], target_image.size[0]
            label = label.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]
            label = F.interpolate(label, size=(H_target, W_target), mode='bilinear', align_corners=False)
            label = label.squeeze(0).squeeze(0)
        else:
            # test 阶段没有 GT
            label = torch.zeros(target_image.size[1], target_image.size[0], dtype=torch.float32)

        # -------------------------
        # 输出
        # -------------------------
        target_size = torch.tensor([target_image.size[1], target_image.size[0]], dtype=torch.int)  # [H, W]
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "target_size": target_size,  # 新增
        }

        if pixel_values is not None:
            out["pixel_values"] = pixel_values

        if image_grid_thw is not None:
            out["image_grid_thw"] = image_grid_thw

        return out


def make_collate_fn(pad_token_id: int):
    def _collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.stack([item["labels"] for item in batch])

        out = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels,
        }
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        if "target_size" in batch[0]:
            out["target_size"] = torch.stack([item["target_size"] for item in batch], dim=0)  # [B,2]
        return out

    return _collate_fn


class CrossGPURankQueue:
    def __init__(self, max_size=4096):
        self.max_size = int(max_size)
        self._pred_queue = deque()
        self._label_queue = deque()
        self._size = 0

    def enqueue(self, preds_detached, labels_detached):
        preds_cpu = preds_detached.float().cpu()
        labels_cpu = labels_detached.float().cpu()
        self._pred_queue.append(preds_cpu)
        self._label_queue.append(labels_cpu)
        self._size += preds_cpu.shape[0]

        while self._size > self.max_size and len(self._pred_queue) > 0:
            removed_preds = self._pred_queue.popleft()
            self._label_queue.popleft()
            self._size -= removed_preds.shape[0]

    def get(self, device):
        if self._size == 0:
            return None, None
        preds = torch.cat(list(self._pred_queue), dim=0).to(device)
        labels = torch.cat(list(self._label_queue), dim=0).to(device)
        return preds, labels


def drop_missing_pairs(df: pd.DataFrame, root_dir: str, split_name: str) -> pd.DataFrame:
    source_paths = df["source"].astype(str).map(lambda p: os.path.join(root_dir, p))
    edited_paths = df["edited"].astype(str).map(lambda p: os.path.join(root_dir, p))
    valid_mask = source_paths.map(os.path.exists) & edited_paths.map(os.path.exists)
    dropped = int((~valid_mask).sum())
    log(f"[{split_name}] dropped {dropped} samples")
    return df.loc[valid_mask].reset_index(drop=True)


def prepare_datasets(processor, cfg: TrainConfig):
    df_train = read_csv_with_fallback(cfg.train_csv)
    df_train["__root_dir"] = cfg.train_img_root
    df_train = drop_missing_pairs(df_train, cfg.train_img_root, "train")

    df_val = read_csv_with_fallback(cfg.val_csv)
    df_val["__root_dir"] = cfg.val_img_root
    df_val = drop_missing_pairs(df_val, cfg.val_img_root, "val")

    df_test = read_csv_with_fallback(cfg.test_csv)
    df_test["__root_dir"] = cfg.test_img_root
    df_test = drop_missing_pairs(df_test, cfg.test_img_root, "test")
    log(f"train samples={len(df_train)}")
    log(f"val samples={len(df_val)}")
    log(f"test samples={len(df_test)}")

    train_dataset = Qwen3VLIQADataset(df_train, cfg.train_img_root, processor, cfg, mode="train")
    val_dataset = Qwen3VLIQADataset(df_val, cfg.val_img_root, processor, cfg, mode="val")
    test_dataset = Qwen3VLIQADataset(df_test, cfg.test_img_root, processor, cfg, mode="val")
    return train_dataset, val_dataset, test_dataset


def build_loader(dataset, batch_size, collate_fn, num_workers, prefetch_factor, persistent_workers, is_train):
    sampler = DistributedSampler(dataset, shuffle=is_train, drop_last=is_train) if world_size() > 1 else None
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": sampler is None and is_train,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": is_train,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)
    return loader, sampler


def build_warmup_cosine_lr_lambda(current_step, total_steps, warmup_steps, min_lr_ratio):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)


def build_epoch_eval_steps(updates_per_epoch):
    marks = [
        max(1, int(round(updates_per_epoch / 1.0))),
        max(1, int(round((1.0 * updates_per_epoch) / 1.0))),
    ]
    return sorted(set(min(updates_per_epoch, mark) for mark in marks))


def get_epoch_loss_weights(cfg: TrainConfig, epoch_idx: int):
    rank_weight = cfg.lambda_rank
    plcc_weight = cfg.lambda_plcc
    if epoch_idx >= max(0, cfg.num_epochs - cfg.final_stage_epochs):
        rank_weight *= cfg.final_stage_rank_multiplier
        plcc_weight *= cfg.final_stage_plcc_multiplier
    return cfg.lambda_huber, rank_weight, plcc_weight


def safe_corr_np(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, 0.0
    plcc = float(np.corrcoef(x, y)[0, 1])
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    srcc = float(np.corrcoef(rx, ry)[0, 1])
    if np.isnan(plcc):
        plcc = 0.0
    if np.isnan(srcc):
        srcc = 0.0
    return plcc, srcc


def all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    if world_size() == 1:
        return x
    gathered = [torch.zeros_like(x) for _ in range(world_size())]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def compute_rank_loss_with_queue(current_logits, current_labels, queue_logits, queue_labels, label_margin):
    if queue_logits is None or queue_labels is None or queue_logits.shape[0] == 0:
        return current_logits.new_zeros(())

    rank_losses = []
    for dim_idx in range(current_logits.shape[-1]):
        pred_cur = current_logits[:, dim_idx].unsqueeze(1)
        pred_mem = queue_logits[:, dim_idx].unsqueeze(0)
        label_cur = current_labels[:, dim_idx].unsqueeze(1)
        label_mem = queue_labels[:, dim_idx].unsqueeze(0)

        label_diff = label_cur - label_mem
        valid_mask = torch.abs(label_diff) > label_margin
        if not torch.any(valid_mask):
            continue

        sign = torch.sign(label_diff)
        pred_diff = pred_cur - pred_mem
        loss_matrix = F.softplus(-sign * pred_diff)
        rank_losses.append(loss_matrix[valid_mask].mean())

    if not rank_losses:
        return current_logits.new_zeros(())
    return torch.stack(rank_losses).mean()


def compute_plcc_loss_with_queue(current_logits, current_labels, queue_logits, queue_labels, max_queue_size, min_std, eps):
    if queue_logits is not None and queue_labels is not None and queue_logits.shape[0] > 0:
        queue_logits = queue_logits[-max_queue_size:]
        queue_labels = queue_labels[-max_queue_size:]
        all_logits = torch.cat([queue_logits.detach(), current_logits], dim=0)
        all_labels = torch.cat([queue_labels.detach(), current_labels], dim=0)
    else:
        all_logits = current_logits
        all_labels = current_labels

    if all_logits.shape[0] < 2:
        return current_logits.new_zeros(())

    plcc_losses = []
    for dim_idx in range(all_logits.shape[-1]):
        pred = all_logits[:, dim_idx]
        target = all_labels[:, dim_idx]

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()

        pred_std = torch.sqrt(torch.mean(pred_centered.pow(2))).clamp_min(min_std)
        target_std = torch.sqrt(torch.mean(target_centered.pow(2))).clamp_min(min_std)
        denom = (pred_std * target_std).clamp_min(eps)

        corr = torch.mean(pred_centered * target_centered) / denom
        corr = torch.clamp(corr, min=-1.0, max=1.0)
        plcc_losses.append(1.0 - corr)

    if not plcc_losses:
        return current_logits.new_zeros(())
    return torch.stack(plcc_losses).mean()


def map_kld_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float().clamp(0.0, 1.0)
    pred = pred.float().clamp(eps, 1.0)

    # 如果是单通道 [B,H,W]，只 sum H,W
    target = target / (target.sum(dim=(1, 2), keepdim=True) + eps)
    pred = pred / (pred.sum(dim=(1, 2), keepdim=True) + eps)

    kld = target * (torch.log(target + eps) - torch.log(pred + eps))
    return kld.sum(dim=(1, 2)).mean()

def save_checkpoint(model_ddp, processor, save_dir: str):
    if not is_main_process():
        return
    os.makedirs(save_dir, exist_ok=True)
    model = model_ddp.module if isinstance(model_ddp, DDP) else model_ddp
    model.vl_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    torch.save(model.SalDecoder.state_dict(), os.path.join(save_dir, "sal_head.pth"))


def validate_multi_head(model_ddp, val_loader, cfg: TrainConfig, device: torch.device):
    model_ddp.eval()
    preds = []
    gts = []
    total_loss = 0.0
    num_batches = 0

    bar = tqdm(val_loader, desc="valid", disable=not is_main_process())
    use_amp = cfg.precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16

    with torch.no_grad():
        for batch in bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                logits = model_ddp(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    image_size=batch["target_size"].to(device),  # 假设你模型 forward 支持 image_size 参数
                )


            loss = map_kld_loss(labels, logits)

            # --- DDP 平均 loss ---
            loss_all = all_gather_no_grad(loss.detach().unsqueeze(0))
            loss_mean = loss_all.mean()

            total_loss += loss_mean.item()
            num_batches += 1

            preds.append(all_gather_no_grad(logits.float()).cpu())
            gts.append(all_gather_no_grad(labels.float()).cpu())

    avg_loss = total_loss / num_batches
    print(avg_loss)

    # pred_arr = torch.cat(preds, dim=0).numpy()
    # gt_arr = torch.cat(gts, dim=0).numpy()
    #
    # metrics = {}
    # score_sums = []
    # for idx, name in enumerate(SCORE_COLS):
    #     plcc, srcc = safe_corr_np(pred_arr[:, idx], gt_arr[:, idx])
    #     score_sum = float(plcc + srcc)
    #     score_sums.append(score_sum)
    #     metrics[f"{name}_plcc"] = float(plcc)
    #     metrics[f"{name}_srcc"] = float(srcc)
    #     metrics[f"{name}_score_sum"] = score_sum
    #
    # metrics["mean_score_sum"] = float(np.mean(score_sums))
    model_ddp.train()
    return avg_loss


def validate_multi_head_(model_ddp, val_loader, cfg: TrainConfig, device: torch.device):
    avg_loss = 1
    model_ddp.train()
    return avg_loss

def run_step_validation_and_maybe_save(model_ddp, processor, val_loader, cfg, step, best_score, tag, device):
    if dist_is_ready():
        dist.barrier()
    metrics = validate_multi_head_(model_ddp, val_loader, cfg, device)
    current_score = metrics

    ckpt_dir = os.path.join(cfg.run_root, "checkpoints_alignment", f"ckpt_{tag}_{step}")
    save_checkpoint(model_ddp, processor, ckpt_dir)

    # if is_main_process():
    #     log(
    #         f"Validation({tag}) step={step}: "
    #         f"V({metrics['visual_plcc']:.4f}/{metrics['visual_srcc']:.4f}) "
    #         f"E({metrics['editing_plcc']:.4f}/{metrics['editing_srcc']:.4f}) "
    #         f"P({metrics['preservation_plcc']:.4f}/{metrics['preservation_srcc']:.4f}) "
    #         f"MeanSum={current_score:.4f}"
    #     )

    if current_score > best_score:
        best_score = current_score
        best_dir = os.path.join(cfg.run_root, "checkpoints_alignment", f"best_{tag}_{step}")
        save_checkpoint(model_ddp, processor, best_dir)

    if dist_is_ready():
        tensor = torch.tensor([best_score], dtype=torch.float32, device=device)
        dist.broadcast(tensor, src=0)
        best_score = float(tensor.item())
    return best_score


def train(cfg: TrainConfig):
    if torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))

    if ws > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed = cfg.seed + global_rank
    set_global_seed(seed)

    if is_main_process():
        os.makedirs(os.path.join(cfg.run_root, "checkpoints_alignment"), exist_ok=True)
        os.makedirs(os.path.join(cfg.run_root, "logs"), exist_ok=True)

    log(f"run_root={cfg.run_root}")
    log(f"model_id={cfg.model_id}")
    log(f"world_size={world_size()} local_rank={local_rank}")

    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3VL8BLoRARegression(
        model_name_or_path=cfg.model_id,
        rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        num_labels=3,
        precision=cfg.precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    train_dataset, val_dataset, test_dataset = prepare_datasets(processor, cfg)
    collate_fn = make_collate_fn(processor.tokenizer.pad_token_id)

    train_loader, train_sampler = build_loader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.train_num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        is_train=True,
    )
    val_loader, _ = build_loader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.val_num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        is_train=False,
    )
    test_loader, _ = build_loader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.val_num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        is_train=False,
    )

    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=cfg.learning_rate,  # SGD 的学习率可能需要比 Adam 小一些或调大
    #     momentum=0.9,  # 通常用 0.9 动量
    #     weight_decay=cfg.weight_decay
    # )

    epoch_batches = max(1, len(train_loader))
    updates_per_epoch = max(1, int(math.ceil(epoch_batches / cfg.grad_accum_steps)))
    max_train_steps = max(1, updates_per_epoch * cfg.num_epochs)
    warmup_steps = int(max_train_steps * cfg.warmup_ratio)
    epoch_eval_steps = build_epoch_eval_steps(updates_per_epoch)

    lr_lambda = lambda step: build_warmup_cosine_lr_lambda(step, max_train_steps, warmup_steps, cfg.min_lr_ratio)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = cfg.precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16"))

    best_score = -1.0
    global_step = 0
    rank_queue = CrossGPURankQueue(max_size=cfg.rank_queue_size)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lambda_huber, lambda_rank, lambda_plcc = get_epoch_loss_weights(cfg, epoch)
        if is_main_process():
            log(
                f"Epoch {epoch+1}/{cfg.num_epochs}: "
                f"huber={lambda_huber:.4f}, rank={lambda_rank:.4f}, plcc={lambda_plcc:.4f}"
            )

        model.train()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        epoch_step = 0
        bar = tqdm(train_loader, desc=f"train {epoch+1}/{cfg.num_epochs}", disable=not is_main_process())

        for batch_idx, batch in enumerate(bar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device, non_blocking=True)

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                sync_context = model.no_sync() if (batch_idx % cfg.grad_accum_steps != 0) else nullcontext()
                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            else:
                # 单卡直接用 nullcontext()，梯度累积逻辑仍然通过 loss / grad_accum_steps 实现
                sync_context = nullcontext()
                amp_ctx = nullcontext()

            with sync_context:
                with amp_ctx:
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        image_size=batch["target_size"].to(device),  # 假设你模型 forward 支持 image_size 参数
                    )
                    labels_f = labels.float() / 255  # [0,1] 的 GT
                    logits_f = logits.float()  # 原始输出 logits，不加 Sigmoid

                    # 1️⃣ KLD loss
                    kld_loss = map_kld_loss(labels_f, logits_f)

                    # 2️⃣ L1 loss 保持数值拟合前景
                    l1_loss = F.l1_loss(torch.sigmoid(logits_f), labels_f)
                    # 注意：L1 这里最好对 logits 做 Sigmoid，否则梯度可能不稳定

                    # 3️⃣ BCEWithLogitsLoss 让零区域预测更接近 0
                    bce_loss = F.binary_cross_entropy_with_logits(logits_f, labels_f)

                    # 4️⃣ 总 loss
                    loss = bce_loss + kld_loss + l1_loss


                    # huber_loss = F.huber_loss(logits_f, labels_f, delta=cfg.huber_delta)
                    # queue_logits, queue_labels = rank_queue.get(device=logits_f.device)
                    # rank_loss = compute_rank_loss_with_queue(
                    #     current_logits=logits_f,
                    #     current_labels=labels_f,
                    #     queue_logits=queue_logits,
                    #     queue_labels=queue_labels,
                    #     label_margin=cfg.rank_label_margin,
                    # )
                    # plcc_loss = compute_plcc_loss_with_queue(
                    #     current_logits=logits_f,
                    #     current_labels=labels_f,
                    #     queue_logits=queue_logits,
                    #     queue_labels=queue_labels,
                    #     max_queue_size=cfg.plcc_queue_size,
                    #     min_std=cfg.plcc_min_std,
                    #     eps=cfg.plcc_eps,
                    # )
                    # loss = lambda_huber * huber_loss + lambda_rank * rank_loss + lambda_plcc * plcc_loss
                    loss = loss / cfg.grad_accum_steps
                    # print(loss)

                if cfg.precision == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            gathered_logits = all_gather_no_grad(logits_f.detach())
            gathered_labels = all_gather_no_grad(labels_f.detach())
            rank_queue.enqueue(gathered_logits, gathered_labels)

            if batch_idx % cfg.grad_accum_steps == 0:
                if cfg.precision == "fp16":
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if cfg.precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch_step += 1

                if is_main_process():
                    bar.set_postfix({
                        "loss": f"{(loss.item() * cfg.grad_accum_steps):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "gn": f"{float(grad_norm):.3f}",
                    })

                if epoch_step in epoch_eval_steps:
                    best_score = run_step_validation_and_maybe_save(
                        model_ddp=model,
                        processor=processor,
                        val_loader=val_loader,
                        cfg=cfg,
                        step=global_step,
                        best_score=best_score,
                        tag=f"epoch{epoch+1}_step{epoch_step}",
                        device=device,
                    )

        best_score = run_step_validation_and_maybe_save(
            model_ddp=model,
            processor=processor,
            val_loader=val_loader,
            cfg=cfg,
            step=global_step,
            best_score=best_score,
            tag=f"epoch{epoch+1}_end",
            device=device,
        )

    if dist_is_ready():
        dist.barrier()
    test_metrics = validate_multi_head(model, test_loader, cfg, device)
    if is_main_process():
        log(
            "Final Test: "
            f"V({test_metrics['visual_plcc']:.4f}/{test_metrics['visual_srcc']:.4f}) "
            f"E({test_metrics['editing_plcc']:.4f}/{test_metrics['editing_srcc']:.4f}) "
            f"P({test_metrics['preservation_plcc']:.4f}/{test_metrics['preservation_srcc']:.4f}) "
            f"MeanSum={test_metrics['mean_score_sum']:.4f}"
        )

    if dist_is_ready():
        dist.barrier()
        dist.destroy_process_group()

    del train_loader, val_loader, test_loader, optimizer, lr_scheduler, model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
