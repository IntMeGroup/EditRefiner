import argparse
import csv
import gc
import json
import math
import os
import random
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]

@dataclass
class TrainConfig:
    model_id: str
    run_root: str
    train_csv: str
    val_csv: str
    test_csv: str
    seed: int
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
    lora_target_modules: list
    lambda_huber: float
    lambda_rank: float
    lambda_plcc: float
    huber_delta: float
    rank_queue_size: int
    rank_label_margin: float
    plcc_queue_size: int
    plcc_min_std: float
    plcc_eps: float
    score_output_scale: float


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-8B LoRA regression")
    parser.add_argument("--model_id", type=str, default="/mnt/sdb/xzt/LLM/Qwen3VL")
    parser.add_argument("--run_root", type=str, default="/mnt/sdb/xzt/ECCV/reward_model/IEQA")
    parser.add_argument("--train_csv", type=str, default="/mnt/sdb/xzt/ECCV/reward_model/IEQA/train.csv")
    parser.add_argument("--val_csv", type=str, default="/mnt/sdb/xzt/ECCV/reward_model/IEQA/test.csv")
    parser.add_argument("--test_csv", type=str, default="/mnt/sdb/xzt/ECCV/reward_model/IEQA/test.csv")
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--max_pixels_per_image", type=int, default=1048576)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--min_lr_ratio", type=float, default=0.4)
    parser.add_argument("--train_num_workers", type=int, default=8)
    parser.add_argument("--val_num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lambda_huber", type=float, default=4.0)
    parser.add_argument("--lambda_rank", type=float, default=0.05)
    parser.add_argument("--lambda_plcc", type=float, default=0.05)
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--rank_queue_size", type=int, default=4096)
    parser.add_argument("--rank_label_margin", type=float, default=0.03)
    parser.add_argument("--plcc_queue_size", type=int, default=256)
    parser.add_argument("--plcc_min_std", type=float, default=0.02)
    parser.add_argument("--plcc_eps", type=float, default=1e-6)
    parser.add_argument("--score_output_scale", type=float, default=100.0)

    args = parser.parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv

    return TrainConfig(
        model_id=args.model_id,
        run_root=args.run_root,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        seed=args.seed,
        max_pixels_per_image=args.max_pixels_per_image,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        lambda_huber=args.lambda_huber,
        lambda_rank=args.lambda_rank,
        lambda_plcc=args.lambda_plcc,
        huber_delta=args.huber_delta,
        rank_queue_size=args.rank_queue_size,
        rank_label_margin=args.rank_label_margin,
        plcc_queue_size=args.plcc_queue_size,
        plcc_min_std=args.plcc_min_std,
        plcc_eps=args.plcc_eps,
        score_output_scale=args.score_output_scale,
    )


def dist_is_ready():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if dist_is_ready() else 1


def rank_id():
    return dist.get_rank() if dist_is_ready() else 0


def is_main_process():
    return rank_id() == 0


def log(message):
    if is_main_process():
        print(message, flush=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_with_fallback(csv_path):
    last_exc = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin1"):
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to decode {csv_path}: {last_exc}")


def resolve_data_path(raw_path, csv_path):
    raw_path = str(raw_path).strip()
    candidate_paths = []
    raw_path_obj = Path(raw_path)
    if raw_path_obj.is_absolute():
        candidate_paths.append(raw_path_obj)
    candidate_paths.append((Path(csv_path).parent / raw_path).resolve())
    candidate_paths.append((REPO_ROOT / raw_path).resolve())

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot resolve path '{raw_path}' from '{csv_path}'")


def build_query(instruction):
    return (
        "You are scoring an image editing result. "
        "The first image is the source image and the second image is the edited image. "
        f"Editing instruction: {instruction.strip()}. "
        f"Evaluate the edited image from perceptual quality, instruction following and visual consistency. "
        "Return the three internal assessment score."
    )


class MultiDimensionDataset(Dataset):
    def __init__(self, csv_path, processor, cfg, split_name):
        self.csv_path = str(csv_path)
        self.processor = processor
        self.cfg = cfg
        self.split_name = split_name
        self.rows = read_csv_with_fallback(self.csv_path)

        if not self.rows:
            raise ValueError(f"{self.csv_path} is empty")

        # ===== 检查列 =====
        required_columns = {"source", "edited", "visual", "editing", "preservation", "instruction"}
        missing_columns = required_columns.difference(self.rows[0].keys())
        if missing_columns:
            raise ValueError(f"{self.csv_path} missing columns: {sorted(missing_columns)}")

        # ===== 标准化 =====
        normalized_rows = []
        for row in self.rows:
            normalized_rows.append(
                {
                    "source": row["source"],
                    "edited": row["edited"],
                    "instruction": str(row["instruction"]).strip(),
                    "score": [
                        float(row["visual"]),
                        float(row["editing"]),
                        float(row["preservation"]),
                    ],
                }
            )

        self.rows = self._drop_missing_pairs(normalized_rows)

    def __len__(self):
        return len(self.rows)

    def _drop_missing_pairs(self, rows):
        valid_rows = []
        dropped = 0
        for row in rows:
            try:
                resolve_data_path(row["source"], self.csv_path)
                resolve_data_path(row["edited"], self.csv_path)
                valid_rows.append(row)
            except FileNotFoundError:
                dropped += 1
        log(f"[{self.split_name}] kept={len(valid_rows)} dropped={dropped}")
        return valid_rows

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        pixels = width * height

        if pixels <= self.cfg.max_pixels_per_image:
            return image

        scale = math.sqrt(self.cfg.max_pixels_per_image / float(pixels))
        new_width = max(16, int(round(width * scale)))
        new_height = max(16, int(round(height * scale)))
        new_width = max(16, (new_width // 16) * 16)
        new_height = max(16, (new_height // 16) * 16)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def __getitem__(self, idx):
        row = self.rows[idx]

        source_path = resolve_data_path(row["source"], self.csv_path)
        edited_path = resolve_data_path(row["edited"], self.csv_path)
        source_path = os.path.join(self.cfg.run_root, row["source"])
        edited_path = os.path.join(self.cfg.run_root, row["edited"])
        instruction = row["instruction"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._load_image(source_path)},
                    {"type": "image", "image": self._load_image(edited_path)},
                    {"type": "text", "text": build_query(instruction)},
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

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),

            # ✅ 关键：变成 3 维 label
            "labels": torch.tensor(row["score"], dtype=torch.float32),  # [3]
        }

        if "pixel_values" in inputs:
            item["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"]

        return item


def make_collate_fn(pad_token_id):
    def _collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        out = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": labels,
        }
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        return out

    return _collate_fn


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
    return DataLoader(**loader_kwargs), sampler


def build_warmup_cosine_lr_lambda(current_step, total_steps, warmup_steps, min_lr_ratio):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)


def build_epoch_eval_steps(updates_per_epoch):
    marks = [
        max(1, int(round(updates_per_epoch))),
        max(1, int(round((2.0 * updates_per_epoch)))),
    ]
    return sorted(set(min(updates_per_epoch, mark) for mark in marks))


def safe_corr_np(x, y):
    x = np.asarray(x, dtype=np.float64)  # [N, 3]
    y = np.asarray(y, dtype=np.float64)  # [N, 3]

    assert x.shape == y.shape, "pred and gt shape mismatch"

    plcc_list = []
    srcc_list = []

    for i in range(x.shape[1]):
        xi = x[:, i]
        yi = y[:, i]

        if len(xi) < 2 or np.std(xi) < 1e-8 or np.std(yi) < 1e-8:
            plcc_list.append(0.0)
            srcc_list.append(0.0)
            continue

        plcc = float(np.corrcoef(xi, yi)[0, 1])

        rx = average_rank(xi)
        ry = average_rank(yi)
        srcc = float(np.corrcoef(rx, ry)[0, 1])

        if np.isnan(plcc):
            plcc = 0.0
        if np.isnan(srcc):
            srcc = 0.0

        plcc_list.append(plcc)
        srcc_list.append(srcc)

    return plcc_list, srcc_list


def average_rank(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def all_gather_no_grad(x):
    if world_size() == 1:
        return x
    gathered = [torch.zeros_like(x) for _ in range(world_size())]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


class CrossGPURankQueue:
    def __init__(self, max_size):
        self.max_size = int(max_size)
        self.pred_queue = deque()
        self.label_queue = deque()
        self.size = 0

    def enqueue(self, preds_detached, labels_detached):
        preds_cpu = preds_detached.float().cpu()
        labels_cpu = labels_detached.float().cpu()
        self.pred_queue.append(preds_cpu)
        self.label_queue.append(labels_cpu)
        self.size += preds_cpu.shape[0]
        while self.size > self.max_size and len(self.pred_queue) > 0:
            removed_preds = self.pred_queue.popleft()
            self.label_queue.popleft()
            self.size -= removed_preds.shape[0]

    def get(self, device):
        if self.size == 0:
            return None, None
        preds = torch.cat(list(self.pred_queue), dim=0).to(device)
        labels = torch.cat(list(self.label_queue), dim=0).to(device)
        return preds, labels


def compute_rank_loss_with_queue(current_logits, current_labels, queue_logits, queue_labels, label_margin):
    if queue_logits is None or queue_labels is None or queue_logits.shape[0] == 0:
        return current_logits.new_zeros(())

    label_diff = current_labels.unsqueeze(1) - queue_labels.unsqueeze(0)
    valid_mask = torch.abs(label_diff) > label_margin
    if not torch.any(valid_mask):
        return current_logits.new_zeros(())

    pred_diff = current_logits.unsqueeze(1) - queue_logits.unsqueeze(0)
    sign = torch.sign(label_diff)
    loss_matrix = F.softplus(-sign * pred_diff)
    return loss_matrix[valid_mask].mean()


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

    pred_centered = all_logits - all_logits.mean()
    target_centered = all_labels - all_labels.mean()
    pred_std = torch.sqrt(torch.mean(pred_centered.pow(2))).clamp_min(min_std)
    target_std = torch.sqrt(torch.mean(target_centered.pow(2))).clamp_min(min_std)
    corr = torch.mean(pred_centered * target_centered) / (pred_std * target_std).clamp_min(eps)
    corr = torch.clamp(corr, min=-1.0, max=1.0)
    return 1.0 - corr


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def save_checkpoint(model_ddp, processor, cfg, save_dir):
    if not is_main_process():
        return
    os.makedirs(save_dir, exist_ok=True)
    model = model_ddp.module if isinstance(model_ddp, DDP) else model_ddp
    model.vl_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    torch.save(model.score_head.state_dict(), os.path.join(save_dir, "score_head.pth"))

    regression_config = model.score_head.export_config()
    regression_config.update(
        {
            "score_output_scale": cfg.score_output_scale,
            "base_model_path": cfg.model_id,
        }
    )
    save_json(os.path.join(save_dir, "regression_config.json"), regression_config)


def validate_single_head(model_ddp, data_loader, cfg, device):
    model_ddp.eval()
    preds = []
    gts = []
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    use_amp = cfg.precision in {"fp16", "bf16"}

    with torch.no_grad():
        bar = tqdm(data_loader, desc="valid", disable=not is_main_process())
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
                )

            preds.append(all_gather_no_grad(logits.float()).cpu())
            gts.append(all_gather_no_grad(labels.float()).cpu())

    pred_arr = torch.cat(preds, dim=0).numpy()
    gt_arr = torch.cat(gts, dim=0).numpy()
    plcc_list, srcc_list = safe_corr_np(pred_arr, gt_arr)
    plcc = float(np.mean(plcc_list))
    srcc = float(np.mean(srcc_list))
    diff = pred_arr - gt_arr
    rmse_list = np.sqrt(np.mean(diff ** 2, axis=0))
    rmse = float(np.mean(rmse_list))
    score_sum = float(plcc + srcc)
    model_ddp.train()
    return {
        "plcc": float(plcc),
        "srcc": float(srcc),
        "rmse": rmse,
        "score_sum": score_sum,
    }


def run_validation_and_maybe_save(model_ddp, processor, val_loader, cfg, step, best_score, tag, device, writer):
    if dist_is_ready():
        dist.barrier()
    metrics = validate_single_head(model_ddp, val_loader, cfg, device)
    current_score = metrics["score_sum"]

    ckpt_dir = os.path.join(cfg.run_root, "checkpoints", f"ckpt_{tag}_{step}")
    save_checkpoint(model_ddp, processor, cfg, ckpt_dir)

    if is_main_process():
        log(
            f"PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
            f"RMSE={metrics['rmse']:.4f} Sum={metrics['score_sum']:.4f}"
        )
        if writer is not None:
            writer.add_scalar("val/plcc", metrics["plcc"], step)
            writer.add_scalar("val/srcc", metrics["srcc"], step)
            writer.add_scalar("val/rmse", metrics["rmse"], step)
            writer.add_scalar("val/score_sum", metrics["score_sum"], step)

    if current_score > best_score:
        best_score = current_score
        best_dir = os.path.join(cfg.run_root, "checkpoints", "best")
        save_checkpoint(model_ddp, processor, cfg, best_dir)
        if is_main_process():
            save_json(os.path.join(cfg.run_root, "logs", "best_val_metrics.json"), metrics)

    if dist_is_ready():
        score_tensor = torch.tensor([best_score], dtype=torch.float32, device=device)
        dist.broadcast(score_tensor, src=0)
        best_score = float(score_tensor.item())
    return best_score


def train(cfg):
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoProcessor

    from modeling_qwen3_vl_regression import Qwen3VLLoRARegression

    if torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required for training")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        os.makedirs(os.path.join(cfg.run_root, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.run_root, "logs"), exist_ok=True)
        save_json(os.path.join(cfg.run_root, "logs", "train_config.json"), asdict(cfg))

    log(f"run_root={cfg.run_root}")
    log(f"model_id={cfg.model_id}")
    log(f"world_size={world_size()} local_rank={local_rank}")

    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3VLLoRARegression(
        model_name_or_path=cfg.model_id,
        rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        precision=cfg.precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    train_dataset = MultiDimensionDataset(cfg.train_csv, processor, cfg, split_name="train")
    val_dataset = MultiDimensionDataset(cfg.val_csv, processor, cfg, split_name="val")
    test_dataset = MultiDimensionDataset(cfg.test_csv, processor, cfg, split_name="test")
    collate_fn = make_collate_fn(processor.tokenizer.pad_token_id)

    train_loader, train_sampler = build_loader(
        train_dataset,
        cfg.batch_size,
        collate_fn,
        cfg.train_num_workers,
        cfg.prefetch_factor,
        cfg.persistent_workers,
        True,
    )
    val_loader, _ = build_loader(
        val_dataset,
        cfg.batch_size,
        collate_fn,
        cfg.val_num_workers,
        cfg.prefetch_factor,
        cfg.persistent_workers,
        False,
    )
    test_loader, _ = build_loader(
        test_dataset,
        cfg.batch_size,
        collate_fn,
        cfg.val_num_workers,
        cfg.prefetch_factor,
        cfg.persistent_workers,
        False,
    )

    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    epoch_batches = max(1, len(train_loader))
    updates_per_epoch = max(1, int(math.ceil(epoch_batches / cfg.grad_accum_steps)))
    max_train_steps = max(1, updates_per_epoch * cfg.num_epochs)
    warmup_steps = int(max_train_steps * cfg.warmup_ratio)
    epoch_eval_steps = build_epoch_eval_steps(updates_per_epoch)

    lr_lambda = lambda step: build_warmup_cosine_lr_lambda(step, max_train_steps, warmup_steps, cfg.min_lr_ratio)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    use_amp = cfg.precision in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16"))
    writer = SummaryWriter(log_dir=os.path.join(cfg.run_root, "logs", "tensorboard")) if is_main_process() else None

    best_score = -1.0
    global_step = 0
    rank_queue = CrossGPURankQueue(max_size=cfg.rank_queue_size)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            log(
                f"Epoch {epoch + 1}/{cfg.num_epochs}: "
                f"huber={cfg.lambda_huber:.4f}, rank={cfg.lambda_rank:.4f}, plcc={cfg.lambda_plcc:.4f}"
            )

        model.train()
        epoch_step = 0
        bar = tqdm(train_loader, desc=f"train {epoch + 1}/{cfg.num_epochs}", disable=not is_main_process())

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

            sync_context = nullcontext()
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

            with sync_context:
                with amp_ctx:
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                    logits_f = logits.float()
                    print(logits_f)
                    labels_f = labels.float()
                    print(labels_f)

                    huber_loss = F.huber_loss(logits_f, labels_f, delta=cfg.huber_delta)
                    queue_logits, queue_labels = rank_queue.get(device=logits_f.device)
                    rank_loss = compute_rank_loss_with_queue(
                        logits_f,
                        labels_f,
                        queue_logits,
                        queue_labels,
                        cfg.rank_label_margin,
                    )
                    plcc_loss = compute_plcc_loss_with_queue(
                        logits_f,
                        labels_f,
                        queue_logits,
                        queue_labels,
                        cfg.plcc_queue_size,
                        cfg.plcc_min_std,
                        cfg.plcc_eps,
                    )
                    mse_loss = F.mse_loss(logits_f, labels_f)
                    loss = mse_loss + plcc_loss
                    loss = loss / cfg.grad_accum_steps

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
                    bar.set_postfix(
                        {
                            "loss": f"{(loss.item() * cfg.grad_accum_steps):.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                            "gn": f"{float(grad_norm):.3f}",
                        }
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", float(loss.item() * cfg.grad_accum_steps), global_step)
                        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                        writer.add_scalar("train/huber_loss", float(huber_loss.item()), global_step)
                        writer.add_scalar("train/rank_loss", float(rank_loss.item()), global_step)
                        writer.add_scalar("train/plcc_loss", float(plcc_loss.item()), global_step)

                if epoch_step in epoch_eval_steps:
                    best_score = run_validation_and_maybe_save(
                        model,
                        processor,
                        val_loader,
                        cfg,
                        global_step,
                        best_score,
                        f"epoch{epoch + 1}_step{epoch_step}",
                        device,
                        writer,
                    )

        best_score = run_validation_and_maybe_save(
            model,
            processor,
            val_loader,
            cfg,
            global_step,
            best_score,
            f"epoch{epoch + 1}_end",
            device,
            writer,
        )

    if dist_is_ready():
        dist.barrier()

    test_metrics = validate_single_head(model, test_loader, cfg, device)
    if is_main_process():
        log(
            f"PLCC={test_metrics['plcc']:.4f} SRCC={test_metrics['srcc']:.4f} "
            f"RMSE={test_metrics['rmse']:.4f} Sum={test_metrics['score_sum']:.4f}"
        )
        save_json(os.path.join(cfg.run_root, "logs", "test_metrics.json"), test_metrics)
        if writer is not None:
            writer.add_hparams(
                {
                    "learning_rate": cfg.learning_rate,
                    "batch_size": cfg.batch_size,
                    "grad_accum_steps": cfg.grad_accum_steps,
                    "num_epochs": cfg.num_epochs,
                    "lora_rank": cfg.lora_rank,
                },
                {
                    "hparam/test_plcc": test_metrics["plcc"],
                    "hparam/test_srcc": test_metrics["srcc"],
                    "hparam/test_score_sum": test_metrics["score_sum"],
                },
            )
            writer.close()

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
