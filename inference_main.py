import argparse
import subprocess
import os
import json
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def run_pipeline(source_image, edited_image, instruction, output_dir="./output", max_iters=4):

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(edited_image))[0]
    mask_img = os.path.join(output_dir, "masks", f"{base_name}.png")

    cmd1 = [
        "python", "./PerceptionAgent/inference_perception.py",
        "--model_dir", "./PerceptionAgent/weights/artifact",
        "--model_dir2", "./PerceptionAgent/weights/failure",
        "--source_image", source_image,
        "--target_image", edited_image,
        "--caption", instruction,
        "--output_dir", output_dir
    ]

    cmd2 = [
        "swift", "infer",
        "--adapters", "./ReasoningAgent/weights",
        "--model_type", "qwen3_vl",
        "--model", "Qwen/Qwen3-VL-8B-Instruct",
        "--stream", "false",
        "--max_new_tokens", "2048",
        "--val_dataset", os.path.join(output_dir, "flaw_regions.json"),
        "--result_path", os.path.join(output_dir, "reasoning.json")
    ]

    def run_infer(src, edt, ins):
        cmd = [
            "python", "./EvaluationAgent/inference_evaluation.py",
            "--source_image", src,
            "--edited_image", edt,
            "--instruction", ins,
            "--peft_dir", "./EvaluationAgent/weights",
            "--model_path", "Qwen/Qwen3-VL-8B-Instruct"
        ]
        subprocess.run(cmd)

        with open(os.path.join(output_dir, "scoring.json"), "r", encoding="utf-8") as f:
            score = json.load(f)

        return (score["visual"] + score["alignment"] + score["preservation"]) / 3.0

    def parse_response(resp_str):
        start = resp_str.find("[")
        end = resp_str.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            return json.loads(resp_str[start:end+1])
        except:
            return []

    def build_prompt(flaws, instruction):
        flaw_texts = [
            f"Region {i+1}: {f['flaw_type']} - {f['description']}"
            for i, f in enumerate(flaws)
        ]

        return (
            "You are given three images:\n"
            "1. source image\n"
            "2. edited image\n"
            "3. flaw mask\n\n"
            "Original editing instruction:\n"
            f"{instruction}\n\n"
            "Detected flaws:\n"
            + "\n".join(flaw_texts) +
            "\n\nPlease re-edit based on instruction, mask, and flaws."
        )

    # ======================
    # model load
    # ======================
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipeline.set_progress_bar_config(disable=True)

    # ======================
    # init score
    # ======================
    best_score = run_infer(source_image, edited_image, instruction)
    print("initial score:", best_score)

    current_image = edited_image

    # ======================
    # loop
    # ======================
    for it in range(max_iters):
        print(f"\n===== Iter {it+1} =====")

        subprocess.run(cmd1)
        subprocess.run(cmd2)

        with open(os.path.join(output_dir, "reasoning.json"), "r", encoding="utf-8") as f:
            last_line = list(f)[-1]

        obj = json.loads(last_line)
        flaws = parse_response(obj["response"])
        prompt = build_prompt(flaws, instruction)

        image1 = Image.open(source_image)
        image2 = Image.open(current_image)
        image3 = Image.open(mask_img).convert("RGB")

        with torch.inference_mode():
            out = pipeline(
                image=[image1, image2, image3],
                prompt=prompt,
                generator=torch.manual_seed(0),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=40,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )

        out_path = os.path.join(output_dir, f"iter_{it+1}.jpg")
        out.images[0].save(out_path)

        score = run_infer(source_image, out_path, instruction)
        print("score:", score, "best:", best_score)

        if score <= best_score:
            print("No improvement → stop.")
            break

        best_score = score
        current_image = out_path

    print("Done. Best score:", best_score)
    return best_score, current_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_image", type=str, required=True)
    parser.add_argument("--edited_image", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_iters", type=int, default=4)

    args = parser.parse_args()

    run_pipeline(
        args.source_image,
        args.edited_image,
        args.instruction,
        args.output_dir,
        args.max_iters
    )
