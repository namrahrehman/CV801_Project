# Quick Start Guide

## 1. Install Dependencies

```bash
conda activate project  # or your environment
pip install -r requirements.txt
```

## 2. Prepare Training Data

Convert your datasets to the format expected by the training scripts:

```bash
# Prepare both SLAKE and VQA_RAD datasets
python prepare_data.py \
    --data_dir data \
    --output data/train.json \
    --dataset both \
    --split train

# This creates data/train.json with format:
# {"text": "...", "instruction": "...", "output": "...", ...}
```

## 3. Train SFT Model

```bash
python train_sft.py \
    --cfg configs/sft.yaml \
    --data data/train.json
```

This will:
- Load Qwen2-VL-2B-Instruct
- Fine-tune with LoRA
- Save to `checkpoints/qwen2vl_sft/`

## 4. Prepare GRPO Data

For GRPO training, you need JSONL files with evaluation records:

```jsonl
{"image_path": "...", "question": "...", "gold": "...", "valid_json": true, "answer_correct": true, "caption": "...", "delta_cf_roi": 0.5, "delta_cf_rand": 0.2}
```

## 5. Train GRPO Model

```bash
python train_grpo.py \
    --base_model Qwen/Qwen2-VL-2B-Instruct \
    --sft_ckpt checkpoints/qwen2vl_sft \
    --data_jsonl data/grpo_data.jsonl \
    --out_dir checkpoints/qwen2vl_grpo \
    --steps 1000 \
    --batch_size 2 \
    --lr 1e-6 \
    --weights "0.2,0.4,0.2,0.2"
```

## Configuration

Edit `configs/sft.yaml` to customize:
- Model ID
- LoRA parameters
- Training hyperparameters
- Output directory

## GPU Setup

Use the `gpu_setup.ipynb` notebook to:
- Detect available GPUs
- Set CUDA_VISIBLE_DEVICES
- Test GPU functionality

## Troubleshooting

1. **Out of memory**: Reduce batch size in config
2. **Missing files**: Check that datasets are in `data/` directory
3. **Import errors**: Ensure `src/utils/` and `src/rewards/` directories exist

