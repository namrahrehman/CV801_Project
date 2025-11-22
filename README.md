# Medical VQA Model Training Pipeline

A two-stage training pipeline for fine-tuning vision-language models on medical visual question answering (VQA) tasks.

## Project Structure

```
project/
├── configs/
│   └── sft.yaml              # SFT training configuration
├── data/
│   ├── SLAKE/                # SLAKE medical VQA dataset
│   ├── VQA_RAD/              # VQA-RAD medical VQA dataset
│   └── outputs_cfproxy/     # Counterfactual proxy outputs
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── prompting.py     # Prompt generation utilities
│   └── rewards/
│       ├── __init__.py
│       └── faithfulness_reward.py  # Reward computation for GRPO
├── train_sft.py              # Supervised Fine-Tuning script
├── train_grpo.py             # Group Relative Policy Optimization script
├── prepare_data.py            # Dataset preparation script
├── VR1_VAQRAD.ipynb          # Evaluation notebook for VQA-RAD dataset
├── VRI_SLAKE.ipynb           # Evaluation notebook for SLAKE dataset
├── gpu_setup.ipynb           # GPU setup and detection notebook
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install transformers trl peft datasets torch sentence-transformers pillow
```

2. Prepare training data:
```bash
# Prepare both datasets for training
python prepare_data.py --output data/train.json --dataset both --split train

# Or prepare individually
python prepare_data.py --output data/slake_train.json --dataset SLAKE --split train
python prepare_data.py --output data/vqa_rad_train.json --dataset VQA_RAD --split train
```

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune the base model on medical VQA data using LoRA:

```bash
python train_sft.py \
    --cfg configs/sft.yaml \
    --data data/train.json
```

This will:
- Load Qwen2-VL-2B-Instruct as base model
- Apply LoRA fine-tuning
- Save checkpoints to `checkpoints/qwen2vl_sft/`

### Stage 2: Group Relative Policy Optimization (GRPO)

Further improve the model using reinforcement learning:

```bash
python train_grpo.py \
    --base_model Qwen/Qwen2-VL-2B-Instruct \
    --sft_ckpt checkpoints/qwen2vl_sft \
    --data_jsonl data/grpo_data.jsonl \
    --out_dir checkpoints/qwen2vl_grpo \
    --steps 1000 \
    --batch_size 2 \
    --lr 1e-6 \
    --weights "0.2,0.4,0.2,0.2"  # fmt,ans,cons,cf
```

## Dataset Format

### SFT Training Data (JSON)

Each line should be a JSON object with:
```json
{
  "instruction": "You are a medical VQA assistant...",
  "input": "",
  "output": "{\"caption\":\"...\",\"reasoning\":[...],\"boxes\":[[...]],\"answer\":\"...\"}",
  "image": "path/to/image.jpg",
  "question": "What modality is used?",
  "answer": "MRI"
}
```

### GRPO Training Data (JSONL)

Each line should be a JSON object with:
```json
{
  "image_path": "path/to/image.jpg",
  "question": "What modality is used?",
  "gold": "MRI",
  "valid_json": true,
  "answer_correct": true,
  "caption": "Medical image showing...",
  "delta_cf_roi": 0.5,
  "delta_cf_rand": 0.2
}
```

## Configuration

Edit `configs/sft.yaml` to adjust:
- Model ID
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (batch size, learning rate, etc.)
- Output directory

## Reward Function

The GRPO training uses a faithfulness reward with four components:
- **Format (w_fmt)**: Valid JSON output (0.2)
- **Answer (w_ans)**: Correct answer (0.4)
- **Consistency (w_cons)**: Semantic similarity between question and caption (0.2)
- **Counterfactual (w_cf)**: Faithfulness metric (0.2)

## Evaluation Notebooks

The project includes evaluation notebooks that test Qwen2-VL-2B-Instruct's weaknesses on medical VQA datasets:

### `VR1_VAQRAD.ipynb` - VQA-RAD Evaluation

Tests Qwen2-VL-2B-Instruct on the VQA-RAD dataset and evaluates:

1. **Answer Accuracy**:
   - Think→Answer accuracy (~38.5%)
   - Caption→Reason→Answer accuracy (~35.9%)
   - JSON parsing success rate (~91.5%)
   - Bounding box detection rate (~75.5%)

2. **Risk-Coverage Analysis**:
   - Caption-question consistency at different thresholds (τ=0.3-0.6)
   - Measures alignment between generated captions and questions
   - Coverage vs accuracy trade-offs

3. **Counterfactual Faithfulness Testing**:
   - **ROI-CF (Region of Interest Counterfactuals)**: Tests if model changes answer when key diagnostic regions are inpainted (~39% flip rate)
   - **Random Region CF**: Tests if model changes answer when random regions are inpainted (baseline comparison)
   - High flip rates indicate the model may not be truly attending to the correct image regions

### `VRI_SLAKE.ipynb` - SLAKE Evaluation

Similar evaluation framework for the SLAKE dataset:

1. **Answer Accuracy**: Tests model performance on SLAKE medical VQA questions
2. **Caption Consistency**: Evaluates caption-question alignment with expanded medical vocabulary
3. **Counterfactual Testing**: ROI-based faithfulness testing with medical image regions

### Key Weaknesses Identified

These notebooks reveal several weaknesses in the base Qwen2-VL-2B-Instruct model:

- **Low accuracy** (~35-39%) on medical VQA tasks
- **Poor JSON compliance** - not all outputs are valid JSON
- **Inconsistent captions** - captions don't always align with questions
- **Low faithfulness** - high ROI-CF flip rates suggest the model may not be properly attending to diagnostic regions
- **Missing bounding boxes** - model doesn't always provide spatial localization

These weaknesses motivate the two-stage training pipeline (SFT + GRPO) to improve:
- Answer accuracy through supervised fine-tuning
- Faithfulness through counterfactual-based reinforcement learning
- Format compliance through structured output training

## Notes

- The datasets include images that need to be processed
- For vision-language models, ensure image paths are accessible
- GPU memory: Use the GPU setup notebook to manage GPU resources
- The model outputs structured JSON with caption, reasoning, bounding boxes, and answer
- Run evaluation notebooks before training to establish baseline performance

