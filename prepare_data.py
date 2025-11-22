#!/usr/bin/env python3
"""
Prepare medical VQA datasets for SFT training.
Converts SLAKE and VQA_RAD datasets to JSON format expected by train_sft.py
"""
import json
import argparse
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_slake_dataset(data_dir, output_file, split="train"):
    """Prepare SLAKE dataset for training"""
    print(f"Loading SLAKE {split} dataset...")
    
    arrow_file = Path(data_dir) / "SLAKE" / split / "data-00000-of-00001.arrow"
    if not arrow_file.exists():
        print(f"Error: {arrow_file} not found")
        return []
    
    ds = load_dataset("arrow", data_files=str(arrow_file))
    records = []
    
    for item in ds['train']:
        # Format according to the prompt structure expected
        prompt = f"""You are a medical VQA assistant. Return ONLY valid JSON:
{{
 "caption":"<1-2 precise sentences>",
 "reasoning":["<step1>","<step2>","<step3>"],
 "boxes":[[x1,y1,x2,y2]],
 "answer":"<short answer>"
}}
Rules:
- Use integers for box coordinates
- Do not add extra keys or text
Question: "{item['question']}"
"""
        
        # Create training record for SFTTrainer
        # SFTTrainer expects "text" field or "instruction"/"input"/"output" format
        output_json = {
            "caption": f"Medical image showing {item.get('location', 'anatomical region')}",
            "reasoning": [
                f"Analyzing {item.get('modality', 'medical')} image",
                f"Question type: {item.get('answer_type', 'OPEN')}",
                f"Content type: {item.get('content_type', 'general')}"
            ],
            "boxes": [[0, 0, 100, 100]],  # Placeholder boxes
            "answer": item['answer']
        }
        
        # Format as text for SFTTrainer (instruction + output)
        text = prompt + "\n" + json.dumps(output_json, indent=2)
        
        record = {
            "text": text,  # SFTTrainer uses "text" field
            "instruction": prompt,
            "input": "",  # Empty input, question is in instruction
            "output": json.dumps(output_json),
            "image": item.get('img_name', ''),  # Image path/name
            "question": item['question'],
            "answer": item['answer'],
            "metadata": {
                "qid": item.get('qid', -1),
                "location": item.get('location', ''),
                "modality": item.get('modality', ''),
                "dataset": "SLAKE"
            }
        }
        records.append(record)
    
    print(f"Prepared {len(records)} records from SLAKE {split}")
    return records

def prepare_vqa_rad_dataset(data_dir, output_file, split="train"):
    """Prepare VQA_RAD dataset for training"""
    print(f"Loading VQA_RAD {split} dataset...")
    
    arrow_file = Path(data_dir) / "VQA_RAD" / split / "data-00000-of-00001.arrow"
    if not arrow_file.exists():
        print(f"Error: {arrow_file} not found")
        return []
    
    ds = load_dataset("arrow", data_files=str(arrow_file))
    records = []
    
    for item in ds['train']:
        prompt = f"""You are a medical VQA assistant. Return ONLY valid JSON:
{{
 "caption":"<1-2 precise sentences>",
 "reasoning":["<step1>","<step2>","<step3>"],
 "boxes":[[x1,y1,x2,y2]],
 "answer":"<short answer>"
}}
Rules:
- Use integers for box coordinates
- Do not add extra keys or text
Question: "{item['question']}"
"""
        
        # Convert image to base64 if it's a PIL Image
        image_str = ""
        if 'image' in item and item['image'] is not None:
            try:
                if isinstance(item['image'], Image.Image):
                    image_str = image_to_base64(item['image'])
                else:
                    image_str = str(item['image'])
            except:
                image_str = ""
        
        output_json = {
            "caption": "Medical radiology image",
            "reasoning": [
                "Analyzing radiology image",
                "Identifying relevant anatomical structures",
                "Formulating answer based on image content"
            ],
            "boxes": [[0, 0, 100, 100]],  # Placeholder boxes
            "answer": item['answer']
        }
        
        # Format as text for SFTTrainer
        text = prompt + "\n" + json.dumps(output_json, indent=2)
        
        record = {
            "text": text,  # SFTTrainer uses "text" field
            "instruction": prompt,
            "input": "",
            "output": json.dumps(output_json),
            "image": image_str,  # Base64 encoded image or path
            "question": item['question'],
            "answer": item['answer'],
            "metadata": {
                "dataset": "VQA_RAD"
            }
        }
        records.append(record)
    
    print(f"Prepared {len(records)} records from VQA_RAD {split}")
    return records

def main():
    parser = argparse.ArgumentParser(description="Prepare medical VQA datasets for SFT training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing datasets")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--dataset", type=str, choices=["SLAKE", "VQA_RAD", "both"], default="both",
                       help="Which dataset to prepare")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], default="train",
                       help="Which split to prepare")
    
    args = parser.parse_args()
    
    all_records = []
    
    if args.dataset in ["SLAKE", "both"]:
        slake_records = prepare_slake_dataset(args.data_dir, args.output, args.split)
        all_records.extend(slake_records)
    
    if args.dataset in ["VQA_RAD", "both"]:
        vqa_rad_records = prepare_vqa_rad_dataset(args.data_dir, args.output, args.split)
        all_records.extend(vqa_rad_records)
    
    # Write to JSON file
    print(f"\nWriting {len(all_records)} records to {args.output}...")
    with open(args.output, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"✓ Successfully prepared {len(all_records)} training records")
    print(f"✓ Output saved to: {args.output}")

if __name__ == "__main__":
    main()

