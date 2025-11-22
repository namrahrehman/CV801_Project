# ~/medfaith/src/utils/prompting.py
def make_prompt(question: str) -> str:
    return f"""You are a medical VQA assistant. Return ONLY valid JSON:
{{
 "caption":"<1-2 precise sentences>",
 "reasoning":["<step1>","<step2>","<step3>"],
 "boxes":[[x1,y1,x2,y2]],
 "answer":"<short answer>"
}}
Rules:
- Use integers for box coordinates
- Do not add extra keys or text
Question: "{question}"
"""
