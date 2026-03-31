import requests
import json
import re

class OllamaExplainer:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{base_url}/api/generate"
        
    def extract_json(self, text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
        
    def generate_explanation(self, features):
        prompt = f"""
You are an expert in gastrointestinal endoscopy.

Based on segmentation results from a polyp detection model, generate a short clinical-style report.

Detected features:
{json.dumps(features, ensure_ascii=False)}

Feature description:
- area_ratio: proportion of polyp area relative to image
- location: relative position (left, right, center)
- shape: round or irregular
- size: small (<5mm), medium (5-10mm), large (>10mm)

Requirements:
- Write in Vietnamese
- Maximum 4 sentences

Return ONLY valid JSON:
{{
    "vi_tri": "...",
    "kich_thuoc": "...",
    "hinh_thai": "...",
    "nhan_dinh": "...",
    "canh_bao": "..."
}}

Notes:
- Do NOT make definitive diagnosis
- Only provide mild or moderate clinical suggestions
- Keep tone professional and objective
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.url, json=payload, timeout=30)
            result = response.json().get('response', "")

            parsed = self.extract_json(result)
            return parsed if parsed else result

        except Exception as e:
            return f"Lỗi kết nối Ollama: {str(e)}"