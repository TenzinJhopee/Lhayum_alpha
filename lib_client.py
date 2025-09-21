import requests
import json

class ApertusSwissLLM:
    def __init__(self, api_key=None, base_url="https://chat.publicai.co"):
        self.api_key = api_key
        self.base_url = base_url
        
    def generate_response(self, prompt, context="", max_tokens=500):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "swiss-llm",  # Adjust model name as needed
            "messages": [
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(f"{self.base_url}/chat/completions", 
                               headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"

# Initialize LLM
llm = ApertusSwissLLM(api_key="sk-768c82ef24604a4db381bf8588a73007")