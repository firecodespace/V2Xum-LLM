import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No API key found")
    exit(1)

genai.configure(api_key=api_key)

print("\nAvailable Gemini models:")
print("="*60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✓ {model.name}")
        print(f"  Methods: {model.supported_generation_methods}")
        print()

