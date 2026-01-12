# test_env.py
from dotenv import load_dotenv
import os

print("Before load_dotenv:")
print(f"  HF_TOKEN: {os.getenv('HF_TOKEN')}")
print(f"  HUGGINGFACE_TOKEN: {os.getenv('HUGGINGFACE_TOKEN')}")

result = load_dotenv()
print(f"\nload_dotenv() returned: {result}")

print("\nAfter load_dotenv:")
print(f"  HF_TOKEN: {os.getenv('HF_TOKEN')}")
print(f"  HUGGINGFACE_TOKEN: {os.getenv('HUGGINGFACE_TOKEN')}")

# Check if .env file exists
from pathlib import Path

env_file = Path('.env')
print(f"\n.env file exists: {env_file.exists()}")
if env_file.exists():
    print(f".env file size: {env_file.stat().st_size} bytes")

    # Try reading it directly
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"\n.env file content:\n{content}")
    except Exception as e:
        print(f"\nError reading .env: {e}")