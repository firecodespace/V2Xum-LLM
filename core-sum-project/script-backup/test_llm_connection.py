"""Test LLM API connection"""

import sys
sys.path.append('/workspace/core-sum-project/src')

from llm.llm_client import get_llm_client

print("\n" + "="*60)
print("TESTING LLM CONNECTION")
print("="*60)

# Initialize client
try:
    client = get_llm_client()
    print(f"✓ Client initialized: {client.provider}/{client.model}\n")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    print("\n⚠️  Make sure you've added your API key to .env file")
    exit(1)

# Test simple prompt
print("Testing simple prompt...")
prompt = "Explain what video summarization is in one sentence."

try:
    response = client.generate(prompt)
    print(f"\n✓ Response received:")
    print(f"  Content: {response.content}")
    print(f"  Tokens used: {response.tokens_used}")
    print(f"  Model: {response.model}")
except Exception as e:
    print(f"\n✗ API call failed: {e}")
    exit(1)

# Test structured JSON output
print("\n" + "-"*60)
print("Testing structured JSON output...")

system_prompt = "You are a helpful assistant that responds in JSON format."
prompt = """
Analyze this video structure and respond in JSON:

Scene 1: "Introduction" (0-100 frames)
Scene 2: "Main content" (100-400 frames)
Scene 3: "Conclusion" (400-500 frames)

Respond with JSON containing:
- scene_rankings: list of {scene_id, importance (1-10), reason}
"""

try:
    response = client.generate(
        prompt, 
        system_prompt=system_prompt,
        response_format={"type": "json_object"}
    )
    
    # Parse JSON
    data = client.parse_json_response(response)
    
    print(f"\n✓ JSON response received:")
    print(f"  Keys: {list(data.keys())}")
    print(f"  Content preview:")
    import json
    print(json.dumps(data, indent=2)[:500])

except Exception as e:
    print(f"\n✗ Structured output failed: {e}")
    exit(1)

print("\n" + "="*60)
print("✓✓✓ LLM CONNECTION TEST PASSED ✓✓✓")
print("="*60)
print("\nReady for Stage 2 implementation!")
print()
