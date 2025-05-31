import httpx
import asyncio

async def test_llm_connection():
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": "Qwen_QwQ-32B-Q6_K_L",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say hello!"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            response.raise_for_status()
            print("Response:", response.json())
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_llm_connection()) 