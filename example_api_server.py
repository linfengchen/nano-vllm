"""
Example script demonstrating how to use the nanovllm OpenAI API server.
"""
import openai
import sys

# Configure the OpenAI client to use local server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # Not used but required by OpenAI client
)


def test_list_models():
    """Test listing available models."""
    print("=" * 50)
    print("Testing: List Models")
    print("=" * 50)
    models = client.models.list()
    for model in models.data:
        print(f"Model: {model.id}")
    print()


def test_chat_completion():
    """Test chat completion (non-streaming)."""
    print("=" * 50)
    print("Testing: Chat Completion (Non-streaming)")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model="qwen",  # Use your model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    print()


def test_chat_completion_streaming():
    """Test chat completion (streaming)."""
    print("=" * 50)
    print("Testing: Chat Completion (Streaming)")
    print("=" * 50)
    
    stream = client.chat.completions.create(
        model="qwen",
        messages=[
            {"role": "user", "content": "Tell me a short story about a robot."},
        ],
        max_tokens=150,
        temperature=0.8,
        stream=True,
    )
    
    print("Streaming response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def test_completion():
    """Test text completion (non-streaming)."""
    print("=" * 50)
    print("Testing: Text Completion (Non-streaming)")
    print("=" * 50)
    
    response = client.completions.create(
        model="qwen",
        prompt="Once upon a time, in a land far away,",
        max_tokens=50,
        temperature=0.7,
    )
    
    print(f"Response: {response.choices[0].text}")
    print(f"Usage: {response.usage}")
    print()


def test_completion_streaming():
    """Test text completion (streaming)."""
    print("=" * 50)
    print("Testing: Text Completion (Streaming)")
    print("=" * 50)
    
    stream = client.completions.create(
        model="qwen",
        prompt="The quick brown fox",
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )
    
    print("Streaming response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end="", flush=True)
    print("\n")


def main():
    """Run all tests."""
    try:
        test_list_models()
        test_chat_completion()
        test_chat_completion_streaming()
        test_completion()
        test_completion_streaming()
        
        print("=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure the API server is running:")
        print("  python -m nanovllm.entrypoints.api_server --model <model_path>")
        sys.exit(1)


if __name__ == "__main__":
    main()
