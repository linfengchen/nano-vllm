import os
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/data/python/vllm/models/Qwen3-4B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    print("loading")
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    print("loaded")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=40960)
    prompts = [
        "你是谁，你有什么本事？",
        # "你来讲一个夸父追日的故事",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    main()
