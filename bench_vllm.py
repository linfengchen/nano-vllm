import os
import time
from random import randint, seed
# from nanovllm import LLM, SamplingParams
from vllm import LLM, TokensPrompt, SamplingParams


def main():
    seed(0)
    num_seqs = 10
    max_input_len = 10
    max_ouput_len = 1024

    path = os.path.expanduser("/data/python/vllm/models/Qwen3-4B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096, tensor_parallel_size=1)


    prompt_token_ids = [[randint(0, 10000) for _ in range(max_input_len)] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_ouput_len) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    
    # 为每个 prompt 创建 TokensPrompt 对象
    prompt_tokens = [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids]
    for i in range(10):
        t = time.time()
        outputs = llm.generate(prompt_tokens, sampling_params, use_tqdm=False)
        t = (time.time() - t)
        
        # 统计总 tokens
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / t
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
