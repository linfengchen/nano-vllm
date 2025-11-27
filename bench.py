import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    seed(0)
    num_seqs = 10
    max_input_len = 10
    max_ouput_len = 1024

    path = os.path.expanduser("/data/python/vllm/models/Qwen3-4B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(max_input_len)] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_ouput_len) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    
    # 前几轮warmup
    for i in range(3):
        t = time.time()
        o = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = (time.time() - t)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
        print(f"[Warmup {i+1}] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    
    # 开始profiling的第4轮
    print("\n开始profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            t = time.time()
            o = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
            t = (time.time() - t)
            total_tokens = sum(sp.max_tokens for sp in sampling_params)
            throughput = total_tokens / t
            print(f"[Profile] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    
    # 保存profiling结果
    print("\n保存profiling结果...")
    prof.export_chrome_trace("profile_trace.json")
    print("Chrome trace已保存到: profile_trace.json")
    
    # 打印算子级别的统计信息
    print("\n=== 按CPU时间排序的Top 20算子 ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    print("\n=== 按CUDA时间排序的Top 20算子 ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 保存详细报告到文件
    with open("profile_report.txt", "w") as f:
        f.write("=== 按CPU时间排序的算子 ===\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        f.write("\n\n=== 按CUDA时间排序的算子 ===\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    print("详细报告已保存到: profile_report.txt")
    
    # 继续运行剩余的benchmark
    for i in range(6):
        t = time.time()
        o = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = (time.time() - t)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
        print(f"[Iter {i+4}] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
