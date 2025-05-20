# LigherLLM

LigherLLM is a simplified version (under 2.5k loc) of [LightLLM](https://github.com/ModelTC/lightllm)'s first commit, designed for educational purposes to understand the core concepts of LLM inference server handling concurrent requests.

## Attribution

All rights and original work belong to the LightLLM team. This is an educational adaptation of their excellent work. For the original project, please refer to [README_ORIG.md](README_ORIG.md) and [LightLLM project](https://github.com/ModelTC/lightllm.git).

## Quick Start

We need a Linux machine with NVIDIA GPU card. First let's setup a virtual env and install dependencies:

```bash
git clone https://github.com/larme/lighterllm.git
cd lighterllm && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Then we can run an offline model inference example:

```bash
python -m examples.naive_model_executor meta-llama/Llama-3.2-3B-Instruct
```

We can edit `examples/naive_model_executor.py`, print out info to study how the model executor works with a naive scheduler.

We can also start a multiprocess server by running:

```bash
python -m lightllm.server.api_server --model_dir meta-llama/Llama-3.2-3B-Instruct --max_total_token_num 8192 --host 0.0.0.0
```

then query the server by running:

```bash
curl 127.0.0.1:8000/generate \
     -X POST \
     -d '{"inputs":"What is AI?","parameters":{"max_new_tokens":1024, "frequency_penalty":1}}' \
     -H 'Content-Type: application/json'
```

## Why LightLLM as the Base

I chose LightLLM as the foundation for this educational project because it incorporates most modern features essential for efficient LLM serving like continuous batching, KV cache, paged/token attention and multiprocess architecture (which will run tokenization, model inference, and detokenization in different processes and the model inference process will focus on model execution to maximize GPU utilization). In fact I think LightLLM is the first framework to use multiprocess architecture. [SGLang](https://github.com/sgl-project/sglang) is inspired by LightLLM and [vLLM](https://github.com/vllm-project/vllm) has also migrated to similar approach.

Also early version of lightllm's model executor kernels are written in [OpenAI Triton](https://github.com/triton-lang/triton) so it's relatively easier to understand than CUDA kernels.

## Changes from LightLLM's first commit

I made the
1. Simplifying the model executor codes:
   - Keep only Llama family model support
   - remove tensor parallelism
   - remove codes for old triton version
2. QoL improvements
   - Llama 3.1/3.2 support
   - safetensors support
   - auto set `eos_id`
   - huggingface model tag support
3. Offline model executor example and other materials for study (WIP)
