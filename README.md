<h1 align="center">⚡ FlashRL ⚡</h1>
<p align="center"><b>Fast RL training with Quantized Rollouts </b>  
(<a href="https://fengyao.notion.site/flash-rl">Blog</a>)</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img src="https://img.shields.io/badge/python-3.10+-blue">
  <img src="https://img.shields.io/pypi/v/flash-llm-rl?color=green">  
</p>

<p align="center">
  <a href="#-flashrl-">What is FlashRL?</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage-guide">Usage Guide</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-road-map">Road Map</a> •
  <a href="#-citation">Citation</a>
</p>


[FlashRL](https://fengyao.notion.site/flash-rl) patches the inference backend to generate RL rollouts in INT8 \& FP8, with accurate rollout logprob. 

![DAPO 32B run](images/dapo_32b.png)
*Figure 1. **Left**: AIME accuracy of Qwen2.5-32B DAPO training with <span style="color: #ff7f0e;">INT8</span> and <span style="color: #1f77b4;">BF16</span> precisions for rollout generation using vLLM engine. **Right**: Training throughput (updates per hour) in the DAPO training (vLLM + BF16 FSDP).*

## ⚡ Quick Start

### Installation

```bash
pip install flash-llm-rl # need to be installed in all nodes in multi-node training
```

(Optional) there are two options to verify the FlashRL install: 1) set `FLASHRL_LOGGING_LEVEL` to `DEBUG` and compare the log with the [provided ones](#examples); 2) for more details / debugging, please follow the [Tutorial](/tutorial/README.md). 

### Rollout Generation w. FP8 Quantization
FlashRL is implemented as a plug-in-and-play manner, using [environment variables](#patcher) `FLASHRL_CONFIG` to control the quantization precision.

```bash 
# for single-node job
export FLASHRL_CONFIG=fp8
bash verl/examples/ppo_trainer/run_qwen2.5-32b.sh

# alternatively, for multi-node jobs via `ray submit`, fp8 online quantization will be turned on via
# > echo "  FLASHRL_CONFIG: 'fp8'" | tee -a verl/trainer/runtime_env.yaml # add `FLASHRL_CONFIG: 'fp8'` to runtime env
# > bash verl/recipe/dapo/run_dapo_qwen2.5_32b.sh # this can be any scripts
```

### Using with SGLang Rollout (veRL)
FlashRL now patches SGLang rollouts as well. Keep using the same `FLASHRL_CONFIG` env var and select the SGLang backend in veRL. By default, FlashRL defers to SGLang’s FP8 format defaults for both weights and KV cache.

```bash
pip install "verl[sglang]" flash-llm-rl
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# Option A: logprob patch only (no quant)
export FLASHRL_CONFIG=bf16
python -m verl.trainer.main_ppo actor_rollout_ref.rollout.name=sglang ...

# Option B: FP8 weights (KV dtype defers to SGLang unless overridden)
export FLASHRL_CONFIG=fp8
python -m verl.trainer.main_ppo actor_rollout_ref.rollout.name=sglang ...

# Optional: pick a specific FP8 KV dtype (otherwise SGLang decides)
# export FLASHRL_KV_CACHE_DTYPE=fp8_e5m2  # or fp8_e4m3
# Optional: disable FP8 KV cache override entirely
# export FLASHRL_DISABLE_FP8_KV=1
```

Notes:
- For very long prompts with input logprobs, prefer setting `logprob_start_len` to limit scoring scope (see SGLang docs).
- `FLASHRL_LMHEAD_FP32` has no effect on SGLang and is ignored with a warning.
- Optional: to avoid patching multiple backends when both vLLM and SGLang are installed, set `FLASHRL_BACKEND` to `sglang`, `vllm`, or `auto` (default is `auto`).

### RL Logprob Patch Only
Setting the config to `bf16` to extract precise logprob used in sampling without rollout quantization. This is useful for applying the [Truncated Importance Sampling](https://fengyao.notion.site/off-policy-rl?source=copy_link). 

```bash 
#  for single-node job
export FLASHRL_CONFIG=bf16
bash verl/examples/ppo_trainer/run_qwen2.5-32b.sh

# alternatively, for multi-node jobs via `ray submit`, RL Logprob Patch Only will be turned on via
# > echo "  FLASHRL_CONFIG: 'bf16'" | tee -a verl/trainer/runtime_env.yaml # add `FLASHRL_CONFIG: 'fp8'` to runtime env
# > bash verl/recipe/dapo/run_dapo_qwen2.5_32b.sh # this can be any scripts
```

## Usage Guide

FlashRL has 3 major functionality, `profiling`, `configure helper`, and `patcher`. 

### Profiling (optional for `fp8` and `bf16`)

This step is not needed for the native `fp8` online quantization supported by `vLLM`, and the logprog-only path `bf16`, and is needed for `int8` or `fp8_channel` quantization. Specifically, profilling compares a `bf16` model and a quantized model to decide how the online quantization should be performed for an updated model. Please find below an example for `Qwen/Qwen2.5-32B` and `Qwen/Qwen2.5-0.5B-Instruct`. The quantized model can be any `w8a8`/`fp8` model produced by `llm-compressor`. Note that, Redhat AI provides various [quantized models](https://huggingface.co/RedHatAI), and can be used here as in the 0.5B-Instruct example.

```bash
# for `Qwen/Qwen2.5-32B`
flashrl profile -m Qwen/Qwen2.5-32B -qm LiyuanLucasLiu/Qwen2.5-32B-quantized.w8a8 -o ${PROFILE_PATH:-"$HOME/profile.32b.pt"} --fn int8

# for `Qwen/Qwen2.5-0.5B-Instruct`
flashrl profile -m Qwen/Qwen2.5-0.5B-Instruct -qm RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8 -o ${PROFILE_PATH:-"$HOME/profile.0_5b.pt"} --fn int8
```

### Configure Helper (optional for `fp8` and `bf16`)

This step is not needed for the native `fp8` online quantization supported by `vLLM`, and the logprog-only path `bf16`, and is needed for `int8` or `fp8_channel` quantization. Specifically, configure helper creates a yaml file for the patcher to use. Please find below an example for `Qwen/Qwen2.5-32B` and `Qwen/Qwen2.5-0.5B-Instruct`. 

```bash
# for `Qwen/Qwen2.5-32B`
flashrl setup -m LiyuanLucasLiu/Qwen2.5-32B-quantized.w8a8 -p $HOME/profile.32b.pt --fn int8 -o ${CONFIG_PATH:-"$HOME/.flashrl_config.32b.yaml"}

# for `Qwen/Qwen2.5-0.5B-Instruct`
flashrl setup -m RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8 -p $HOME/profile.0_5b.pt --fn int8 -o ${CONFIG_PATH:-"$HOME/.flashrl_config.0_5b.yaml"}
```

### Patcher

Patcher would check the environment variable and operates accordingly. Please find the supported environment variables as below. 

|  Environment Variable | Usage | 
|--|--|
| `FLASHRL_CONFIG` | applies patcher if configured, supports `bf16`, `fp8`, local profile paths (e.g., `$HOME/.flashrl_config.32b.yaml`), and uploaded profiles (e.g., `LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-quantized.w8a8-RedHatAI/flashrl_config.yaml`) |
| `FLASHRL_LMHEAD_FP32` | if set to `1`, forcing `vLLM` conducting `lm head` compute in `bf16` (ignored for SGLang)
| `FLASHRL_KV_CACHE_DTYPE` | if set, explicitly selects the FP8 KV cache dtype (e.g., `fp8_e5m2` or `fp8_e4m3`); otherwise FlashRL defers to SGLang’s default |
| `FLASHRL_DISABLE_FP8_KV` | if set to `1`, FlashRL will not set `kv_cache_dtype` even if `FLASHRL_KV_CACHE_DTYPE` is provided |
| `FLASHRL_LOGGING_LEVEL` | set to `DEBUG` to turn on verbose logging for FlashRL functions |
| `FLASHRL_LOGGING_FILE` | if set, will save the log to files as well | 
| `FLASHRL_TEST_RELOAD` | functionality provided to test FlashRL install, check [this guide](./tutorial/verify_flashrl_install.md) for more details |

## Examples

| Run Detail | Script | Command | Log |
|--|--|--|--|
| INT8 Rollout for Qwen2.5-0.5B-Instruct on GSM8K | [Script](https://github.com/yaof20/verl/blob/flash-rl/recipe/flash_rl/gsm8k_qwen0_5b_int8.sh) | `bash recipe/flash_rl/gsm8k_qwen0_5b_int8.sh flash-int8-TIS-2 2` | [Wandb](https://wandb.ai/llychinalz/Flash-GSM8K?nw=2yfyyqo0fm) [Log](https://github.com/yaof20/verl/blob/flash-rl/recipe/flash_rl/logs/gsm8k_int8.log) |
| INT8 Rollout for Qwen2.5-32B-Instruct on DAPO | [Script](https://github.com/yaof20/verl/blob/flash-rl/recipe/flash_rl/dapo_qwen32b_int8.sh) | `bash recipe/flash_rl/dapo_qwen32b_int8.sh flash-int8-TIS-8 8` | [Wandb](https://wandb.ai/llychinalz/Flash-DAPO/?nw=w2j18d5w12) |
| FP8 Rollout for Qwen2.5-0.5B-Instruct on DAPO | [Script](https://github.com/yaof20/verl/blob/flash-rl/recipe/flash_rl/gsm8k_qwen0_5b_fp8.sh) | `bash recipe/flash_rl/gsm8k_qwen0_5b_fp8.sh flash-fp8-TIS-2 2` | [Wandb](https://wandb.ai/llychinalz/Flash-GSM8K?nw=cih3nmuhn8p) [Log](https://github.com/yaof20/verl/blob/flash-rl/recipe/flash_rl/logs/gsm8k_fp8.log) |
| FP8 Rollout for Qwen2.5-32B-Instruct on DAPO | [Script](https://github.com/yaof20/verl/blob/flash-rl/recipe/flash_rl/dapo_qwen32b_int8.sh) | `bash recipe/flash_rl/dapo_qwen32b_fp8.sh flash-fp8-TIS-8 8`| IN Progress |

## Tested Environments

Below are the combinations of the environments that we have tested on.

| Image | CUDA | Ray | vLLM | verl | flash-rl | GSM8K 8bit example | DAPO INT8 example |
|--|--|--|--|--|--|--|--|
| `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | 12.6 | 2.43.0 | 0.8.3 | [flash-rl](https://github.com/yaof20/verl/tree/flash-rl/recipe/flash_rl) | 1.0.1 | ✅ Tested | ✅ Tested |
| `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0` | 12.6 | 2.43.0 | 0.8.4 | [flash-rl](https://github.com/yaof20/verl/tree/flash-rl/recipe/flash_rl) | 1.0.1 | ✅ Tested | |
| `hiyouga/verl:ngc-th2.7.0-cu12.6-vllm0.9.1` | 12.6 | 2.43.0 | 0.9.1 | [flash-rl-vllm0.9.1](https://github.com/yaof20/verl/tree/flash-rl-vllm0.9.1/recipe/flash_rl) | 1.0.2| ✅ Tested | |
| `hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0` | 12.6 | 2.48.0 | 0.10.0 | [flash-rl-vllm0.9.1](https://github.com/yaof20/verl/tree/flash-rl-vllm0.9.1/recipe/flash_rl) | 1.0.2| ✅ Tested | |

### SGLang Compatibility
- Minimum: `sglang >= 0.3.6`
  - Reason: FP8 rollout flags are supported via server/engine args `--quantization fp8` (weights) and `--kv-cache-dtype` (KV cache). FlashRL does not force a KV dtype; SGLang’s default behavior applies unless you set `FLASHRL_KV_CACHE_DTYPE`.
  - Note: SGLang docs indicate FP8 KV cache requires CUDA 11.8+.
- Recommended: `sglang >= 0.4.x`
  - Stable `return_logprob`, `logprob_start_len`, and `top_logprobs_num` in native `/generate` and Python APIs. Use `logprob_start_len` to limit prompt scoring for long inputs.
  - You can gate Flash‑RL’s activation to a single backend via `FLASHRL_BACKEND=sglang|vllm|auto`.
  - High concurrency with `return_logprob=True` can increase memory pressure; consider reducing `--mem-fraction-static` and/or throttling.

On older SGLang versions, Flash‑RL falls back to logprob‑patch‑only behavior without modifying engine quantization.

## 🚧 Roadmap & Future Improvements

We're working on several improvements to Flash-RL:

- [ ] **Support of Other RL Toolkits**: Currently Flash-RL only supports `VeRL`, we are working on rolloing out support for other packages like `OpenRLHF`
- [x] **Support of Other LLM Inference Toolkits**: SGLang rollout support added (env-driven, via veRL SGLang backend)
- [ ] **Further Throughput Optimization**: We are working on implementing efficient GPU kernels to accelerate online quantization

## 📚 Citation

If you find our work useful, please cite us:

```bibtex
@misc{yao2025offpolicy,
  title = {Your Efficient RL Framework Secretly Brings You Off-Policy RL Training},
  url = {https://fengyao.notion.site/off-policy-rl},
  author = {Yao, Feng and Liu, Liyuan and Zhang, Dinghuai and Dong, Chengyu and Shang, Jingbo and Gao, Jianfeng},
  journal = {Feng Yao's Notion},
  year = {2025},
  month = aug,
}
@misc{yao2025flashrl,
  title = {FlashRL: 8Bit Rollouts, Full Power RL},
  url = {https://fengyao.notion.site/flash-rl},
  author = {Liu, Liyuan and Yao, Feng and Zhang, Dinghuai and Dong, Chengyu and Shang, Jingbo and Gao, Jianfeng},
  journal = {Feng Yao's Notion},
  year = {2025},
  month = aug,
}
```

## Questions?

If you have any questions related to the code or the blog, feel free to reach out to us at [Liyuan Liu](llychinalz@gmail.com) and [Feng Yao](fengyao@ucsd.edu).
