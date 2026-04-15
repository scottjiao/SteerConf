
# SteerConf: Steering LLMs for Confidence Elicitation

[![arXiv](https://img.shields.io/badge/arXiv-2503.02863-b31b1b.svg)](https://arxiv.org/abs/2503.02863)
[![Venue](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> **SteerConf: Steering LLMs for Confidence Elicitation**
>
> [Ziang Zhou](https://github.com/scottjiao), Tianyuan Jin, Jieming Shi, Qing Li
>
> **NeurIPS 2025** &nbsp;|&nbsp; [Paper (arXiv)](https://arxiv.org/abs/2503.02863)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1: Query LLMs with Steering Prompts](#step-1-query-llms-with-steering-prompts)
  - [Step 2: Extract Answers](#step-2-extract-answers)
  - [Step 3: Aggregate Results across Steering Levels](#step-3-aggregate-results-across-steering-levels)
  - [Step 4: Visualization and Evaluation](#step-4-visualization-and-evaluation)
  - [Running the Full Pipeline](#running-the-full-pipeline)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [Citation](#citation)
- [License](#license)

---

## Overview

**SteerConf** is a framework for eliciting and calibrating confidence estimates from Large Language Models (LLMs). It introduces *steering prompts* that systematically vary between cautious and confident directions to influence the verbalized confidence of LLMs, and then aggregates these multi-level confidence signals to produce better-calibrated uncertainty estimates.

The key idea is to prompt an LLM at multiple steering levels (e.g., *very cautious*, *cautious*, *vanilla*, *confident*, *very confident*) and aggregate the resulting confidence values. This approach improves calibration metrics such as ECE, AUROC, AUPRC, and AURC compared to standard single-query verbalized confidence.

---

## Repository Structure

```
SteerConf/
├── query_vanilla_or_cot.py          # Query LLMs with vanilla/CoT prompts at different steering levels
├── extract_answers.py               # Extract predicted answers and confidence from raw LLM responses
├── vis_aggregated_conf.py           # Visualize and evaluate aggregated confidence metrics
├── start_llama_server.py            # Start a local Llama model server (socket-based)
├── scripts/
│   ├── confidence_after_answer_important_decisions_llama.py
│   │                                # End-to-end pipeline script for Llama models
│   ├── confidence_after_answer_important_decisions_llama_summarized.sh
│   │                                # Shell script for visualization across steering levels
│   └── result_aggregate.py          # Aggregate results from all steering levels into a single file
├── utils/
│   ├── llm_query_helper.py          # LLM API interface (OpenAI-compatible and Llama)
│   ├── api_llama.py                 # Socket client for the local Llama server
│   ├── dataset_loader.py            # Dataset loading utilities
│   ├── compute_metrics.py           # Calibration metric computation (ECE, AUROC, AUPRC, AURC)
│   ├── extract_result_lib.py        # Answer/confidence extraction from LLM text responses
│   ├── extract_result_lib2.py       # Additional extraction utilities
│   └── inference.py                 # Model inference utilities (FastChat-based)
├── dataset/                         # Dataset directory (not tracked; see Setup)
├── api_key.txt                      # API key file (not tracked; see Setup)
└── README.md
```

---

## Requirements

- Python ≥ 3.8
- Core dependencies:
  - `openai`
  - `transformers`
  - `torch`
  - `scikit-learn`
  - `netcal`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `adjustText`
- For local Llama inference:
  - `huggingface_hub`
  - `fastchat` (optional, for FastChat-based inference)
  - A GPU with sufficient VRAM (≥ 24 GB recommended for 4-bit quantized 70B models)

Install dependencies:

```bash
pip install openai transformers torch scikit-learn netcal numpy pandas matplotlib seaborn adjustText huggingface_hub
```

---

## Setup

### 1. Prepare the API key

Create a file named `api_key.txt` in the project root containing your LLM provider API key:

```bash
echo "your-api-key-here" > api_key.txt
```

### 2. Configure the API endpoint

Edit `utils/llm_query_helper.py` and set the `base_site` variable to your LLM provider's base URL:

```python
base_site = "https://your-api-endpoint.example.com"
```

### 3. Prepare datasets

Place the benchmark datasets under the `dataset/` directory. The expected layout is:

```
dataset/
├── BigBench/
│   ├── sport_understanding.json
│   ├── object_counting.json
│   ├── strategy_qa.json
│   └── date_understanding.json
├── MMLU/
│   ├── business_ethics_test.csv
│   └── professional_law_test.csv
├── grade_school_math/
│   └── data/
│       └── test.jsonl
└── ScienceQA/                      # (optional)
    ├── pid_splits.json
    └── problems.json
```

### 4. (Optional) Start the local Llama server

If you plan to use a locally hosted Llama model, first log in to Hugging Face and start the server:

```bash
python start_llama_server.py
```

The server listens on `localhost:12370` by default.

---

## Usage

The pipeline consists of four stages: **Query → Extract → Aggregate → Evaluate**. Each stage can be run independently.

### Step 1: Query LLMs with Steering Prompts

Use `query_vanilla_or_cot.py` to query an LLM at a specific steering level. The key arguments are:

```bash
python query_vanilla_or_cot.py \
    --dataset_name <DATASET_NAME> \
    --data_path <PATH_TO_DATASET> \
    --output_file <OUTPUT_JSON> \
    --model_name <MODEL_NAME> \
    --task_type <multi_choice_qa|open_number_qa> \
    --prompt_type <PROMPT_TYPE> \
    --sampling_type self_random \
    --num_ensemble 1 \
    --temperature_for_ensemble 0.7 \
    [--use_cot]
```

**Steering levels** (controlled by `--prompt_type`):

| Prompt Type | Steering Direction |
|---|---|
| `confidence_after_answer_important_decisions_very_cautious` | Very cautious (lowest confidence) |
| `confidence_after_answer_important_decisions_cautious` | Cautious |
| `vanilla` | Neutral (no steering) |
| `confidence_after_answer_important_decisions_confident` | Confident |
| `confidence_after_answer_important_decisions_very_confident` | Very confident (highest confidence) |

**Example** (Llama-3.3-70B-Instruct with CoT, cautious steering, on SportsUND):

```bash
python query_vanilla_or_cot.py \
    --dataset_name BigBench_sportUND \
    --data_path dataset/BigBench/sport_understanding.json \
    --output_file final_output/cautious_self_random_1_cot/Llama-3.3-70b-instruct/BigBench_sportUND/result.json \
    --model_name Llama-3.3-70b-instruct \
    --task_type multi_choice_qa \
    --prompt_type confidence_after_answer_important_decisions_cautious \
    --sampling_type self_random \
    --num_ensemble 1 \
    --temperature_for_ensemble 0.7 \
    --use_cot
```

### Step 2: Extract Answers

Parse the raw LLM responses to extract predicted answers and confidence values:

```bash
python extract_answers.py \
    --input_file <RESULT_JSON> \
    --model_name <MODEL_NAME> \
    --dataset_name <DATASET_NAME> \
    --task_type <TASK_TYPE> \
    --prompt_type <PROMPT_TYPE> \
    --sampling_type self_random \
    --num_ensemble 1 \
    [--use_cot]
```

This produces a `*_processed.json` file alongside the input file.

### Step 3: Aggregate Results across Steering Levels

After querying at all five steering levels, aggregate them into a unified result file:

```bash
python scripts/result_aggregate.py \
    --cot "cot" \
    --model Llama-3.3-70b-instruct \
    --datasets_list "0123456"
```

The `--datasets_list` argument is a string of dataset indices:

| Index | Dataset |
|---|---|
| `0` | BigBench_sportUND |
| `1` | Business_Ethics |
| `2` | Professional_Law |
| `3` | BigBench_ObjectCounting |
| `4` | BigBench_strategyQA |
| `5` | BigBench_DateUnderstanding |
| `6` | GSM8K |

### Step 4: Visualization and Evaluation

Compute calibration metrics and generate visualizations:

```bash
python vis_aggregated_conf.py \
    --input_file <PROCESSED_JSON> \
    --model_name <MODEL_NAME> \
    --dataset_name <DATASET_NAME> \
    --task_type <TASK_TYPE> \
    --prompt_type <PROMPT_TYPE> \
    --sampling_type self_random \
    --num_ensemble 1 \
    [--use_cot]
```

The following metrics are computed (see `utils/compute_metrics.py`):

- **Accuracy** — Prediction correctness
- **AUROC** — Area Under the ROC Curve
- **AUPRC** — Area Under the Precision–Recall Curve (positive and negative)
- **AURC** — Area Under the Risk–Coverage Curve
- **ECE** — Expected Calibration Error

Output files are saved under the `visual/` and `log/` subdirectories of the input file's parent directory.

### Running the Full Pipeline

To run the full pipeline (query → extract → evaluate) across all steering levels and all datasets for Llama-3.3-70B-Instruct with CoT:

```bash
python scripts/confidence_after_answer_important_decisions_llama.py
```

To run visualization on the aggregated (summarized) results:

```bash
bash scripts/confidence_after_answer_important_decisions_llama_summarized.sh
```

---

## Supported Datasets

| Dataset | Task Type | Source |
|---|---|---|
| ScienceQA | Multiple-choice QA | [ScienceQA](https://scienceqa.github.io/) |
| BigBench — Sport Understanding | Multiple-choice QA | [BIG-Bench](https://github.com/google/BIG-bench) |
| BigBench — Date Understanding | Multiple-choice QA | [BIG-Bench](https://github.com/google/BIG-bench) |
| BigBench — Strategy QA | Multiple-choice QA | [BIG-Bench](https://github.com/google/BIG-bench) |
| BigBench — Object Counting | Open-ended numerical QA | [BIG-Bench](https://github.com/google/BIG-bench) |
| MMLU — Business Ethics | Multiple-choice QA | [MMLU](https://github.com/hendrycks/test) |
| MMLU — Professional Law | Multiple-choice QA | [MMLU](https://github.com/hendrycks/test) |
| GSM8K | Open-ended numerical QA | [GSM8K](https://github.com/openai/grade-school-math) |

---

## Supported Models

| Model | Interface |
|---|---|
| GPT-3.5 / GPT-4 / GPT-4o | OpenAI-compatible API (via `utils/llm_query_helper.py`) |
| Llama-3.3-70B-Instruct | Local server (via `start_llama_server.py` + `utils/api_llama.py`) |

To add support for additional models, extend the model dispatch logic in `utils/llm_query_helper.py`.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhou2025steerconf,
  title={SteerConf: Steering LLMs for Confidence Elicitation},
  author={Zhou, Ziang and Jin, Tianyuan and Shi, Jieming and Li, Qing},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## License

This project is released for academic and research purposes. Please refer to the repository for specific license terms.

