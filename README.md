<div align="center">

# PertAL

**An Efficient and Robust Active Learning Framework for Budget-Constrained Single-Cell Perturbation Screens**

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-orange)]()

Prioritize perturbations intelligently — match full-dataset performance with ~1/3 of the training data.

[Paper]() | [Data](https://drive.google.com/drive/folders/1Hh00_cO6oRBOU6kAzhyblaJ0xhJZfyGB?dmr=1&ec=wgc-drive-globalnav-goto) | [GitHub](https://github.com/JieZheng-ShanghaiTech/PertAL)

</div>

---

![Overview](Overview_of_PertAL.png)

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Download data](#download-data)
- [Generate LLM priors](#generate-llm-priors)
- [Run experiments](#run-experiments)
- [Use Hydra configs](#use-hydra-configs)
- [Reproduce paper results](#reproduce-paper-results)
- [Evaluation metrics](#evaluation-metrics)
- [Baselines](#baselines)
- [Project structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

PertAL is an active learning framework that guides low-budget single-cell perturbation screening. It selects the most informative perturbations by combining three complementary scoring modules into a single composite score:

**S_final = α · S_LLM + β · S_Div + γ · S_Sen** (default: α=0.2, β=1, γ=1)

| Module                          | Score | What it captures                                                                                                     |
| ------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| LLM-driven biological reasoning | S_LLM | GPT-4.1-mini scores each gene on hubness (0–10), phenotype impact (0–10), and knowledge scarcity (0–10)              |
| Multi-view diversity assessment | S_Div | Fuses scFM embeddings (scGPT) with cross-modal kernels (Perturb-seq, OPS) via mean kernel fusion and BMDAL selection |
| Gradient-based sensitivity      | S_Sen | Measures model prediction sensitivity to each candidate perturbation using gradient information                      |

The backbone predictor is [GEARS](https://github.com/snap-stanford/GEARS), a GNN-based model for perturbation response prediction.

---

## Installation

### Set up the environment

```bash
conda create -n pertAL python=3.10
conda activate pertAL
```

### Install the package

```bash
pip install -e .
```

### Install PyTorch Geometric CUDA extensions

Adjust the `--find-links` URL to match your CUDA version. The example below targets CUDA 12.4 with PyTorch 2.4:

```bash
pip install -r requirements-cuda.txt -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

> Browse [data.pyg.org/whl](https://data.pyg.org/whl/) for other CUDA/PyTorch combinations.

### (Optional) Install Hydra support

```bash
pip install -e ".[hydra]"
```

---

## Download data

PertAL uses three Perturb-seq datasets (top 1000 HVGs each):

| Dataset       | Cell line | Perturbations         |
| ------------- | --------- | --------------------- |
| Replogle K562 | K562      | ~1000 essential genes |
| Replogle RPE1 | RPE1      | ~1000 essential genes |
| Adamson       | K562      | Smaller set           |

**Step 1.** Download the data from [Google Drive](https://drive.google.com/drive/folders/1Hh00_cO6oRBOU6kAzhyblaJ0xhJZfyGB?dmr=1&ec=wgc-drive-globalnav-goto).

**Step 2.** Place everything under `data/` at the project root. The expected layout:

```
data/
├── <dataset>.h5ad files
├── <dataset>_kernels/
│   └── knowledge_kernels_1k/
│       ├── scgpt_blood/
│       ├── rpe1_kernel/
│       ├── k562_kernel/
│       ├── ops_A549_kernel/
│       ├── ops_HeLa_HPLM_kernel/
│       ├── ops_HeLa_DMEM_kernel/
│       └── biogpt_kernel/
└── Prior_kernel_preprocess.ipynb
```

Each dataset uses a different subset of cross-modal kernels:

| Dataset       | Extra kernels                                                                           |
| ------------- | --------------------------------------------------------------------------------------- |
| replogle_k562 | rpe1_kernel, ops_A549_kernel, biogpt_kernel, ops_HeLa_HPLM_kernel, ops_HeLa_DMEM_kernel |
| replogle_rpe1 | k562_kernel, ops_A549_kernel, ops_HeLa_HPLM_kernel, ops_HeLa_DMEM_kernel                |
| adamson_k562  | ops_A549_kernel, ops_HeLa_HPLM_kernel, ops_HeLa_DMEM_kernel                             |

> The scFM kernel (`scgpt_blood` by default) is specified via `--prior_scfm_kernel`.

### Build custom kernels

Use the provided notebook to construct your own feature kernels:

```bash
jupyter notebook data/Prior_kernel_preprocess.ipynb
```

> OPS data is sourced from [Broad PERISCOPE](https://github.com/broadinstitute/2022_PERISCOPE).

---

## Generate LLM priors

Two scripts in `llm/` produce the LLM-based gene scores. Both use the OpenAI-compatible API format, so they work with GPT-4.1-mini, Qwen2.5-7B-Instruct, DeepSeek-R1, or any compatible endpoint.

### Step 1 — Score genes

`llm/LLM_prior_extractor_async.py` asynchronously scores every candidate gene on three dimensions (hubness, phenotype impact, knowledge scarcity).

1. Open the script and fill in the config block:

```python
API_KEY = "your-api-key"
PLATFORM_BASE_URL = "https://your-api-endpoint.com/v1"
MODEL_NAME = "gpt-4.1-mini"
CELL_LINE = "K562"
DATASET_NAME = "Replogle"
seed = 1
```

2. Prepare a gene list file named `<DATASET_NAME>_<CELL_LINE>_genes.txt` (one gene per line).

3. Run:

```bash
python llm/LLM_prior_extractor_async.py
```

Output is a CSV with per-gene scores and a composite `V_llm_raw` column.

### Step 2 — Judge and revise scores

`llm/LLM_prior_judge_async.py` sends each gene's scores to a "judge" LLM that evaluates plausibility, produces rationality scores, and outputs revised values.

1. Configure the script:

```python
API_KEY = "your-api-key"
BASE_URL = "https://your-api-endpoint.com/v1"
MODEL = "gpt-4.1-mini"
CELL_LINE = "K562"
DATASET = "Replogle"
```

2. Make sure the input CSV from Step 1 is in the expected path.

3. Run:

```bash
python llm/LLM_prior_judge_async.py
```

> This script supports resume — it skips genes already present in the output CSV.

**LLM parameters used in the paper:** temperature=0.1, top_p=1.0.

---

## Run experiments

The main entry point is `run.py`.

### Basic usage

```bash
python run.py \
  --strategy PertAL \
  --device 0 \
  --dataset_name replogle_k562 \
  --seed 1 \
  --prior_scfm_kernel scgpt_blood \
  --llm_name gpt41-mini \
  --llm_weight 0.2
```

### Parameters

| Argument              | Default         | Description                                             |
| --------------------- | --------------- | ------------------------------------------------------- |
| `--strategy`          | `PertAL`        | Active learning strategy                                |
| `--device`            | `1`             | GPU index                                               |
| `--dataset_name`      | `replogle_k562` | One of `replogle_k562`, `replogle_rpe1`, `adamson_k562` |
| `--seed`              | `5`             | Random seed                                             |
| `--prior_scfm_kernel` | `scgpt_blood`   | scFM kernel name                                        |
| `--llm_name`          | `gpt41-mini`    | LLM prior identifier                                    |
| `--llm_weight`        | `0.2`           | Weight α for S_LLM                                      |

Internal defaults (set in `run.py`):

| Parameter        | Value |
| ---------------- | ----- |
| `n_init_labeled` | 100   |
| `n_round`        | 5     |
| `n_query`        | 100   |
| `hidden_size`    | 64    |
| `epochs`         | 20    |
| `batch_size`     | 256   |

Results are saved to `./results/`.

---

## Use Hydra configs

An alternative entry point uses [Hydra](https://hydra.cc/) for structured configuration.

### Install the extra

```bash
pip install -e ".[hydra]"
```

### Run with defaults

```bash
python run/pipeline/train.py
```

### Override parameters

```bash
python run/pipeline/train.py dataset=replogle_k562 device=0 seed=1
```

### Config hierarchy

```
run/conf/
├── config.yaml          # Top-level defaults
├── dataset/             # replogle_k562, replogle_rpe1, adamson_k562
├── model/               # gears
├── strategy/            # pertal
├── llm/                 # gpt41-mini
└── dir/                 # default paths
```

---

## Reproduce paper results

<details>
<summary><strong>Full commands for all datasets and seeds</strong></summary>

**Hardware:** single NVIDIA Tesla V100 GPU (32 GB).

**Settings:** 5 AL rounds, 100 perturbations queried per round (large datasets), seeds 1–5, α=0.2, β=1, γ=1.

### Replogle K562

```bash
for SEED in 1 2 3 4 5; do
  python run.py \
    --strategy PertAL \
    --device 0 \
    --dataset_name replogle_k562 \
    --seed $SEED \
    --prior_scfm_kernel scgpt_blood \
    --llm_name gpt41-mini \
    --llm_weight 0.2
done
```

### Replogle RPE1

```bash
for SEED in 1 2 3 4 5; do
  python run.py \
    --strategy PertAL \
    --device 0 \
    --dataset_name replogle_rpe1 \
    --seed $SEED \
    --prior_scfm_kernel scgpt_blood \
    --llm_name gpt41-mini \
    --llm_weight 0.2
done
```

### Adamson K562

```bash
for SEED in 1 2 3 4 5; do
  python run.py \
    --strategy PertAL \
    --device 0 \
    --dataset_name adamson_k562 \
    --seed $SEED \
    --prior_scfm_kernel scgpt_blood \
    --llm_name gpt41-mini \
    --llm_weight 0.2
done
```

### Using Hydra

```bash
for SEED in 1 2 3 4 5; do
  python run/pipeline/train.py dataset=replogle_k562 device=0 seed=$SEED
  python run/pipeline/train.py dataset=replogle_rpe1  device=0 seed=$SEED
  python run/pipeline/train.py dataset=adamson_k562   device=0 seed=$SEED
done
```

</details>

---

## Evaluation metrics

All metrics are computed on the **top 20 differentially expressed genes** per perturbation:

| Metric              | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| Pearson correlation | Correlation between predicted and observed expression changes        |
| MSE                 | Mean squared error between predicted and observed expression changes |

---

## Baselines

PertAL is benchmarked against 10 active learning baselines:

| Strategy       | Type             |
| -------------- | ---------------- |
| Random         | Uncertainty-free |
| BALD           | Bayesian         |
| BatchBALD      | Bayesian         |
| BADGE          | Gradient         |
| ACS-FW         | Gradient         |
| Core-Set       | Diversity        |
| LCMD           | Diversity        |
| KMeansSampling | Diversity        |
| TypiClust      | Diversity        |
| IterPert       | Domain-specific  |

Baseline implementations used in the paper are from the respective original repositories and are not included in this codebase. Only the `PertAL` strategy is provided here.

---

## Project structure

```
PertAL/
├── run.py                             # Main CLI entry point
├── pyproject.toml                     # PEP 621 packaging
├── requirements-cuda.txt              # PyG CUDA wheel links
├── Overview_of_PertAL.png             # Architecture figure
├── pertal/                            # Core package
│   ├── pertal.py                      # PertAL orchestrator class
│   ├── config.py                      # Central configuration/constants
│   ├── data_pert.py                   # Dataset loading and handling
│   ├── utils.py                       # Utility functions
│   ├── nets_pert.py                   # Neural network definitions
│   ├── scoring/                       # Scoring modules
│   │   ├── base.py                    # Base scorer / composite scoring
│   │   ├── gradient.py                # Gradient-based sensitivity (S_Sen)
│   │   └── llm.py                     # LLM prior scoring (S_LLM)
│   ├── AL_strategies/                 # Active learning strategies
│   │   ├── registry.py                # Strategy factory/registry
│   │   ├── active_learning.py         # PertAL strategy implementation
│   │   └── strategy.py               # Base strategy class
│   ├── gears/                         # GEARS model integration
│   │   ├── gears.py                   # GEARS wrapper
│   │   ├── model.py                   # GNN model architecture
│   │   ├── pertdata.py                # Perturbation data handling
│   │   ├── inference.py               # Inference utilities
│   │   ├── data_utils.py              # Data utilities
│   │   └── utils.py                   # GEARS utilities
│   └── bmdal/                         # Batch mode AL selection (S_Div)
│       ├── algorithms.py              # BMDAL algorithm implementations
│       ├── selection.py               # Selection methods
│       ├── features.py                # Feature representations
│       ├── feature_maps.py            # Feature map utilities
│       ├── feature_data.py            # Feature data handling
│       ├── layer_features.py          # Layer feature utilities
│       └── utils.py                   # BMDAL utilities
├── llm/                               # LLM prior generation
│   ├── LLM_prior_extractor_async.py   # Gene scoring via LLM
│   └── LLM_prior_judge_async.py       # Score verification/auditing
├── run/                               # Hydra-based entry point
│   ├── pipeline/train.py              # Hydra training script
│   └── conf/                          # Hydra config hierarchy
│       ├── config.yaml
│       ├── dataset/
│       ├── model/
│       ├── strategy/
│       ├── llm/
│       └── dir/
├── data/                              # Datasets (download required)
│   └── Prior_kernel_preprocess.ipynb   # Kernel generation notebook
└── results/                           # Output directory
```

---

## Citation

If you find PertAL useful in your research, please cite:

```bibtex
@article{tao2026pertal,
  title   = {PertAL: An Efficient and Robust Active Learning Framework
             for Budget-Constrained Single-Cell Perturbation Screens},
  author  = {Tao, Siyu and Li, Yuanxian and Wu, Min and Zheng, Jie},
  journal = {},
  year    = {2026},
  url     = {https://github.com/JieZheng-ShanghaiTech/PertAL}
}
```

---

## Acknowledgements

PertAL builds on ideas and code from:

- [IterPert](https://github.com/Genentech/iterative-perturb-seq/tree/master) — iterative active learning for Perturb-seq
- [bmdal_reg](https://github.com/dholzmueller/bmdal_reg) — batch mode deep active learning

---

## Contact

**Jie Zheng** (corresponding author)
ShanghaiTech University
[zhengjie@shanghaitech.edu.cn](mailto:zhengjie@shanghaitech.edu.cn)

---
