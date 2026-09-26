<h1 align="center">
    Writer-R1: Dynamic Rubric Mining for Reinforcement Learning of Open-Ended Writing
</h1>

## 🎯 Who Should Pay Attention to Our Work?

* **Researchers in Open-Ended Writing:** Creative and long-form writing has no verifiable reference answer, and static metrics or zero-shot LLM judges give unstable supervision. Writer-R1 converts each request into an explicit rubric with consolidated dimensions and observable scoring levels, so quality becomes checkable rather than implied.
* **RL Practitioners on Non-Verifiable Tasks:** Math and code benefit from automatic verification, while open-ended responses do not. Writer-R1 derives sequence-level rewards from mined rubrics and optimizes them with GSPO, requiring no human annotation and no trained reward model.
* **LLM Evaluation Researchers:** The mined criteria rival human-curated rubrics when guiding frontier writers, transfer across six target models, and produce consistent rankings under four LLM evaluators and expert assessment, offering a scalable alternative to fixed benchmarks.
* **Developers of Efficient LLMs:** At the 4B and 8B scales, Writer-R1 surpasses general-purpose models above 100B parameters and a 32B writing specialist on WritingBench, and a compact RubricGen-4B supplies task-specific criteria at inference with only the user request as input.

## ✨ Core Contributions

1. **Inductive Rubric Generation (IRG).**
   IRG adapts the open, axial, and selective coding stages of Grounded Theory into three instructed LLM operations that build a rubric `R(q) = {(d_k, a_k, B_k)}` from each prompt. Discovery proposes 15 to 20 candidate dimensions with rationales, at least half specific to the request; consolidation merges them into 5 to 8 dimensions with explicit assessment scopes, so no property is rewarded twice and no decisive requirement is pruned; scoring specification writes 5 to 8 consecutive two-point intervals per dimension (`I_{k,m} = {2m-1, 2m}`), whose adjacent levels differ in observable evidence. Separating discovery from consolidation lets the two decisions use different granularities, which single-pass generation cannot express.

2. **One Specification for Generation and Reward.**
   The writer input is the direct concatenation `x_q = q ⊕ serialize(R(q))`, so the model reads the exact criteria that later score its output. The same rubric selects SFT demonstrations and defines the RL reward `r(q,y) = Σ s_k(y,q) / Σ s_k^max`, a maximum-weighted average of normalized dimension scores. Each rubric is built once per prompt and reused across sampled responses and policy updates, which amortizes construction cost and keeps the assessment basis fixed while the policy changes.

3. **A 19K Prompt-Rubric Corpus with SFT + GSPO Training.**
   The 6K LongWrite prompts are expanded with Qwen3-32B under two reconstruction strategies, length-aware rewriting and rewriting with length recovery, followed by manual review and decontamination against WritingBench and HelloBench at a 0.10 token-trigram overlap threshold. SFT trains on 15K best-of-five demonstrations selected by the rubric; RL continues on the remaining prompts with groups of 8 responses, sequence-level importance ratios, clipping at `ε = 3e-4` and `ε_high = 4e-4`, KL coefficient 0.01, and dynamic resampling of zero-variance groups.

4. **RubricGen-4B and Cross-Model Rubric Transfer.**
   A compact generator trained on all 19K prompt-rubric pairs predicts the rubric for a new request, so inference requires only the writing request from the user. Mined criteria also guide writers that never saw them in training: applied as revision feedback to DeepSeek-V3.2-Exp, RubricGen-4B criteria reach 0.864 on WritingBench, above human-curated rubrics at 0.838 and Kimi-K2 criteria at 0.849.

## 📊 Results

We evaluate on WritingBench (1,000 requests spanning six domains and 100 subdomains, scored by its dedicated critic) and HelloBench (five long-form tasks, scored by DeepSeek-V3.2-Exp), with Qwen3-4B and Qwen3-8B as backbones.

* **WritingBench overall.** Writer-R1-4B and Writer-R1-8B score **0.863** and **0.871**, the two highest results in the comparison, improving over their backbones by **0.182** and **0.158**. Both surpass the strongest general-purpose baseline Kimi-K2 (0.859) and the strongest writing-specialized baseline LongWriter-Zero-32B (0.802), with gains in all six domains and all style, format, and length metrics.
* **HelloBench.** Writer-R1-4B improves from 0.641 to 0.700 and Writer-R1-8B from 0.682 to **0.733**, leading LongWriter-Zero-32B (0.713) with a mean output of 1,835 words versus 2,712. The quality gain does not come from writing longer.
* **Construction ablation.** Under matched budgets, IRG exceeds one-shot direct rubric generation by **0.097** after both SFT and SFT+RL. Direct generation without structure scores below training with no rubrics at all (0.719 vs 0.725), while RL adds +0.047 on top of either construction method.
* **Robustness across evaluators.** Writer-R1-8B ranks first under every evaluation source: DeepSeek-V3.2-Exp, Qwen3-235B-A22B, GLM-5.1, Kimi-K2, and averaged scores from four human experts. Human assessment rises from 0.546 to **0.708** at 4B and from 0.592 to **0.741** at 8B, ahead of LongWriter-Zero-32B at 0.684.

## 🛠️ Quick Start

### Installation

The data and evaluation scripts rely on vLLM, Transformers, an OpenAI-compatible client, jsonlines, and pandas. Training uses LLaMA-Factory for SFT and MS-Swift with DeepSpeed ZeRO-3 for RL.

```bash
pip install vllm transformers openai jsonlines pandas tqdm
pip install llamafactory ms-swift deepspeed
```

API endpoints and keys for the constructor, writer, and evaluator models are configured in `src/llm/` and `evaluator/`. `generate_rubric_direct.py` instead loads a local checkpoint through vLLM by setting `model_name_or_path` at the top of the file.

### Corpus Expansion and Rubric Mining

```bash
# Expand prompts and sample responses through an OpenAI-compatible endpoint
python expand_corpus.py --query_file prompts.jsonl --output_folder data/output

# Three-stage IRG construction (input and output paths set in __main__)
python generate_rubric.py

# Single-call rubric generation, the DRG baseline in the ablation
python generate_rubric_direct.py
```

`Example.json` shows one request with its mined rubric, including dimension names, scope descriptions, and per-interval level text. The code uses the term checklist for what the paper calls a rubric; they denote the same object.

### Inference and Rubric-Guided Revision

```bash
# Direct answering on the benchmark
python generate_response.py --query_file benchmark_query/benchmark_all.jsonl --output_file responses.jsonl

# Iterative revision under rubric feedback (transfer study)
python revise_with_rubric.py      --query_file benchmark_query/benchmark_all.jsonl --output_folder runs/single_model
python revise_with_rubric_dual.py --query_file benchmark_query/benchmark_all.jsonl --output_folder runs/writer_reviewer
```

### Evaluation

```bash
# Score responses against their rubrics
python evaluate_benchmark.py --evaluator critic \
    --query_criteria_file benchmark_query/benchmark_all.jsonl \
    --input_file responses.jsonl \
    --output_file eval/scores.jsonl

# Aggregate domain and requirement scores into an Excel report
python calculate_scores.py --score_dir eval/ --output_excel results.xlsx
```

### Training

```bash
# Stage 1: SFT with LLaMA-Factory
bash train_bash1/full_sft_qwen3.sh

# Stage 2: GSPO with MS-Swift; start the rollout server first
bash train_bash2/vllm_qwen3.sh
bash train_bash2/gspo_qwen3.sh
```

Key settings, matching the configuration reported in the paper:

| Setting | Value |
| --- | --- |
| SFT / RL data | 15K / about 4K prompt-rubric pairs |
| Demonstration selection | best of 5 Qwen3-32B candidates under the rubric |
| Constructor and reward model | Qwen3-32B |
| Group size / dynamic resampling | 8 responses / up to 3 attempts |
| GSPO clipping | ε = 3e-4, ε_high = 4e-4 |
| KL coefficient | 0.01 |
| Rollout decoding | temperature 1.0, top-p 0.9, top-k 50, max 4,096 tokens |
| Precision and parallelism | bf16, DeepSpeed ZeRO-3, NVIDIA A800 80GB |

`train_bash2/reward.py` implements the rubric-based reward as an MS-Swift plugin, scoring each sampled response against the stored rubric and aggregating per-dimension scores by their maxima.

## 📁 Repository Structure

```
├── generate_rubric.py             # IRG: discovery, consolidation, scoring specification
├── generate_rubric_direct.py      # Single-call rubric generation (DRG baseline, local vLLM)
├── expand_corpus.py               # Corpus expansion and response sampling (multi-threaded)
├── expand_corpus_cycle.py         # Iterative variant of the expansion pipeline
├── generate_response.py           # Direct benchmark inference
├── revise_with_rubric.py          # Rubric-guided revision with a single model
├── revise_with_rubric_dual.py     # Rubric-guided revision with a separate reviewer
├── evaluate_benchmark.py          # Rubric-based scoring of benchmark responses
├── calculate_scores.py            # Domain and requirement aggregation, Excel export
├── prompt.py                      # Evaluation prompts and scoring rules
├── Example.json                   # One request with its mined rubric
├── benchmark_query/               # WritingBench prompts and style/format/length subsets
├── evaluator/                     # Critic and LLM scoring clients
├── src/llm/                       # OpenAI-compatible API wrappers
├── train_bash1/                   # SFT launcher and LLaMA-Factory configuration
└── train_bash2/                   # GSPO launcher, rubric reward plugin, rollout server
```

## 📄 License

Writer-R1 is released under the [Apache 2.0 License](./LICENSE).
