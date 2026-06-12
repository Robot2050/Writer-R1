<h1 align="center">
    Writer-R1
</h1>

## 🎯 Who Should Pay Attention to Our Work?

* **Researchers in Generative and Long-Form Writing:** Scholars studying open-ended generation, discourse-level planning, stylistic control, logical progression, instruction satisfaction, and long-text generation in scenarios such as reports, narratives, legal drafting, educational content, and marketing copywriting.
* **Researchers in Open-Ended Evaluation and Reward Modeling:** Researchers addressing the lack of verifiable reference answers, high annotation costs, evaluative bias, and coarse-grained feedback signals in automatic evaluation for generative writing.
* **Practitioners of Reinforcement Learning for LLMs:** Engineers and scientists exploring how prompt-adaptive, interpretable, and fine-grained evaluation signals can be converted into rewards for reinforcement learning in open-ended generation tasks.
* **Developers of Compact Writing-Oriented LLMs:** Teams aiming to improve smaller-scale models through evaluation-guided training, prompt-specific skill construction, and reward-based optimization rather than relying solely on model scaling.

## 🛠️ Quick Start

The following scripts constitute the workflow for response generation, skill-guided refinement, benchmark evaluation, score aggregation, and model training:

* `generate_response.py`
  Performs *direct inference* with the target model to generate an initial response without skill-guided rewriting or iterative refinement.

* `llm_reflection.py`
  Executes a *single-model skill-guided rewriting pipeline*, where the model uses dynamically constructed evaluation skills to assess and refine its own output.

* `llm_reflection2.py`
  Implements a *dual-model skill-guided refinement setup*: one model generates the candidate response, while another model provides evaluation-oriented critique and guided rewriting based on prompt-specific skills.

* `evaluate_benchmark.py`
  Runs *benchmark evaluation* and computes scores according to the specified evaluation protocol.

* `calculate_scores.py`
  Aggregates and post-processes evaluation outputs to obtain *final consolidated scores*.

* `train_bash`
  Provides the *training entry script* for reward-based optimization of writing-oriented models, enabling reproducible execution of the Writer-R1 training pipeline.

## ✨ Core Contributions

This paper introduces **SkillForge** and **Writer-R1**, a unified evaluation-training framework for open-ended generative writing. The core contributions are fourfold:

1. **Dynamic Evaluation Skill Mining for Open-Ended Writing**
   The paper proposes **SkillForge**, a fully automatic framework that derives fine-grained, interpretable, and prompt-specific evaluation skills from user prompts. Inspired by the iterative coding and category refinement process of **Grounded Theory**, SkillForge decomposes writing prompts into actionable requirements and dynamically identifies relevant evaluation dimensions instead of relying on fixed human-designed rubrics or predefined criteria.

2. **Skill-Guided Self-Play Rewriting at Inference Time**
   At the inference stage, the mined evaluation skills serve as explicit revision targets for skill-guided self-play rewriting. Rather than depending on holistic self-judgment or direct answering alone, the model refines its output according to prompt-specific criteria covering content relevance, discourse organization, stylistic expression, logical progression, format constraints, and instruction satisfaction.

3. **Reward-Based Training of Writer-R1 Models**
   The framework further converts dynamically constructed evaluation skills into reward signals for reinforcement learning. Based on these signals, the paper trains two writing-oriented models, **Writer-R1-4B** and **Writer-R1-8B**, enabling them to better internalize prompt analysis, requirement following, discourse organization, stylistic control, and long-form writing ability.

4. **Scalable Skill Generation and Closed-Loop Optimization**
   To support scalable skill construction, the paper builds a **19K evaluation skill dataset** and trains **EvalSkillGen-4B**, a compact model that directly maps user prompts to evaluation skills. This establishes a closed loop connecting prompt-specific evaluation, inference-time refinement, and reward-based training, allowing evaluation criteria to guide both immediate rewriting and long-term model optimization.

## 📊 Results

The experimental evaluation of **Writer-R1** demonstrates consistent and robust improvements across multiple dimensions:

* **Reliable Dynamic Evaluation Skills:** The automatically constructed prompt-specific evaluation skills can closely approximate or outperform human-curated criteria. They also provide fine-grained, interpretable, and actionable feedback compared with holistic LLM-based judgments, supporting more structured evaluation and refinement of open-ended writing outputs.
* **Superior Writing Performance:** Both **Writer-R1-4B** and **Writer-R1-8B**, trained with the proposed SkillForge framework, consistently outperform strong general-purpose and writing-specialized baselines on **WritingBench** and **HelloBench**, covering both general writing and long-text generation tasks.
* **Scalable Skill Construction:** The proposed **19K evaluation skill dataset** enables the training of **EvalSkillGen-4B**, a compact model that directly generates evaluation skills from user prompts. Experiments show that skills generated by both the full SkillForge pipeline and EvalSkillGen-4B exhibit strong criteria alignment and transferability across evaluator and target models.
* **Effective Inference-Time Refinement:** Skill-guided self-play rewriting provides more stable inference-time improvement than direct answering or holistic self-judgment. Additional analyses on evaluator sources, human evaluation, and decoding behavior further confirm that the observed gains are not tied to a specific evaluator and reflect the general effectiveness of the SkillForge framework.



