<h1 align="center">
    Writer-R1: Enhancing Generative Writing in LLMs via Memory-augmented Replay Policy Optimization
</h1>

## 🎯 Who Should Pay Attention to Our Work?

* **Researchers in Long-Form Text Generation:** Scholars addressing the notorious challenges of long-horizon planning, such as factual inconsistencies, loose inter-paragraph transitions, and narrative collapse in ultra-long outputs (e.g., novels, reports, legal documents).
* **Practitioners of RLHF:** Engineers and scientists exploring Reinforcement Learning from Human/AI Feedback who struggle with the lack of verifiable ground truth and high annotation costs in divergent, open-ended tasks like creative writing.
* **LLM Alignment Specialists:** Those interested in moving beyond standard Supervised Fine-Tuning to more robust alignment paradigms that combine parameter internalization with inference-time reflection.
* **Developers of Efficient LLMs:** Teams aiming to maximize the capabilities of smaller models (e.g., 7B parameter scale or lower) to rival or surpass typically larger foundation models (100B+) through algorithmic innovation rather than sheer scale.


## 🛠️ Quick Start

The following scripts constitute the workflow for generation, self-reflection, evaluation, scoring aggregation, and lightweight model training:

* `generate_response.py`
  Performs *direct inference* with the target model to produce an initial answer without self-critique or iterative refinement.

* `llm_reflection.py`
  Executes a *self-reflection pipeline* in which a single model both generates and then refines its own output through structured reasoning and revision.

* `llm_reflection2.py`
  Implements a *dual-model reflection setup*: one model is designated to generate the candidate response, while a second (potentially stronger or differently specialized) model conducts reflection, critique, and guided improvement.

* `evaluate_benchmark.py`
  Runs *benchmark evaluation* and computes corresponding scores according to the specified evaluation protocol.

* `calculate_scores.py`
  Aggregates and post-processes evaluation outputs to compute *final consolidated scores*.

* `train_bash`
  Provides the *training entry script* for aligning smaller models, enabling reproducible execution of the training pipeline.


## ✨ Core Contributions

This paper introduces **Writer-R1**, a unified framework that bridges the gap between training and inference to solve the lack of verifiable reference answers in creative writing. The core contributions are fourfold:

1. **Dynamic Evaluation via Grounded Theory & Multi-Agents**
   The paper proposes a novel methodology that instantiates the three-stage coding process of **Grounded Theory** into a multi-agent collaborative workflow. Instead of relying on static metrics or expensive human review, this system dynamically mines user prompts to construct interpretable, reusable, and fine-grained evaluation checklists. This effectively transforms abstract writing quality into executable, hierarchical scoring rules without requiring human annotation.

2. **Memory-Augmented Replay Policy Optimization (MRPO) for Inference**
   At the inference stage, the paper introduces a self-reflection mechanism called MRPO. Unlike standard generation, this approach guides the model to perform text-space gradient updates. By utilizing the dynamic evaluation checklist as a constraint, the model engages in structured self-assessment and iterative rewriting. Crucially, a memory buffer is introduced to store and retrieve transferable reflection and rewriting experiences across different samples.

3. **End-to-End Alignment via MRPO Training**
   The framework extends MRPO to the training phase to achieve permanent parameter alignment. By converting the dynamically generated evaluation checklists into sequence-level reward signals, the authors construct a 19K training set. They utilize a hybrid paradigm of **SFT + RL**, adopting the GSPO objective. This allows the model to internalize long-horizon planning and self-diagnostic capabilities, mitigating the variance and instability often associated with long-sequence training.

4. **Closing the Loop between "Learning" and "Thought"**
   Conceptually, the work operationalizes the philosophical distinction between "Learning" (parameter updates) and "Thought" (inference-time reflection). It establishes a closed loop where dynamic criteria inform both immediate self-correction (inference) and long-term policy optimization (training), solving the limitations of relying solely on one or the other.

## 📊 Results

The experimental evaluation of **Writer-R1** demonstrates robust and state-of-the-art performance across several dimensions:

* **High-Quality Automated Evaluation:** The automatically constructed dynamic evaluation criteria achieve performance gains comparable to human annotations. Furthermore, these criteria provide more stable and reliable feedback signals compared to using off-the-shelf LLMs as zero-shot judges, significantly reducing evaluative bias.
* **Superior Model Performance:** The **Writer-R1-4B** model, trained using this framework, consistently outperforms baseline models across multiple creative writing tasks.
* **Parameter Efficiency:** Despite its relatively small size, Writer-R1-4B surpasses the writing capabilities of several open-source models with over **100B parameters**, validating the efficacy of the proposed alignment and optimization strategy.
* **Effective Self-Correction:** The MRPO mechanism proves that models can achieve controlled iterative improvement without additional training during the inference phase, while the training phase successfully internalizes these improvements for end-to-end optimization.


