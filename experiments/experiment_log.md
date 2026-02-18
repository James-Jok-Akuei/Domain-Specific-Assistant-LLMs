# Hyperparameter Tuning Experiments

## Overview

This document tracks all hyperparameter tuning experiments conducted to optimize the medical healthcare assistant model.

---

## Experimental Setup

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (NF4)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
- **Training Data**: 5,000 medical Q&A pairs
- **Validation Data**: 500 samples
- **Hardware**: Google Colab T4 GPU (15GB)

---

## Experiments Summary Table

| Experiment | Learning Rate | Batch Size | Gradient Accum | Effective Batch | Epochs | Train Loss | Eval Loss | Training Time | GPU Memory | BLEU  | ROUGE-L | Notes |
|------------|--------------|------------|----------------|-----------------|--------|------------|-----------|---------------|------------|-------|---------|-------|
| Baseline   | -            | -          | -              | -               | 0      | -          | -         | -             | -          | 0.052 | 0.142   | Pre-trained model without fine-tuning |
| Exp-1      | 1e-4         | 4          | 2              | 8               | 1      | 1.456      | 1.234     | ~45 min       | ~11 GB     | 0.087 | 0.189   | High LR causes instability |
| Exp-2      | 5e-5         | 4          | 2              | 8               | 2      | 1.289      | 1.156     | ~1.5 hr       | ~11 GB     | 0.124 | 0.223   | Better convergence |
| Exp-3      | 3e-5         | 8          | 2              | 16              | 2      | 1.178      | 1.089     | ~1.7 hr       | ~12 GB     | 0.156 | 0.251   | Larger batch helps |
| **Exp-4**  | **2e-5**     | **8**      | **2**          | **16**          | **3**  | **1.067**  | **1.032** | **~2.5 hr**   | **~12 GB** | **0.178** | **0.268** | **Best performance** ✅ |

---

## Detailed Experiment Reports

### Baseline (No Fine-tuning)

**Purpose**: Establish baseline performance of pre-trained model

**Configuration**:
- Model: Pre-trained TinyLlama-1.1B-Chat-v1.0
- No training performed

**Results**:
- BLEU Score: 0.052
- ROUGE-L: 0.142
- Perplexity: 47.3

**Observations**:
- Generic responses lacking medical specificity
- Frequently provides overly cautious responses
- Limited medical terminology usage
- Inconsistent answer quality

---

### Experiment 1: High Learning Rate

**Purpose**: Test aggressive learning rate for fast convergence

**Configuration**:
```python
learning_rate = 1e-4
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
num_train_epochs = 1
warmup_steps = 100
optimizer = "paged_adamw_8bit"
```

**Results**:
- Training Loss: 1.456
- Validation Loss: 1.234
- BLEU Score: 0.087
- ROUGE-L: 0.189
- Training Time: 45 minutes
- GPU Memory: ~11 GB peak

**Observations**:
- Fast training but unstable convergence
- Loss oscillates during training
- Modest improvement over baseline
- Risk of overshooting optimal parameters

**Conclusion**: Learning rate too high; need more conservative approach

---

### Experiment 2: Moderate Learning Rate, Extended Training

**Purpose**: Test moderate learning rate with longer training

**Configuration**:
```python
learning_rate = 5e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
num_train_epochs = 2
warmup_steps = 100
optimizer = "paged_adamw_8bit"
```

**Results**:
- Training Loss: 1.289
- Validation Loss: 1.156
- BLEU Score: 0.124
- ROUGE-L: 0.223
- Training Time: 1.5 hours
- GPU Memory: ~11 GB peak

**Observations**:
- More stable convergence
- Better performance than Exp-1
- Validation loss still decreasing at end of training
- Model responses more medically accurate

**Conclusion**: Good direction; could benefit from lower LR and longer training

---

### Experiment 3: Lower Learning Rate, Larger Batch

**Purpose**: Test impact of larger effective batch size

**Configuration**:
```python
learning_rate = 3e-5
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 2
warmup_steps = 100
optimizer = "paged_adamw_8bit"
```

**Results**:
- Training Loss: 1.178
- Validation Loss: 1.089
- BLEU Score: 0.156
- ROUGE-L: 0.251
- Training Time: 1.7 hours
- GPU Memory: ~12 GB peak

**Observations**:
- Smooth training curves
- Better generalization (lower eval loss)
- Larger batch size stabilizes gradients
- Significant improvement in BLEU and ROUGE

**Conclusion**: Larger batch size beneficial; try even lower LR with more epochs

---

### Experiment 4: Optimal Configuration (BEST) ✅

**Purpose**: Find optimal balance of LR, batch size, and training duration

**Configuration**:
```python
learning_rate = 2e-5
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 3
warmup_steps = 100
weight_decay = 0.01
max_grad_norm = 1.0
lr_scheduler_type = "cosine"
optimizer = "paged_adamw_8bit"
```

**Results**:
- Training Loss: 1.067
- Validation Loss: 1.032
- BLEU Score: 0.178
- ROUGE-L: 0.268
- Perplexity: 12.8
- Training Time: 2.5 hours
- GPU Memory: ~12 GB peak

**Detailed Metrics**:
- BLEU-1: 0.245
- BLEU-2: 0.198
- BLEU-3: 0.162
- BLEU-4: 0.134
- ROUGE-1: 0.312
- ROUGE-2: 0.198
- ROUGE-L: 0.268

**Improvement Over Baseline**:
- BLEU: +242% improvement
- ROUGE-L: +89% improvement
- Perplexity: -73% improvement

**Training Curve Analysis**:
- Smooth convergence throughout training
- No signs of overfitting
- Validation loss continues to decrease
- Learning rate cosine schedule works well

**Observations**:
- Best overall performance across all metrics
- Responses are medically accurate and detailed
- Good balance between fluency and correctness
- Model handles various medical topics well
- Appropriate use of medical terminology

**Sample Outputs**:

*Question*: "What is the function of the mitochondria?"

*Response*: "The mitochondria are membrane-bound organelles known as the 'powerhouses' of the cell. Their primary function is to generate adenosine triphosphate (ATP) through cellular respiration and oxidative phosphorylation. They also play roles in calcium homeostasis, apoptosis regulation, and cellular signaling."

**Conclusion**: This configuration provides the best balance of performance, training time, and resource usage. Selected as final model.

---

## Key Findings

### 1. Learning Rate Sensitivity
- **Too high (1e-4)**: Unstable training, suboptimal convergence
- **Moderate (5e-5)**: Good but can be improved
- **Optimal (2e-5)**: Best stability and performance
- **Pattern**: Medical domain benefits from conservative learning rates

### 2. Batch Size Impact
- Larger effective batch size (16 vs 8) improved stability
- Better gradient estimates lead to smoother convergence
- Memory usage acceptable with gradient accumulation

### 3. Training Duration
- Single epoch: Insufficient for complex medical knowledge
- 2 epochs: Good improvement but not optimal
- 3 epochs: Best results without overfitting
- Pattern: Medical domain requires sufficient exposure

### 4. Optimizer Choice
- `paged_adamw_8bit` works well with memory constraints
- Weight decay (0.01) prevents overfitting
- Gradient clipping (1.0) stabilizes training

### 5. Resource Utilization
- 4-bit quantization essential for Colab free tier
- Peak memory: ~12 GB (fits comfortably in T4)
- Training time: ~2.5 hours (reasonable for project scope)

---

## Recommendations for Future Work

1. **Extended Training**: Try 4-5 epochs with early stopping
2. **Learning Rate**: Experiment with 1e-5 for even more stable training
3. **LoRA Rank**: Test r=32 for potentially better capacity
4. **Data Augmentation**: Paraphrase questions for more diverse training
5. **Warmup Ratio**: Increase warmup for more gradual learning
6. **Multi-task Learning**: Include medical entity recognition
7. **Curriculum Learning**: Start with simpler questions, progress to complex

---

## Experimental Methodology

### Evaluation Protocol
1. Generate predictions on held-out test set (100 samples)
2. Calculate BLEU, ROUGE, and Perplexity
3. Qualitative analysis of responses
4. Compare against baseline pre-trained model

### Reproducibility
- Random seed: 42 (fixed across all experiments)
- Same data splits for all experiments
- Consistent evaluation methodology
- Documented hyperparameters

### Hardware Constraints
- Google Colab T4 GPU (15GB VRAM)
- 12GB RAM for Python runtime
- Limited to ~3 hour continuous runtime
- 4-bit quantization required

---

## Conclusion

Through systematic experimentation, we identified that a learning rate of 2e-5, batch size of 8 (effective 16 with gradient accumulation), and 3 epochs of training provides the best results for fine-tuning TinyLlama on medical Q&A data. This configuration achieves:

- **242% improvement** in BLEU score over baseline
- **89% improvement** in ROUGE-L score
- **73% reduction** in perplexity

The fine-tuned model demonstrates strong medical knowledge and generates accurate, detailed responses appropriate for the healthcare domain.

---

**Last Updated**: February 2026  
**Experiment Conducted By**: ML Assignment Team  
**Total Training Time**: ~6.5 hours across all experiments
