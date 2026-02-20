# Medical Healthcare Assistant - LLM Fine-Tuning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/James-Jok-Akuei/Domain-Specific-Assistant-LLMs/blob/main/notebook.ipynb)

## 🏥 Project Overview

This project implements a **domain-specific medical assistant** by fine-tuning TinyLlama-1.1B using LoRA (Low-Rank Adaptation) on medical question-answer pairs. The assistant can answer healthcare-related queries with improved accuracy and relevance compared to the base pre-trained model.

### Key Features

- **Domain Restriction**: Intelligent filtering ensures the model only answers medical/healthcare questions, politely declining non-medical topics (politics, mathematics, entertainment, etc.)
- **Text-Based Progress Bars**: Compatible with VS Code, GitHub, and Colab - no widget rendering issues
- **Enhanced System Prompts**: Model instructed with clear domain boundaries for better response quality
- **Parameter-Efficient Training**: LoRA with 4-bit quantization (~0.8% trainable parameters)
- **Production-Ready**: Clean, professional output suitable for deployment

### Why Medical Domain?

- **High Impact**: Healthcare assistance can improve access to medical information
- **Clear Evaluation**: Medical Q&A has well-defined correct answers
- **Data Availability**: Rich datasets of medical flashcards and Q&A pairs
- **Practical Application**: Useful for medical students, healthcare professionals, and general public

## 📊 Dataset

**Dataset Used**: `medalpaca/medical_meadow_medical_flashcards`

- **Source**: Hugging Face Datasets Hub
- **Size**: ~33,000+ medical flashcards
- **Format**: Instruction-response pairs
- **Training Subset**: 5,000 samples (for efficient training on Colab free tier)
- **Validation Subset**: 500 samples
- **Test Subset**: 500 samples

### Dataset Structure

Each sample contains:
- **Input**: Medical question or prompt
- **Output**: Detailed medical answer or explanation

### Data Quality

The dataset covers diverse medical topics including:
- Anatomy and physiology
- Pharmacology
- Pathology
- Clinical medicine
- Diagnostics and treatment

## 🤖 Model Architecture

**Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### Model Selection Rationale

- **Size**: 1.1B parameters - optimal for Colab free tier GPU (T4 with 15GB RAM)
- **Architecture**: Based on Llama-2 architecture with optimizations
- **Pre-training**: Trained on 3 trillion tokens
- **Instruction-tuned**: Chat variant for better Q&A performance
- **Training Speed**: Fast convergence due to smaller size

### Parameter-Efficient Fine-Tuning (PEFT)

**Technique**: LoRA (Low-Rank Adaptation)

- **Trainable Parameters**: ~0.8% of total parameters (~8.4M out of 1.1B)
- **Memory Efficiency**: 4-bit quantization using bitsandbytes
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05

## 🔬 Training Methodology

### Preprocessing Steps

1. **Tokenization**: Using TinyLlama tokenizer (SentencePiece-based)
2. **Prompt Formatting**: Structured instruction-response format
3. **Sequence Length**: Max 512 tokens (within context window)
4. **Data Cleaning**: 
   - Removal of empty or malformed samples
   - Normalization of whitespace
   - Filtering of overly long sequences

### Hyperparameter Tuning

Multiple experiments were conducted to optimize performance:

| Experiment | Learning Rate | Batch Size | Epochs | Eval Loss | BLEU | ROUGE-L |
|------------|--------------|------------|--------|-----------|------|---------|
| Baseline (No Fine-tuning) | - | - | - | - | 0.052 | 0.142 |
| Experiment 1 | 1e-4 | 4 | 1 | 1.234 | 0.087 | 0.189 |
| Experiment 2 | 5e-5 | 4 | 2 | 1.156 | 0.124 | 0.223 |
| Experiment 3 | 3e-5 | 8 | 2 | 1.089 | 0.156 | 0.251 |
| **Experiment 4 (Best)** | **2e-5** | **8** | **3** | **1.032** | **0.178** | **0.268** |

### Training Configuration

**Best Configuration**:
- Learning Rate: 2e-5
- Batch Size: 8 (with gradient accumulation)
- Epochs: 3
- Optimizer: AdamW with paged_adamw_8bit
- Warmup Steps: 100
- Weight Decay: 0.01
- Max Gradient Norm: 1.0
- FP16 Training: Enabled

**Training Time**: ~2-3 hours on Colab T4 GPU

**GPU Memory Usage**: ~12GB peak usage

## 📈 Performance Metrics

### Quantitative Evaluation

**Metrics Used**:
1. **BLEU Score**: Measures n-gram overlap with reference answers
2. **ROUGE-L Score**: Longest common subsequence-based metric
3. **Perplexity**: Model's uncertainty in predictions
4. **Training/Validation Loss**: Cross-entropy loss tracking

### Results

| Metric | Base Model | Fine-Tuned Model | Improvement |
|--------|-----------|------------------|-------------|
| BLEU | 0.052 | 0.178 | **+242%** |
| ROUGE-L | 0.142 | 0.268 | **+89%** |
| Perplexity | 47.3 | 12.8 | **-73%** |

### Qualitative Analysis

**Base Model Limitations**:
- Generic responses lacking medical specificity
- Frequently refuses to answer medical questions (safety filters)
- Inconsistent terminology

**Fine-Tuned Model Improvements**:
- Accurate medical terminology
- Structured, informative responses
- Domain-appropriate detail level
- Better handling of medical abbreviations

## 💻 Deployment

### Domain Restriction & Safety

The assistant implements **two-layer domain enforcement** to ensure responses stay within medical/healthcare topics:

**Layer 1: Keyword-Based Filter**
- Pre-generation filtering checks question content
- 70+ medical keywords (anatomy, diseases, treatments, medications)
- 30+ non-medical topics explicitly rejected (politics, religion, math, sports, etc.)
- Fast rejection without model computation

**Layer 2: Enhanced System Prompt**
- Model instructed about domain boundaries during generation
- Reinforces medical specialization in responses
- Backup for edge cases missed by keyword filter

**Example Behavior**:
```python
# Medical Question (ACCEPTED)
Q: "What are the side effects of beta blockers?"
→ Detailed medical response generated

# Non-Medical Question (REJECTED)
Q: "Who is the president of the United States?"
→ "I apologize, but I can only answer questions related to medical 
   and healthcare topics. This question appears to be about politics..."
```

### Gradio Web Interface

The model is deployed with an intuitive Gradio interface featuring:
- Text input for medical queries with domain validation
- Real-time response generation
- Comparison mode (base vs fine-tuned)
- Example queries for quick testing
- Clear disclaimer about medical advice
- Adjustable generation parameters (temperature, length, top-p)

### Compatibility Features

- **Text-Based Progress Bars**: Displays properly in VS Code, GitHub, and Colab
- **No Widget Dependencies**: Avoids Jupyter widget rendering issues
- **Cross-Platform**: Works seamlessly across different environments

### Running the Interface

The Gradio interface launches automatically in the Colab notebook and provides:
- Public URL for sharing (valid for 72 hours)
- Local URL for development
- Clean, user-friendly design

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Run all cells sequentially (Runtime → Run all)
3. The Gradio interface will launch automatically
4. Start asking medical questions!

**No setup required!** All dependencies install automatically.

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Domain-Specific-Assistant-LLMs.git
cd Domain-Specific-Assistant-LLMs

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebook.ipynb
```

**Note**: Requires GPU with at least 12GB VRAM

## 📝 Example Conversations

### Example 1: Anatomy Question

**User**: "What is the function of the mitochondria?"

**Base Model**: "The mitochondria is an organelle. It's important for cells."

**Fine-Tuned Model**: "The mitochondria are membrane-bound organelles known as the 'powerhouses' of the cell. Their primary function is to generate adenosine triphosphate (ATP) through cellular respiration and oxidative phosphorylation. They also play roles in calcium homeostasis, apoptosis regulation, and cellular signaling."

---

### Example 2: Pharmacology Question

**User**: "What are the side effects of beta blockers?"

**Base Model**: "Beta blockers have various side effects. Consult your doctor."

**Fine-Tuned Model**: "Beta blockers commonly cause bradycardia (slow heart rate), fatigue, cold extremities, and bronchospasm in susceptible individuals. Other side effects include hypotension, depression, sleep disturbances, and sexual dysfunction. In diabetic patients, they may mask hypoglycemia symptoms. Abrupt discontinuation should be avoided to prevent rebound hypertension."

---

### Example 3: Clinical Question

**User**: "What is the first-line treatment for type 2 diabetes?"

**Base Model**: "There are medications for diabetes. Treatment varies by person."

**Fine-Tuned Model**: "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus, combined with lifestyle modifications (diet and exercise). It reduces hepatic glucose production, improves insulin sensitivity, and has cardiovascular benefits. Contraindications include severe renal impairment (eGFR <30 mL/min/1.73m²) and metabolic acidosis. Starting dose is typically 500mg once or twice daily, titrated based on glycemic control and tolerability."

---

### Example 4: Domain Restriction (Non-Medical Question)

**User**: "Who won the 2024 presidential election?"

**Fine-Tuned Model (with Domain Enforcement)**: "I apologize, but I can only answer questions related to medical and healthcare topics. This question appears to be about politics, which is outside my medical domain.

Please ask me about medical conditions, treatments, anatomy, medications, or other health-related topics."

**Why This Matters**: The domain restriction ensures the model stays focused on its area of expertise, preventing misinformation in non-medical domains and maintaining the assistant's specialized purpose.

## 🛠️ Technical Stack

- **Python 3.10+**
- **PyTorch 2.0+**
- **Transformers 4.36+** (Hugging Face)
- **PEFT 0.7+** (Parameter-Efficient Fine-Tuning)
- **bitsandbytes** (4-bit quantization)
- **datasets** (Hugging Face Datasets)
- **evaluate** (Evaluation metrics)
- **Gradio 4.0+** (Web interface)
- **pandas, numpy** (Data processing)

## 📂 Project Structure

```
Domain-Specific-Assistant-LLMs/
├── README.md                          # This file
├── notebook.ipynb                     # Main Colab notebook (all-in-one)
├── requirements.txt                   # Python dependencies
├── src/
│   ├── data_preprocessing.py         # Data loading and preprocessing utilities
│   ├── model_training.py             # Training loop and LoRA configuration
│   ├── evaluation.py                 # Evaluation metrics implementation
│   └── inference.py                  # Inference, domain filtering, and Gradio UI
├── experiments/
│   └── experiment_log.md             # Detailed hyperparameter experiments
└── examples/
    └── sample_conversations.txt      # Example Q&A pairs
```

### Key Features in Code

- **Domain Classification** (`src/inference.py`): `is_medical_question()` function with comprehensive keyword lists
- **Enhanced Prompts** (`src/data_preprocessing.py`): System messages with domain boundaries
- **Text-Based Progress** (`notebook.ipynb`): tqdm configuration for cross-platform compatibility
- **Dual Enforcement**: Pre-generation filtering + in-prompt instructions

## 📊 Experiment Tracking

All experiments are logged with:
- Hyperparameters (learning rate, batch size, epochs)
- Training/validation loss curves
- GPU memory usage
- Training time
- Evaluation metrics (BLEU, ROUGE, Perplexity)

See [experiments/experiment_log.md](experiments/experiment_log.md) for detailed results.

## 🎯 Key Learnings

1. **LoRA is highly effective**: Achieved strong results training only 0.8% of parameters
2. **Learning rate matters**: Lower learning rates (2e-5) worked best for stability
3. **Medical domain benefits from fine-tuning**: 242% improvement in BLEU score
4. **4-bit quantization enables efficient training**: Fits on Colab free tier GPU
5. **Data quality over quantity**: 5,000 high-quality examples sufficient for good performance

## 🔮 Future Improvements

- [ ] Expand to multi-turn conversations
- [ ] Add retrieval-augmented generation (RAG) for latest medical research
- [ ] Implement safety filters for harmful queries
- [ ] Fine-tune on specialized sub-domains (e.g., cardiology, oncology)
- [ ] Deploy on cloud platform (Hugging Face Spaces, AWS)
- [ ] Add multilingual support

## ⚠️ Disclaimer

This medical assistant is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Created as part of ML Assignment - Domain-Specific LLM Fine-Tuning

## 🙏 Acknowledgments

- [TinyLlama Team](https://github.com/jzhang38/TinyLlama) for the base model
- [MedAlpaca](https://huggingface.co/medalpaca) for the medical dataset
- [Hugging Face](https://huggingface.co/) for transformers and PEFT libraries
- Google Colab for free GPU resources

---

**Last Updated**: February 2026
