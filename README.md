# Medical Healthcare Assistant - LLM Fine-Tuning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/James-Jok-Akuei/Domain-Specific-Assistant-LLMs/blob/main/notebook.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/)

> **A domain-specific medical assistant built by fine-tuning TinyLlama-1.1B with LoRA, achieving 242% improvement in BLEU score with strict medical domain enforcement.**

---

## � Project at a Glance

| Metric | Value | Description |
|--------|-------|-------------|
| 🎯 **BLEU Score** | **0.178** (+242%) | N-gram overlap with reference answers |
| 📈 **ROUGE-L** | **0.268** (+89%) | Longest common subsequence similarity |
| 🎲 **Perplexity** | **12.8** (-73%) | Model uncertainty reduction |
| ⚡ **Training Time** | **2-3 hours** | On Colab T4 GPU (free tier) |
| 💾 **Trainable Params** | **0.8%** (8.4M/1.1B) | Parameter-efficient LoRA fine-tuning |
| 📚 **Dataset Size** | **5,000 samples** | Medical flashcard Q&A pairs |
| 🛡️ **Domain Control** | **Two-layer enforcement** | Keyword filter + enhanced prompts |
| 🚀 **Deployment** | **Gradio Web UI** | Interactive, shareable interface |

### Key Achievements ✨

- ✅ **242% improvement** in BLEU score over base model
- ✅ **Robust domain restriction** - rejects 100% of non-medical questions in testing
- ✅ **Production-ready** - deployed Gradio interface with parameter controls
- ✅ **Efficient training** - works on free Colab tier with 4-bit quantization
- ✅ **Clean implementation** - modular code, comprehensive documentation

---

## �📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#key-features)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Methodology](#-training-methodology)
- [Performance Metrics](#-performance-metrics)
- [Deployment](#-deployment)
- [How to Run](#-how-to-run)
- [Example Conversations](#-example-conversations)
- [Video Demonstration](#-video-demonstration)
- [Technical Stack](#️-technical-stack)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Citation](#-citation)
- [License & Disclaimer](#-license--disclaimer)

---

## 🏥 Project Overview

This project implements a **domain-specific medical assistant** by fine-tuning TinyLlama-1.1B using LoRA (Low-Rank Adaptation) on medical question-answer pairs. The assistant can answer healthcare-related queries with improved accuracy and relevance compared to the base pre-trained model.

### ✨ Key Features at a Glance

<table>
<tr>
<td width="50%">

#### 🛡️ Domain Restriction
- **Two-layer enforcement** (keyword + prompt)
- 70+ medical keywords
- 30+ non-medical topics rejected
- 100% non-medical rejection rate

</td>
<td width="50%">

#### ⚡ Performance Gains
- **242% BLEU improvement**
- **89% ROUGE-L improvement**
- **73% perplexity reduction**
- Superior medical terminology

</td>
</tr>
<tr>
<td width="50%">

#### 💻 Technical Excellence
- LoRA fine-tuning (0.8% trainable params)
- 4-bit quantization (BitsAndBytes)
- Runs on Colab free tier
- 2-3 hour training time

</td>
<td width="50%">

#### 🚀 Production Ready
- Gradio web interface
- Adjustable parameters
- Public URL sharing
- Cross-platform compatible

</td>
</tr>
</table>

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

## � Quick Start

Get started in 3 steps:

```bash
1️⃣ Open in Colab: Click the badge above
2️⃣ Run all cells: Runtime → Run all
3️⃣ Use the interface: Gradio UI launches automatically
```

**That's it!** No installation, no setup. Start asking medical questions immediately.

### Local Installation (Optional)

```bash
git clone https://github.com/James-Jok-Akuei/Domain-Specific-Assistant-LLMs.git
cd Domain-Specific-Assistant-LLMs
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

**Requirements**: GPU with 12GB+ VRAM, Python 3.10+
### 🔄 End-to-End Workflow

```mermaid
graph LR
    A[📥 Load Dataset] --> B[🔧 Preprocess]
    B --> C[🤖 Load TinyLlama]
    C --> D[⚙️ Configure LoRA]
    D --> E[🏋️ Train 3 Epochs]
    E --> F[📊 Evaluate]
    F --> G[💾 Save Model]
    G --> H[🚀 Deploy Gradio]
    H --> I[✅ Ready to Use!]
```

**Pipeline Overview:**

1. **Data Loading** → Medical Meadow flashcards (5K samples)
2. **Preprocessing** → Tokenization + prompt formatting
3. **Model Setup** → TinyLlama + 4-bit quantization + LoRA adapters
4. **Training** → 3 epochs, 8 batch size, 2e-5 learning rate
5. **Evaluation** → BLEU, ROUGE, Perplexity metrics
6. **Deployment** → Gradio web interface with domain enforcement
7. **Testing** → Medical & non-medical question validation
---

## �📊 Dataset

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

### Option 1: Google Colab (Recommended) ⭐

**Easiest method - no setup required:**

1. Click the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/James-Jok-Akuei/Domain-Specific-Assistant-LLMs/blob/main/notebook.ipynb) badge
2. **Runtime → Run all** (or press Ctrl+F9)
3. Wait ~2-3 hours for training to complete
4. Gradio interface launches automatically with public URL
5. Start testing with medical questions!

**Benefits:**
- ✅ Free T4 GPU access
- ✅ All dependencies pre-installed
- ✅ No local setup required
- ✅ Shareable public URL for demos

### Option 2: Local Setup

**Requirements:**
- Python 3.10 or higher
- CUDA-capable GPU (12GB+ VRAM recommended)
- 20GB free disk space

**Installation:**

```bash
# Clone the repository
git clone https://github.com/James-Jok-Akuei/Domain-Specific-Assistant-LLMs.git
cd Domain-Specific-Assistant-LLMs

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebook.ipynb
```

**Note**: Training locally requires significant GPU memory. Consider using Colab for initial experimentation.

---

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

---

## 🎥 Video Demonstration

A comprehensive video demonstration is available showcasing:

### What's Covered (8 minutes)

1. **Project Overview** - Problem statement and objectives
2. **Technical Architecture** - LoRA fine-tuning with 4-bit quantization
3. **Domain Restriction** - Two-layer enforcement mechanism
4. **Performance Results** - 242% BLEU improvement, 89% ROUGE-L improvement
5. **Live Demo** - Real-time Gradio interface with test cases:
   - ✅ Medical questions (accepted with detailed responses)
   - ❌ Non-medical questions (politely rejected)
   - 🎛️ Parameter controls (temperature, max tokens)

### Key Demonstrations

| Test Type | Example Question | Expected Behavior |
|-----------|-----------------|-------------------|
| Medical - Pharmacology | "What are the side effects of beta blockers?" | Detailed response with medical terminology |
| Medical - Clinical | "What is the first-line treatment for type 2 diabetes?" | Accurate clinical information |
| Non-Medical - Politics | "Who is the president of the United States?" | Politely declined with domain explanation |
| Non-Medical - Math | "What is the square root of 144?" | Rejected as outside medical domain |

**Video Script**: Available in the repository for reference

---

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

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Gradio Public URL Not Working

**Issue**: Gradio link shows "Could not connect" or expires

**Solutions**:
- ✅ Gradio public URLs expire after 72 hours or when Colab session ends
- ✅ Restart the Colab notebook to generate a new public URL
- ✅ Keep the Colab tab open and interact every ~60 minutes to prevent timeout
- ✅ For permanent deployment, use Hugging Face Spaces

#### 2. CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size in training configuration
per_device_train_batch_size = 4  # Instead of 8
gradient_accumulation_steps = 4  # Instead of 2

# Clear GPU memory
import torch
torch.cuda.empty_cache()
```

#### 3. Widget Rendering Errors

**Issue**: `application/vnd.jupyter.widget-view+json` errors in VS Code/GitHub

**Solutions**:
- ✅ This is expected - widgets only render in Jupyter/Colab
- ✅ Text-based progress bars configured as fallback
- ✅ Outputs are functional, just visual display differs
- ✅ All functionality works correctly despite warnings

#### 4. Model Generating Garbled Output

**Issue**: Model produces corrupted text like `<<|<<<<`

**Solutions**:
```python
# Reload the fine-tuned model before using Gradio
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./medical-assistant-lora/final_model",
    device_map="auto",
    torch_dtype=torch.float16
)
```

#### 5. Slow Training on Colab

**Issue**: Training takes longer than expected

**Solutions**:
- ✅ Ensure GPU is enabled: Runtime → Change runtime type → T4 GPU
- ✅ Check GPU utilization: `!nvidia-smi`
- ✅ Reduce dataset size for faster experimentation
- ✅ Use smaller max_length (256 instead of 512)

#### 6. Installation Issues

**Issue**: Package installation failures

**Solutions**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with specific versions
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Getting Help

- 📖 Check [experiments/experiment_log.md](experiments/experiment_log.md) for detailed training logs
- 🐛 Open an issue on GitHub with error logs and environment details
- 💬 Include Colab/Jupyter version, Python version, and GPU type

---

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

---

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{domain_medical_assistant_2026,
  title={Medical Healthcare Assistant: Domain-Specific LLM Fine-Tuning with LoRA},
  author={James Jok Akuei},
  year={2026},
  month={February},
  howpublished={\url{https://github.com/James-Jok-Akuei/Domain-Specific-Assistant-LLMs}},
  note={Fine-tuned TinyLlama-1.1B for medical Q&A with domain restriction}
}
```

### Acknowledgments & References

This project builds upon:

- **TinyLlama**: [Zhang et al., 2023](https://github.com/jzhang38/TinyLlama) - Base model architecture
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning
- **Medical Meadow**: [MedAlpaca Team](https://huggingface.co/medalpaca) - Medical dataset
- **Hugging Face**: Transformers, PEFT, and Datasets libraries

---

## 📄 License & Disclaimer

### License

MIT License - See [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

**Conditions**: Include original license and copyright notice.

### ⚠️ Important Disclaimer

**Conditions**: Include original license and copyright notice.

### ⚠️ Important Disclaimer

**FOR EDUCATIONAL PURPOSES ONLY**

This medical assistant is an educational project demonstrating domain-specific LLM fine-tuning. It should **NOT** be used for:

- ❌ Diagnosing medical conditions
- ❌ Prescribing medications or treatments
- ❌ Replacing professional medical advice
- ❌ Emergency medical situations
- ❌ Making healthcare decisions

**Always consult qualified healthcare professionals** for medical concerns, diagnosis, or treatment.

This model may:
- Generate inaccurate or outdated information
- Miss critical medical details
- Not consider individual patient factors
- Lack context of latest medical research

**Use responsibly and at your own risk.**

---

## 👤 Author & Contact

**James Jok Akuei**

- GitHub: [@James-Jok-Akuei](https://github.com/James-Jok-Akuei)
- Repository: [Domain-Specific-Assistant-LLMs](https://github.com/James-Jok-Akuei/Domain-Specific-Assistant-LLMs)
- Project Type: ML Assignment - Domain-Specific LLM Fine-Tuning
- Date: February 2026

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

Special thanks to:

- **[TinyLlama Team](https://github.com/jzhang38/TinyLlama)** - For the efficient base model
- **[MedAlpaca](https://huggingface.co/medalpaca)** - For the comprehensive medical dataset
- **[Hugging Face](https://huggingface.co/)** - For transformers, PEFT, and datasets libraries
- **Google Colab** - For free GPU resources enabling this research
- **ML Community** - For open-source tools and documentation

---

## 📊 Project Stats

- **Model Size**: 1.1B parameters (8.4M trainable)
- **Training Time**: ~2-3 hours on T4 GPU
- **Dataset**: 5,000 medical Q&A pairs
- **Performance Improvement**: 242% BLEU increase
- **Lines of Code**: ~1,500 (notebook + utilities)
- **Platform**: Google Colab (free tier)

---

## 🔗 Useful Links

- 📓 [Main Notebook](notebook.ipynb) - Complete implementation
- 📖 [Experiment Log](experiments/experiment_log.md) - Detailed hyperparameter tuning
- 💬 [Sample Conversations](examples/sample_conversations.txt) - Example outputs
- 🤗 [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- 📊 [Medical Meadow Dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

---

**Last Updated**: February 22, 2026

**Status**: ✅ Complete - Ready for demonstration and evaluation

---

<div align="center">

**If you find this project helpful, please consider giving it a ⭐️!**

Made with ❤️ for advancing medical AI accessibility

</div>
