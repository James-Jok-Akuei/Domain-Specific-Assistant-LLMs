"""
Model Training Utilities for Medical Healthcare Assistant

This module provides functions for setting up and training LLMs with LoRA/PEFT.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from typing import Optional, Dict, Any
import json


def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = True
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes configuration for 4-bit quantization.
    
    Args:
        load_in_4bit: Whether to load model in 4-bit
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant: Use nested quantization
        
    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    
    return bnb_config


def load_model_and_tokenizer(
    model_name: str,
    use_quantization: bool = True,
    trust_remote_code: bool = True
):
    """
    Load model and tokenizer with optional quantization.
    
    Args:
        model_name: Name/path of the model
        use_quantization: Whether to use 4-bit quantization
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"🤖 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    if use_quantization:
        bnb_config = create_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote_code
        )
    
    print(f"✅ Model loaded!")
    print(f"   Total parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Create LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout probability
        bias: Bias type ('none', 'all', or 'lora_only')
        task_type: Task type for PEFT
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )
    
    return lora_config


def setup_peft_model(
    model,
    lora_config: Optional[LoraConfig] = None
):
    """
    Setup model for PEFT training with LoRA.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        PEFT model ready for training
    """
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Create default LoRA config if not provided
    if lora_config is None:
        lora_config = create_lora_config()
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n📊 Model Statistics After LoRA:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    return model


def create_training_arguments(
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    logging_steps: int = 50,
    save_strategy: str = "epoch",
    evaluation_strategy: str = "epoch",
    fp16: bool = True,
    optim: str = "paged_adamw_8bit",
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments for Trainer.
    
    Args:
        output_dir: Directory to save checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        save_strategy: Checkpoint save strategy
        evaluation_strategy: Evaluation strategy
        fp16: Use FP16 training
        optim: Optimizer to use
        **kwargs: Additional arguments
        
    Returns:
        TrainingArguments object
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        fp16=fp16,
        optim=optim,
        load_best_model_at_end=True,
        report_to="none",
        **kwargs
    )
    
    return training_args


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments
):
    """
    Train the model using Hugging Face Trainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        
    Returns:
        Tuple of (trainer, training_result)
    """
    print("\n🚀 Starting training...")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Evaluation samples: {len(eval_dataset):,}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {training_args.learning_rate}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    train_result = trainer.train()
    
    print("\n✅ Training completed!")
    print(f"   Training loss: {train_result.training_loss:.4f}")
    print(f"   Training time: {train_result.metrics['train_runtime']:.2f}s")
    
    return trainer, train_result


def save_model(
    model,
    tokenizer,
    output_dir: str,
    save_name: str = "final_model"
):
    """
    Save fine-tuned model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        save_name: Name of the saved model
    """
    save_path = f"{output_dir}/{save_name}"
    
    print(f"\n💾 Saving model to {save_path}...")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("✅ Model saved successfully!")


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    use_quantization: bool = True
):
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        base_model_name: Name of the base model
        adapter_path: Path to LoRA adapters
        use_quantization: Whether to use quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"🤖 Loading fine-tuned model...")
    print(f"   Base model: {base_model_name}")
    print(f"   Adapter path: {adapter_path}")
    
    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_name,
        use_quantization=use_quantization
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("✅ Fine-tuned model loaded!")
    
    return model, tokenizer


def get_gpu_memory_usage():
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated
        }
    else:
        return {"gpu_available": False}


def save_training_config(
    config: Dict[str, Any],
    output_path: str
):
    """
    Save training configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Training config saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("🧪 Testing model training utilities...")
    
    # Test configurations
    bnb_config = create_bnb_config()
    print(f"\n✅ BitsAndBytes config created")
    
    lora_config = create_lora_config()
    print(f"✅ LoRA config created: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    training_args = create_training_arguments()
    print(f"✅ Training arguments created: epochs={training_args.num_train_epochs}")
    
    # Check GPU
    gpu_info = get_gpu_memory_usage()
    print(f"\n📊 GPU Memory: {gpu_info}")
    
    print("\n✅ All utilities working correctly!")
