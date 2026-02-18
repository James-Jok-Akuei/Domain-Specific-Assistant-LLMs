"""
Evaluation Utilities for Medical Healthcare Assistant

This module provides functions for evaluating LLM performance using various metrics.
"""

import torch
import numpy as np
import evaluate
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional
import pandas as pd
import json


def load_evaluation_metrics():
    """
    Load evaluation metrics (BLEU, ROUGE).
    
    Returns:
        Dictionary of metric objects
    """
    print("📊 Loading evaluation metrics...")
    
    metrics = {
        'bleu': evaluate.load("bleu"),
        'rouge': evaluate.load("rouge")
    }
    
    # Download NLTK data for metrics
    import nltk
    nltk.download('punkt', quiet=True)
    
    print("✅ Metrics loaded!")
    return metrics


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        question: Input question
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated response string
    """
    from src.data_preprocessing import create_prompt_template
    
    # Create prompt
    prompt = create_prompt_template(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response


def batch_generate_predictions(
    model,
    tokenizer,
    questions: List[str],
    max_samples: Optional[int] = None,
    max_new_tokens: int = 150,
    batch_size: int = 1,
    show_progress: bool = True
) -> List[str]:
    """
    Generate predictions for a batch of questions.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        questions: List of questions
        max_samples: Maximum number of samples to process
        max_new_tokens: Maximum tokens per response
        batch_size: Batch size for generation
        show_progress: Whether to show progress bar
        
    Returns:
        List of generated responses
    """
    if max_samples is not None:
        questions = questions[:max_samples]
    
    predictions = []
    iterator = tqdm(questions, desc="Generating predictions") if show_progress else questions
    
    for question in iterator:
        response = generate_response(
            model, tokenizer, question,
            max_new_tokens=max_new_tokens
        )
        predictions.append(response)
    
    return predictions


def calculate_bleu_score(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate BLEU score.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with BLEU scores
    """
    bleu_metric = evaluate.load("bleu")
    
    # Format references for BLEU (needs list of lists)
    references_formatted = [[ref] for ref in references]
    
    results = bleu_metric.compute(
        predictions=predictions,
        references=references_formatted
    )
    
    return {
        'bleu': results['bleu'],
        'bleu_1': results['precisions'][0],
        'bleu_2': results['precisions'][1],
        'bleu_3': results['precisions'][2],
        'bleu_4': results['precisions'][3]
    }


def calculate_rouge_scores(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE scores
    """
    rouge_metric = evaluate.load("rouge")
    
    results = rouge_metric.compute(
        predictions=predictions,
        references=references
    )
    
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL'],
        'rougeLsum': results['rougeLsum']
    }


def calculate_perplexity(
    trainer,
    eval_dataset
) -> float:
    """
    Calculate perplexity on evaluation dataset.
    
    Args:
        trainer: Hugging Face Trainer object
        eval_dataset: Evaluation dataset
        
    Returns:
        Perplexity value
    """
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    perplexity = np.exp(eval_results['eval_loss'])
    return perplexity


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    trainer=None,
    max_samples: int = 100,
    input_field: str = "input",
    output_field: str = "output"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        test_dataset: Test dataset
        trainer: Optional Trainer for perplexity calculation
        max_samples: Maximum number of samples to evaluate
        input_field: Name of input field
        output_field: Name of output field
        
    Returns:
        Dictionary with all evaluation metrics
    """
    print("\n📊 Starting comprehensive evaluation...")
    print(f"   Evaluating on {min(max_samples, len(test_dataset))} samples")
    
    # Extract questions and references
    if hasattr(test_dataset, input_field):
        questions = [test_dataset[i][input_field] for i in range(min(max_samples, len(test_dataset)))]
        references = [test_dataset[i][output_field] for i in range(min(max_samples, len(test_dataset)))]
    else:
        questions = [test_dataset[i]['input'] for i in range(min(max_samples, len(test_dataset)))]
        references = [test_dataset[i]['output'] for i in range(min(max_samples, len(test_dataset)))]
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = batch_generate_predictions(
        model, tokenizer, questions,
        max_samples=max_samples
    )
    
    # Calculate BLEU
    print("\n📊 Calculating BLEU scores...")
    bleu_scores = calculate_bleu_score(predictions, references)
    
    # Calculate ROUGE
    print("📊 Calculating ROUGE scores...")
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    # Calculate perplexity if trainer provided
    perplexity = None
    if trainer is not None:
        print("📊 Calculating perplexity...")
        perplexity = calculate_perplexity(trainer, test_dataset)
    
    # Combine results
    results = {
        **bleu_scores,
        **rouge_scores
    }
    
    if perplexity is not None:
        results['perplexity'] = perplexity
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.4f}")
    print("=" * 80)
    
    return results


def compare_models(
    base_model,
    finetuned_model,
    tokenizer,
    test_dataset,
    max_samples: int = 50,
    input_field: str = "input",
    output_field: str = "output"
) -> pd.DataFrame:
    """
    Compare base model and fine-tuned model performance.
    
    Args:
        base_model: Base pre-trained model
        finetuned_model: Fine-tuned model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        max_samples: Number of samples to compare
        input_field: Name of input field
        output_field: Name of output field
        
    Returns:
        DataFrame with comparison results
    """
    print("\n🔄 Comparing base vs fine-tuned model...")
    
    # Extract test samples
    questions = [test_dataset[i][input_field] for i in range(min(max_samples, len(test_dataset)))]
    references = [test_dataset[i][output_field] for i in range(min(max_samples, len(test_dataset)))]
    
    # Generate predictions from both models
    print("\n📝 Generating base model predictions...")
    base_predictions = batch_generate_predictions(base_model, tokenizer, questions, max_samples=max_samples)
    
    print("📝 Generating fine-tuned model predictions...")
    finetuned_predictions = batch_generate_predictions(finetuned_model, tokenizer, questions, max_samples=max_samples)
    
    # Calculate metrics for both
    print("\n📊 Calculating metrics for base model...")
    base_bleu = calculate_bleu_score(base_predictions, references)
    base_rouge = calculate_rouge_scores(base_predictions, references)
    
    print("📊 Calculating metrics for fine-tuned model...")
    finetuned_bleu = calculate_bleu_score(finetuned_predictions, references)
    finetuned_rouge = calculate_rouge_scores(finetuned_predictions, references)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
        'Base Model': [
            base_bleu['bleu'],
            base_rouge['rouge1'],
            base_rouge['rouge2'],
            base_rouge['rougeL']
        ],
        'Fine-Tuned Model': [
            finetuned_bleu['bleu'],
            finetuned_rouge['rouge1'],
            finetuned_rouge['rouge2'],
            finetuned_rouge['rougeL']
        ]
    })
    
    # Calculate improvement
    comparison_df['Improvement (%)'] = (
        (comparison_df['Fine-Tuned Model'] - comparison_df['Base Model']) / 
        comparison_df['Base Model'] * 100
    )
    
    print("\n" + "=" * 80)
    print("🔄 MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    return comparison_df


def qualitative_evaluation(
    model,
    tokenizer,
    test_questions: List[str],
    max_new_tokens: int = 200
) -> List[Dict[str, str]]:
    """
    Perform qualitative evaluation with sample questions.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        test_questions: List of test questions
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of dictionaries with questions and responses
    """
    print("\n💬 Performing qualitative evaluation...")
    
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        response = generate_response(
            model, tokenizer, question,
            max_new_tokens=max_new_tokens
        )
        
        print(f"🤖 Response: {response[:200]}..." if len(response) > 200 else f"🤖 Response: {response}")
        
        results.append({
            'question': question,
            'response': response
        })
    
    return results


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Evaluation results saved to {output_path}")


def create_evaluation_report(
    results: Dict[str, Any],
    comparison_df: Optional[pd.DataFrame] = None,
    output_path: str = "evaluation_report.txt"
):
    """
    Create a comprehensive evaluation report.
    
    Args:
        results: Evaluation results
        comparison_df: Optional comparison DataFrame
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MEDICAL HEALTHCARE ASSISTANT - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EVALUATION METRICS:\n")
        f.write("-" * 80 + "\n")
        for metric, value in results.items():
            f.write(f"{metric:20s}: {value:.4f}\n")
        f.write("\n")
        
        if comparison_df is not None:
            f.write("MODEL COMPARISON:\n")
            f.write("-" * 80 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n📄 Evaluation report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("🧪 Testing evaluation utilities...")
    
    # Test sample data
    predictions = [
        "Hypertension is high blood pressure, a common cardiovascular condition.",
        "Diabetes is a metabolic disorder characterized by elevated blood glucose."
    ]
    
    references = [
        "Hypertension, or high blood pressure, is a chronic medical condition.",
        "Diabetes mellitus is a disease characterized by high blood sugar levels."
    ]
    
    # Calculate metrics
    print("\n📊 Testing BLEU calculation...")
    bleu_scores = calculate_bleu_score(predictions, references)
    print(f"BLEU Score: {bleu_scores['bleu']:.4f}")
    
    print("\n📊 Testing ROUGE calculation...")
    rouge_scores = calculate_rouge_scores(predictions, references)
    print(f"ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
    
    print("\n✅ All evaluation utilities working correctly!")
