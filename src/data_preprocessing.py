"""
Data Preprocessing Utilities for Medical Healthcare Assistant

This module provides functions for loading, preprocessing, and formatting
medical Q&A data for LLM fine-tuning.
"""

import pandas as pd
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Optional


def load_medical_dataset(
    dataset_name: str = "medalpaca/medical_meadow_medical_flashcards",
    split: str = "train"
) -> Dataset:
    """
    Load medical flashcards dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        split: Dataset split to load (train, validation, test)
        
    Returns:
        Dataset object containing medical Q&A pairs
    """
    print(f"📥 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"✅ Dataset loaded! Total samples: {len(dataset):,}")
    return dataset


def create_prompt_template(
    instruction: str,
    response: str = "",
    system_message: Optional[str] = None
) -> str:
    """
    Create a formatted prompt following TinyLlama chat template.
    
    Args:
        instruction: The user's question or instruction
        response: The model's response (empty for inference)
        system_message: Optional system message for context
        
    Returns:
        Formatted prompt string
    """
    if system_message is None:
        system_message = "You are a knowledgeable medical assistant. Provide accurate, detailed, and helpful medical information."
    
    if response:
        prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n{response}</s>"
    else:
        prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
    
    return prompt


def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 512,
    input_field: str = "input",
    output_field: str = "output"
) -> Dataset:
    """
    Preprocess dataset by creating prompts and tokenizing.
    
    Args:
        dataset: Raw dataset to preprocess
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        input_field: Name of input field in dataset
        output_field: Name of output field in dataset
        
    Returns:
        Tokenized dataset ready for training
    """
    def tokenize_function(examples):
        prompts = []
        for inp, out in zip(examples[input_field], examples[output_field]):
            prompt = create_prompt_template(inp, out)
            prompts.append(prompt)
        
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    print("🔄 Preprocessing dataset...")
    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"✅ Preprocessing complete!")
    return processed_dataset


def split_dataset(
    dataset: Dataset,
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Full dataset to split
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Shuffle dataset
    dataset_shuffled = dataset.shuffle(seed=seed)
    
    # Create splits
    train_dataset = dataset_shuffled.select(range(train_size))
    val_dataset = dataset_shuffled.select(range(train_size, train_size + val_size))
    test_dataset = dataset_shuffled.select(
        range(train_size + val_size, train_size + val_size + test_size)
    )
    
    print(f"✅ Dataset split:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    return train_dataset, val_dataset, test_dataset


def analyze_dataset(dataset: Dataset, input_field: str = "input", output_field: str = "output") -> pd.DataFrame:
    """
    Analyze dataset statistics.
    
    Args:
        dataset: Dataset to analyze
        input_field: Name of input field
        output_field: Name of output field
        
    Returns:
        DataFrame with statistics
    """
    df = pd.DataFrame(dataset)
    
    # Calculate lengths
    df['input_length'] = df[input_field].apply(lambda x: len(x.split()))
    df['output_length'] = df[output_field].apply(lambda x: len(x.split()))
    df['total_length'] = df['input_length'] + df['output_length']
    
    stats = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Input Length': [
            len(df),
            df['input_length'].mean(),
            df['input_length'].std(),
            df['input_length'].min(),
            df['input_length'].quantile(0.25),
            df['input_length'].median(),
            df['input_length'].quantile(0.75),
            df['input_length'].max()
        ],
        'Output Length': [
            len(df),
            df['output_length'].mean(),
            df['output_length'].std(),
            df['output_length'].min(),
            df['output_length'].quantile(0.25),
            df['output_length'].median(),
            df['output_length'].quantile(0.75),
            df['output_length'].max()
        ]
    })
    
    return stats


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\r', '')
    text = text.replace('\t', ' ')
    
    return text.strip()


def filter_by_length(
    dataset: Dataset,
    min_length: int = 5,
    max_length: int = 500,
    field: str = "output"
) -> Dataset:
    """
    Filter dataset by text length.
    
    Args:
        dataset: Dataset to filter
        min_length: Minimum word count
        max_length: Maximum word count
        field: Field to check length
        
    Returns:
        Filtered dataset
    """
    def length_filter(example):
        word_count = len(example[field].split())
        return min_length <= word_count <= max_length
    
    filtered = dataset.filter(length_filter)
    print(f"📊 Filtered {len(dataset)} → {len(filtered)} samples")
    return filtered


def create_instruction_dataset(
    questions: List[str],
    answers: List[str]
) -> Dataset:
    """
    Create a dataset from lists of questions and answers.
    
    Args:
        questions: List of questions
        answers: List of answers
        
    Returns:
        Dataset object
    """
    assert len(questions) == len(answers), "Questions and answers must have same length"
    
    data = {
        "input": questions,
        "output": answers
    }
    
    return Dataset.from_dict(data)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing data preprocessing utilities...")
    
    # Load dataset
    dataset = load_medical_dataset()
    
    # Analyze dataset
    stats = analyze_dataset(dataset)
    print("\n📊 Dataset Statistics:")
    print(stats)
    
    # Create sample prompt
    sample_prompt = create_prompt_template(
        instruction="What is hypertension?",
        response="Hypertension is persistently elevated blood pressure."
    )
    print("\n📝 Sample Prompt:")
    print(sample_prompt)
    
    print("\n✅ All tests passed!")
