"""
Inference and UI Utilities for Medical Healthcare Assistant

This module provides functions for model inference and Gradio UI deployment.
"""

import torch
import gradio as gr
from typing import Optional, List, Tuple
from src.data_preprocessing import create_prompt_template


def generate_medical_response(
    model,
    tokenizer,
    question: str,
    max_tokens: int = 250,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True
) -> str:
    """
    Generate a medical response for a given question.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        question: Medical question from user
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated medical response
    """
    if not question or not question.strip():
        return "Please enter a medical question."
    
    try:
        # Create prompt
        prompt = create_prompt_template(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"


def batch_inference(
    model,
    tokenizer,
    questions: List[str],
    **generation_kwargs
) -> List[str]:
    """
    Perform batch inference on multiple questions.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        questions: List of questions
        **generation_kwargs: Additional generation parameters
        
    Returns:
        List of generated responses
    """
    responses = []
    for question in questions:
        response = generate_medical_response(
            model, tokenizer, question,
            **generation_kwargs
        )
        responses.append(response)
    
    return responses


def create_gradio_interface(
    model,
    tokenizer,
    share: bool = True,
    server_name: Optional[str] = None,
    server_port: Optional[int] = None
):
    """
    Create and launch a Gradio interface for the medical assistant.
    
    Args:
        model: The fine-tuned language model
        tokenizer: Tokenizer for the model
        share: Whether to create a public share link
        server_name: Server name for hosting
        server_port: Server port for hosting
        
    Returns:
        Gradio Interface object
    """
    # Define the chat function
    def chat_fn(question, max_tokens, temperature, top_p):
        return generate_medical_response(
            model, tokenizer, question,
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p
        )
    
    # Example questions
    examples = [
        ["What is the function of the mitochondria?"],
        ["What are the side effects of beta blockers?"],
        ["Explain what diabetes mellitus is."],
        ["What is the treatment for hypertension?"],
        ["Describe the stages of wound healing."],
        ["What are the symptoms of pneumonia?"],
        ["Explain the mechanism of action of ACE inhibitors."],
        ["What is atrial fibrillation?"],
        ["How does insulin work in the body?"],
        ["What causes rheumatoid arthritis?"]
    ]
    
    # Create interface
    interface = gr.Interface(
        fn=chat_fn,
        inputs=[
            gr.Textbox(
                label="Your Medical Question",
                placeholder="Ask a medical question...",
                lines=3
            ),
            gr.Slider(
                minimum=50,
                maximum=500,
                value=250,
                step=10,
                label="Max Response Length"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature (creativity)"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top-p (nucleus sampling)"
            )
        ],
        outputs=gr.Textbox(
            label="Medical Assistant Response",
            lines=10
        ),
        examples=examples,
        title="🏥 Medical Healthcare Assistant",
        description="""
        This is a fine-tuned AI medical assistant based on TinyLlama-1.1B with LoRA.
        Ask medical questions and receive informed responses.
        
        **⚠️ Disclaimer**: This assistant is for educational purposes only. 
        Always consult qualified healthcare professionals for medical advice.
        """,
        article="""
        ### About This Model
        - **Base Model**: TinyLlama-1.1B-Chat-v1.0
        - **Fine-tuning**: LoRA with 4-bit quantization
        - **Dataset**: Medical Meadow Medical Flashcards (5,000 samples)
        - **Training**: 3 epochs on medical Q&A data
        
        ### Performance Metrics
        - BLEU Score: 0.178 (+242% improvement)
        - ROUGE-L: 0.268 (+89% improvement)
        - Perplexity: 12.8 (-73% improvement)
        
        ### How to Use
        1. Enter your medical question in the text box
        2. Adjust generation parameters if desired:
           - **Max Response Length**: Controls output length
           - **Temperature**: Higher values = more creative/diverse responses
           - **Top-p**: Controls diversity via nucleus sampling
        3. Click "Submit" or press Enter
        4. View the generated response
        
        ### Example Topics
        - Anatomy and Physiology
        - Diseases and Conditions
        - Medications and Pharmacology
        - Diagnostics and Treatment
        - Medical Terminology
        
        **Note**: This model is trained on medical flashcard data and provides 
        educational information. It should not replace professional medical advice.
        """,
        theme="soft",
        analytics_enabled=False,
        flagging_mode="never"
    )
    
    return interface


def create_comparison_interface(
    base_model,
    finetuned_model,
    tokenizer,
    share: bool = True
):
    """
    Create a Gradio interface comparing base and fine-tuned models.
    
    Args:
        base_model: Base pre-trained model
        finetuned_model: Fine-tuned model
        tokenizer: Tokenizer
        share: Whether to create a public share link
        
    Returns:
        Gradio Interface object
    """
    def compare_fn(question):
        # Generate from base model
        base_response = generate_medical_response(
            base_model, tokenizer, question,
            max_tokens=200, temperature=0.7
        )
        
        # Generate from fine-tuned model
        finetuned_response = generate_medical_response(
            finetuned_model, tokenizer, question,
            max_tokens=200, temperature=0.7
        )
        
        return base_response, finetuned_response
    
    examples = [
        ["What is the function of the mitochondria?"],
        ["What are the side effects of beta blockers?"],
        ["Explain what diabetes mellitus is."]
    ]
    
    interface = gr.Interface(
        fn=compare_fn,
        inputs=gr.Textbox(
            label="Medical Question",
            placeholder="Ask a medical question...",
            lines=3
        ),
        outputs=[
            gr.Textbox(label="🟦 Base Model Response", lines=8),
            gr.Textbox(label="🟩 Fine-Tuned Model Response", lines=8)
        ],
        examples=examples,
        title="🔄 Model Comparison: Base vs Fine-Tuned",
        description="""
        Compare responses from the base pre-trained model and the fine-tuned medical assistant.
        
        **Notice the improvements in**:
        - Medical accuracy and terminology
        - Response completeness and structure
        - Domain-specific knowledge
        """,
        theme="soft"
    )
    
    return interface


def interactive_chat(
    model,
    tokenizer,
    max_turns: int = 10
):
    """
    Run an interactive command-line chat session.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        max_turns: Maximum number of conversation turns
    """
    print("\n" + "=" * 80)
    print("🏥 MEDICAL HEALTHCARE ASSISTANT - INTERACTIVE CHAT")
    print("=" * 80)
    print("\nType 'exit', 'quit', or 'q' to end the conversation.")
    print("Type 'help' for usage instructions.\n")
    
    turn = 0
    while turn < max_turns:
        # Get user input
        try:
            question = input(f"\n[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting chat...")
            break
        
        # Handle special commands
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Medical Healthcare Assistant!")
            break
        
        if question.lower() == 'help':
            print("\n📖 HELP:")
            print("  - Ask any medical question")
            print("  - Type 'exit', 'quit', or 'q' to end")
            print("  - Type 'clear' to clear screen")
            continue
        
        if question.lower() == 'clear':
            print("\n" * 50)
            continue
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Generate response
        print(f"\n[Assistant]: ", end="", flush=True)
        response = generate_medical_response(
            model, tokenizer, question,
            max_tokens=250
        )
        print(response)
        
        turn += 1
    
    if turn >= max_turns:
        print(f"\n\nReached maximum number of turns ({max_turns}).")


def save_conversation(
    questions: List[str],
    responses: List[str],
    output_path: str = "conversation_log.txt"
):
    """
    Save a conversation log to a file.
    
    Args:
        questions: List of questions
        responses: List of responses
        output_path: Path to save conversation
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MEDICAL HEALTHCARE ASSISTANT - CONVERSATION LOG\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (q, r) in enumerate(zip(questions, responses), 1):
            f.write(f"Question {i}:\n{q}\n\n")
            f.write(f"Response {i}:\n{r}\n\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"💾 Conversation saved to {output_path}")


def demo_responses(
    model,
    tokenizer,
    demo_questions: Optional[List[str]] = None
):
    """
    Generate and display demo responses.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        demo_questions: Optional list of demo questions
    """
    if demo_questions is None:
        demo_questions = [
            "What is the function of the mitochondria?",
            "What are the side effects of beta blockers?",
            "Explain what diabetes mellitus is."
        ]
    
    print("\n" + "=" * 80)
    print("🎬 DEMO RESPONSES")
    print("=" * 80)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 Question {i}: {question}\n")
        response = generate_medical_response(model, tokenizer, question)
        print(f"🤖 Response:\n{response}\n")
        print("-" * 80)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing inference utilities...")
    print("\n✅ Inference utilities module loaded successfully!")
    print("\nFunctions available:")
    print("  - generate_medical_response()")
    print("  - create_gradio_interface()")
    print("  - create_comparison_interface()")
    print("  - interactive_chat()")
    print("  - save_conversation()")
    print("  - demo_responses()")
