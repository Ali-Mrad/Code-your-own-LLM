# tasks/instruction_finetune/inference.py
import torch, tiktoken
from main_2.generation import generate
from .data import format_input
from main_2.train import text_to_token_ids, token_ids_to_text

tok = tiktoken.get_encoding("gpt2")

def generate_response(
    model,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int | None = 40,
    device: str = "cpu",
):
    prompt = format_input({"instruction": instruction, "input": input_text})
    idx    = text_to_token_ids(prompt, tok).to(device)         
    out_ids = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=model.cfg["context_length"],
        temperature=temperature,
        top_k=top_k,
    )
    full_text = token_ids_to_text(out_ids, tok)                
    return full_text.split("### Response:\n")[-1].strip()