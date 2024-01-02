def generate_from_huggingface_completion(
        prompt: str,
        model,
        tokenizer,
        model_endpoint: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop_sequences: list[str] | None = None,
) -> str:
    tok_enc = tokenizer.encode(prompt)
    print(f"INPUT TOKENS: {len(tok_enc)}")
    input = tokenizer.decode(tok_enc)
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    generation_output = model.generate(
        **model_input,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    output = tokenizer.decode(generation_output[0])
    generation = output[len(input):]

    return generation
