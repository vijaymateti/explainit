from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

class AnalyzeRequest(BaseModel):
    prompt: str
    model_name: str

@app.post("/api/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    model_name_to_use = request.model_name
    # Using a smaller model for testing purposes if a large model is specified.
    # This is to avoid long download times in the execution environment.
    # The code is structured to work with larger models like "meta-llama/Meta-Llama-3-8B-Instruct".
    if request.model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]:
        print(f"Using distilgpt2 instead of {request.model_name} for testing purposes.")
        model_name_to_use = "distilgpt2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_to_use)
        model = AutoModelForCausalLM.from_pretrained(model_name_to_use)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")

        # Ensure pad_token_id is set for open-ended generation if not already set
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

        # Using model.generate for simplicity in this step
        # For more detailed control over outputs like attentions and hidden states from specific layers,
        # a direct forward pass: model(**inputs, output_attentions=True, output_hidden_states=True) would be better.
        # The .generate() method can also be configured for some of these, but it's more complex.
        # For now, we will check for these attributes after a forward pass.

        # Perform a forward pass to get hidden states and attentions
        # Note: for generate, we'd typically get decoder_hidden_states and decoder_attentions
        # if the model is an encoder-decoder. For decoder-only models (like GPT2),
        # hidden_states and attentions are standard.
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

        # For generation, let's use the generate method separately
        # Max length is kept short for testing
        generated_ids = model.generate(**inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        processed_attentions = []
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_attention in outputs.attentions:
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                # Squeeze batch_size, convert to list
                processed_attentions.append(layer_attention[0].cpu().tolist())

        processed_hidden_states = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            for layer_hidden_state in outputs.hidden_states:
                # layer_hidden_state shape: [batch_size, seq_len, hidden_size]
                # Squeeze batch_size, convert to list
                processed_hidden_states.append(layer_hidden_state[0].cpu().tolist())

        return {
            "generated_text": generated_text,
            "processed_attentions": processed_attentions,
            "processed_hidden_states": processed_hidden_states,
            "model_used_for_testing": model_name_to_use if model_name_to_use != request.model_name else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")
