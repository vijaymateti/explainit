import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app # Changed to absolute import

client = TestClient(app)

# Helper to create a mock model and tokenizer
def create_mock_model_tokenizer():
    mock_tokenizer = MagicMock()
    # Simulate tokenizer call: tokenizer("prompt", return_tensors="pt")
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1,1,1]]} # Dummy tokenized output
    mock_tokenizer.eos_token_id = 50256 # A common eos_token_id

    mock_model = MagicMock()
    mock_model.config.pad_token_id = mock_tokenizer.eos_token_id

    # Simulate model forward pass: model(**inputs, output_attentions=True, output_hidden_states=True)
    # Needs 'attentions' and 'hidden_states' attributes which are tuples of tensors (mocked as lists of lists)
    # Dimensions:
    # Attentions: layer_count x [batch_size, num_heads, seq_len, seq_len] -> using 1 layer, 1 head, seq_len 3
    # Hidden States: layer_count x [batch_size, seq_len, hidden_size] -> using 1 layer, seq_len 3, hidden_size 4
    mock_outputs = MagicMock()

    # Mock attentions
    # Data for one layer: B x H x S x S (1x1x3x3)
    # The .tolist() in main.py will get rid of the batch dim for attentions
    # So, the mock_attention_layer_tensor[0].cpu().tolist() should return H x S x S
    dummy_attention_data_one_layer_squeezed = [[[0.1]*3]*3]*1 # H x S x S (Heads x SeqLen x SeqLen)
    mock_attention_layer_tensor = MagicMock()
    # Simulate layer_attention[0].cpu().tolist()
    mock_attention_layer_tensor.__getitem__(0).cpu.return_value.tolist.return_value = dummy_attention_data_one_layer_squeezed
    mock_outputs.attentions = tuple([mock_attention_layer_tensor] * 1) # 1 layer

    # Mock hidden states
    # Data for one layer: B x S x H_dim (1x3x4)
    # The .tolist() in main.py will get rid of the batch dim for hidden_states
    # So, the mock_hidden_state_layer_tensor[0].cpu().tolist() should return S x H_dim
    dummy_hidden_state_data_one_layer_squeezed = [[0.1]*4]*3 # S x H_dim (Tokens x HiddenDimension)
    mock_hidden_state_layer_tensor = MagicMock()
    # Simulate layer_hidden_state[0].cpu().tolist()
    mock_hidden_state_layer_tensor.__getitem__(0).cpu.return_value.tolist.return_value = dummy_hidden_state_data_one_layer_squeezed
    mock_outputs.hidden_states = tuple([mock_hidden_state_layer_tensor] * 1) # 1 layer

    mock_model.return_value = mock_outputs # This mocks the __call__ behavior for forward pass

    # Simulate model.generate()
    # generate returns a tensor of token ids, e.g., [[1, 2, 3, 4]]
    mock_model.generate.return_value = [[1, 2, 3, 4]] # Dummy generated token IDs

    # Simulate tokenizer.decode()
    mock_tokenizer.decode.return_value = "mocked text output"

    return mock_model, mock_tokenizer

@patch('main.AutoModelForCausalLM.from_pretrained')
@patch('main.AutoTokenizer.from_pretrained')
def test_analyze_success(mock_tokenizer_from_pretrained, mock_model_from_pretrained):
    mock_model, mock_tokenizer = create_mock_model_tokenizer()
    mock_model_from_pretrained.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    response = client.post("/api/analyze", json={"prompt": "Hello world", "model_name": "distilgpt2"})
    if response.status_code != 200:
        print("test_analyze_success failed response:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert "generated_text" in data
    assert "processed_attentions" in data
    assert "processed_hidden_states" in data

    # Check if distilgpt2 was used (as per current backend logic for this model name)
    # No model_used_for_testing key is added if the requested model is used directly.
    assert data.get("model_used_for_testing") is None

    # Basic structure checks for attentions (list of layers -> list of heads -> list of lists)
    assert isinstance(data["processed_attentions"], list)
    if len(data["processed_attentions"]) > 0:
        layer_0_att = data["processed_attentions"][0]
        assert isinstance(layer_0_att, list) # Heads
        if len(layer_0_att) > 0:
            head_0_att = layer_0_att[0]
            assert isinstance(head_0_att, list) # Query sequence length
            if len(head_0_att) > 0:
                 assert isinstance(head_0_att[0], list) # Key sequence length (should be numbers here)


    # Basic structure checks for hidden states (list of layers -> list of tokens -> list of floats)
    assert isinstance(data["processed_hidden_states"], list)
    if len(data["processed_hidden_states"]) > 0:
        layer_0_hs = data["processed_hidden_states"][0]
        assert isinstance(layer_0_hs, list) # Tokens
        if len(layer_0_hs) > 0:
            token_0_hs = layer_0_hs[0]
            assert isinstance(token_0_hs, list) # Hidden state vector components (floats)
            if len(token_0_hs) > 0:
                assert isinstance(token_0_hs[0], float) # or int, depending on model precision

@patch('main.AutoModelForCausalLM.from_pretrained')
@patch('main.AutoTokenizer.from_pretrained')
def test_analyze_actual_model_name_substitution(mock_tokenizer_from_pretrained, mock_model_from_pretrained):
    mock_model, mock_tokenizer = create_mock_model_tokenizer()
    mock_model_from_pretrained.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # This test checks if a larger model name gets substituted by distilgpt2 as per current backend logic
    response = client.post("/api/analyze", json={"prompt": "Explain AI", "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"})
    if response.status_code != 200:
        print("test_analyze_actual_model_name_substitution failed response:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert "generated_text" in data
    assert data.get("model_used_for_testing") == "distilgpt2"

def test_analyze_invalid_model_name():
    # This model should not exist. The from_pretrained mocks won't be called if the name isn't "distilgpt2"
    # or one of the recognized large models that get substituted.
    # So, the original behavior of attempting to load and failing (due to network or model not found) is expected.
    # We need to ensure the mocks are NOT called, or that they raise an error if called with this name.

    # For this test, we want the original from_pretrained to be called, and it should fail.
    # So we don't apply the broad mocks here, or we make them selective.
    # A simpler way is to let it call the actual functions, which will fail due to network/not found.
    # The previous run showed this correctly results in a 500.

    # To make this test independent of network for "this-model-does-not-exist-123",
    # we can patch from_pretrained to raise an exception when called with this specific model name.
    def selective_mock_from_pretrained(model_name_str):
        if model_name_str == "this-model-does-not-exist-123":
            raise Exception("Mocked: Model not found")
        # For "distilgpt2" or other models that might be tried as fallbacks by the endpoint,
        # we could return a working mock, but the endpoint logic as tested for this-model-does-not-exist-123
        # doesn't seem to have a fallback TO distilgpt2, only FROM other specified large models.
        # So, if "distilgpt2" is attempted by the SUT after "this-model-does-not-exist-123" fails,
        # this mock would need to handle that.
        # For now, assume the SUT fails directly on "this-model-does-not-exist-123".
        mock_model, _ = create_mock_model_tokenizer() # We only need the model part for this specific mock
        return mock_model


    with patch('main.AutoModelForCausalLM.from_pretrained', side_effect=selective_mock_from_pretrained), \
         patch('main.AutoTokenizer.from_pretrained', side_effect=selective_mock_from_pretrained): # Tokenizer also needs to be mocked
        response = client.post("/api/analyze", json={"prompt": "Test", "model_name": "this-model-does-not-exist-123"})

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    # The error message will now come from our mocked exception if selective_mock_from_pretrained is triggered.
    # Or, if the endpoint logic changes, it might be different.
    # Let's check for the mocked message or the original network error.
    assert "Error loading model: Mocked: Model not found" in data["detail"] or \
           "We couldn't connect to 'https://huggingface.co'" in data["detail"]


def test_analyze_missing_prompt():
    response = client.post("/api/analyze", json={"model_name": "distilgpt2"}) # Missing "prompt"
    assert response.status_code == 422  # Unprocessable Entity from Pydantic
    data = response.json()
    assert "detail" in data
    # Check if the error detail points out the missing 'prompt' field for Pydantic v2
    found_prompt_error = False
    for error in data["detail"]:
        if error.get("type") == "missing" and "prompt" in error.get("loc", []):
            found_prompt_error = True
            break
    assert found_prompt_error

def test_analyze_missing_model_name():
    response = client.post("/api/analyze", json={"prompt": "Hello"}) # Missing "model_name"
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data
    found_model_error = False
    # Check if the error detail points out the missing 'model_name' field for Pydantic v2
    for error in data["detail"]:
        if error.get("type") == "missing" and "model_name" in error.get("loc", []):
            found_model_error = True
            break
    assert found_model_error
