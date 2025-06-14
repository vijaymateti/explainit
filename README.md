# LLM Reasoning Visualizer

## Overview/Purpose

The LLM Reasoning Visualizer is a web application designed to help users understand the internal workings of Large Language Models (LLMs). By providing a prompt and selecting a model, users can see not only the generated text but also visualizations of the model's attention mechanisms and hidden state representations across its layers.

This project is inspired by research aimed at making LLMs more transparent and interpretable, such as the work done by Anthropic and others in the field. Understanding how these models arrive at their outputs is crucial for debugging, improving performance, and ensuring safe and reliable behavior.

**Key Technologies:**
*   **Frontend:** React, Next.js, TypeScript, Tailwind CSS
*   **Backend:** Python, FastAPI, Pydantic
*   **LLM Interaction:** Hugging Face `transformers` library, PyTorch

## Features

*   **Prompt Analysis:** Input custom prompts to be processed by selected LLMs.
    *   Supports models like `distilgpt2` (primarily for testing/demonstration in the current setup).
    *   Designed to support larger models such as Llama 3.1 (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) and Mistral-7B (e.g., `mistralai/Mistral-7B-Instruct-v0.2`), though the backend may substitute these with `distilgpt2` for resource reasons in the default configuration.
*   **Generated Text Viewing:** Clearly displays the text output produced by the model.
*   **Attention Visualization (Attention Viewer):**
    *   Allows users to select specific layers and attention heads.
    *   Displays attention scores as a heatmap, showing how much each token attends to other tokens in the input sequence.
*   **Hidden State Exploration (Hidden State Explorer):**
    *   Allows users to select a specific token from the input prompt.
    *   Visualizes the change in the L2 norm of the selected token's hidden state vector across all layers of the model, providing insight into how the token's representation evolves.

## Project Structure

The project is organized as a monorepo with two main components:

```
/
├── backend/      # FastAPI application (Python)
├── frontend/     # Next.js application (TypeScript/React)
└── README.md     # This file
```

## Setup and Installation

### Prerequisites

*   **Node.js:** Version 18.0 or later (e.g., v18+, v20+).
*   **Python:** Version 3.9 or later (e.g., Python 3.9, 3.10, 3.11).
*   **pip:** Python package installer (usually comes with Python).

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The `requirements.txt` is configured to install a CPU-only version of PyTorch to save space.
    ```bash
    pip install -r requirements.txt
    ```

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
    (Or `yarn install` if you prefer Yarn and have it installed, though `package-lock.json` is present.)

## Running the Application

### 1. Run Backend Server

*   Ensure you are in the `backend` directory.
*   Activate your virtual environment if you created one (e.g., `source venv/bin/activate`).
*   Start the FastAPI server using Uvicorn:
    ```bash
    uvicorn main:app --reload --port 8000
    ```
*   The backend server will be accessible at `http://localhost:8000`. The `--reload` flag enables auto-reloading on code changes.

### 2. Run Frontend Development Server

*   Ensure you are in the `frontend` directory.
*   Start the Next.js development server:
    ```bash
    npm run dev
    ```
*   The frontend application will be accessible at `http://localhost:3000`.
*   API requests from the frontend to `/api/...` are automatically proxied to the backend server running on port 8000 (this needs to be configured in `next.config.js` if not already done, but is standard practice). *For this project, the frontend directly calls `http://localhost:8000/api/analyze`.*

Once both servers are running, open `http://localhost:3000` in your web browser to use the application.

## API Endpoint Description

The backend provides a single primary endpoint for analysis.

### `POST /api/analyze`

This endpoint accepts a user prompt and a model identifier, processes the prompt through the specified language model, and returns the generated text along with attention and hidden state data.

**Request Body (JSON):**

```json
{
  "prompt": "Your input text for the language model.",
  "model_name": "identifier_of_the_model_from_hugging_face"
}
```

*   **`prompt` (string, required):** The text you want the language model to process.
*   **`model_name` (string, required):** The Hugging Face model identifier.
    *   Examples: `"distilgpt2"`, `"meta-llama/Meta-Llama-3-8B-Instruct"`, `"mistralai/Mistral-7B-Instruct-v0.2"`.
    *   **Note:** The current backend implementation is configured to use `"distilgpt2"` as a substitute if large model names like `"meta-llama/Meta-Llama-3-8B-Instruct"` or `"mistralai/Mistral-7B-Instruct-v0.2"` are specified. This is for demonstration purposes and to manage computational resources. The response will indicate if such a substitution occurred.

**Success Response (200 OK - JSON):**

The response contains the model's output and the processed data for visualization.

```json
{
  "generated_text": "The model's textual output based on the prompt.",
  "processed_attentions": [ /* Array of layers */
    [ /* Array of heads per layer */
      [ /* 2D array: Query Token Index -> Key Token Index attention scores */ ]
    ]
  ],
  "processed_hidden_states": [ /* Array of layers (incl. embeddings) */
    [ /* Array of tokens per layer */
      [ /* Array of hidden dimension values for that token */ ]
    ]
  ],
  "model_used_for_testing": "distilgpt2" // Optional: Present if a substitution occurred
}
```

*   **`generated_text` (string):** The text generated by the language model.
*   **`processed_attentions` (array):** A nested array representing attention scores. Dimensions are typically: `[num_layers][num_heads][sequence_length][sequence_length]`.
*   **`processed_hidden_states` (array):** A nested array representing hidden states. Dimensions are typically: `[num_layers_plus_embeddings][sequence_length][hidden_size]`.
*   **`model_used_for_testing` (string, optional):** If present, indicates that the requested `model_name` was substituted with a smaller model (e.g., "distilgpt2") for testing or resource reasons.

**Error Responses:**

*   **`422 Unprocessable Entity`:** Returned if the request body fails validation (e.g., `prompt` or `model_name` is missing). The response will contain details about the validation errors.
*   **`500 Internal Server Error`:** Returned if there's an issue on the server-side during model loading, inference, or data processing. The response may contain a `detail` field with more information about the error.

## How to Use Visualizations

Once you submit a prompt and the backend returns results:

1.  **Model Output:** The text generated by the LLM will be displayed at the top of the results section.

2.  **Attention Viewer Tab:**
    *   **Selection:** Use the "Layer" and "Head" dropdown menus to choose the specific attention layer and head you want to inspect.
    *   **Heatmap:** The visualization displays a heatmap where rows and columns represent the tokens in your input prompt (as tokenized by a simple frontend tokenizer). The color intensity of each cell `(row_i, col_j)` indicates the attention score from token `i` (query) to token `j` (key). Darker cells mean higher attention. Hover over a cell to see the exact score.

3.  **Hidden State Explorer Tab:**
    *   **Token Selection:** Use the "Select Token" dropdown to choose a specific token from your input prompt.
    *   **L2 Norm Chart:** A line chart visualizes the L2 norm (magnitude) of the selected token's hidden state vector as it passes through each layer of the model. The X-axis represents the layers (from input embeddings to the final layer), and the Y-axis represents the calculated L2 norm. This helps to see how the representation of that specific token evolves or changes in strength throughout the model.
```
