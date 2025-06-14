"use client";

import React, { useState } from 'react';
import AttentionViewer from '@/components/AttentionViewer'; // Assuming @ is src
import HiddenStateExplorer from '@/components/HiddenStateExplorer'; // Assuming @ is src

interface AnalysisData {
  generated_text: string;
  processed_attentions: number[][][][]; // Layer -> Head -> SeqLen -> SeqLen
  processed_hidden_states: number[][][]; // Layer -> SeqLen -> HiddenSize
  model_used_for_testing?: string;
}

export default function Home() {
  const [prompt, setPrompt] = useState<string>('');
  const [modelName, setModelName] = useState<string>('meta-llama/Meta-Llama-3-8B-Instruct');

  const [results, setResults] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt, model_name: modelName }),
      });

      if (!response.ok) {
        const errorData = await response.json(); // Try to get error detail from backend
        throw new Error(errorData.detail || `HTTP error! status: ${response.status} ${response.statusText}`);
      }

      const data: AnalysisData = await response.json();
      setResults(data);
    } catch (e: any) {
      setError(e.message || "Failed to fetch analysis. Ensure the backend server is running.");
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center p-8 bg-gray-50">
      <div className="w-full max-w-2xl">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-800">LLM Reasoning Visualizer</h1>
        </header>

        <form onSubmit={handleSubmit} className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
          <div className="mb-6">
            <label htmlFor="prompt" className="block text-gray-700 text-sm font-bold mb-2">
              Enter your prompt:
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="shadow appearance-none border rounded w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:shadow-outline h-32 resize-none"
              placeholder="e.g., Explain the theory of relativity in simple terms."
              required
              disabled={isLoading}
            />
          </div>

          <div className="mb-6">
            <label htmlFor="modelName" className="block text-gray-700 text-sm font-bold mb-2">
              Select a model:
            </label>
            <select
              id="modelName"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="shadow border rounded w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:shadow-outline bg-white"
              disabled={isLoading}
            >
              <option value="meta-llama/Meta-Llama-3-8B-Instruct">Llama-3.1-8B (Instruct)</option>
              <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B (Instruct v0.2)</option>
              <option value="distilgpt2">DistilGPT-2 (for testing)</option>
            </select>
          </div>

          <div className="flex items-center justify-center">
            <button
              type="submit"
              className={`font-bold py-3 px-6 rounded focus:outline-none focus:shadow-outline transition-colors duration-150 ${
                isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-700 text-white'
              }`}
              disabled={isLoading}
            >
              {isLoading ? 'Analyzing...' : 'Generate & Analyze'}
            </button>
          </div>
        </form>

        <section className="mt-10 p-6 bg-white shadow-md rounded">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">Results</h2>
          {isLoading && <p className="text-blue-500">Analyzing, please wait...</p>}
          {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>}

          {!isLoading && !error && results && (
            <div>
              {/* Model Output Card */}
              <div className="p-4 border rounded shadow-md bg-white mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Model Output:</h3>
                {results.model_used_for_testing && (
                  <p className="text-sm text-yellow-700 bg-yellow-100 p-2 rounded mb-3">
                    Note: The backend used <strong>{results.model_used_for_testing}</strong> for this analysis as a substitute for the requested model for testing purposes.
                  </p>
                )}
                <p className="text-gray-700 whitespace-pre-wrap">{results.generated_text}</p>
              </div>

              {/* Tab Navigation */}
              <div className="flex border-b mb-4">
                <button
                  className={`py-2 px-4 font-medium text-sm ${activeTab === 'attention' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('attention')}
                >
                  Attention Viewer
                </button>
                <button
                  className={`py-2 px-4 font-medium text-sm ${activeTab === 'hiddenState' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('hiddenState')}
                >
                  Hidden State Explorer
                </button>
              </div>

              {/* Tab Content Area */}
              <div className="py-4">
                {activeTab === 'attention' && (
                  results?.processed_attentions && results.processed_attentions.length > 0 ? (
                    <AttentionViewer attentionData={results.processed_attentions} promptText={prompt} />
                  ) : (
                    <p className="text-gray-600">No attention data to display or data is empty.</p>
                  )
                )}
                {activeTab === 'hiddenState' && (
                  results?.processed_hidden_states && results.processed_hidden_states.length > 0 ? (
                    <HiddenStateExplorer hiddenStateData={results.processed_hidden_states} promptText={prompt} />
                  ) : (
                    <p className="text-gray-600">No hidden state data to display or data is empty.</p>
                  )
                )}
              </div>
            </div>
          )}

          {!isLoading && !error && !results && (
            <div id="results-placeholder" className="text-gray-600">
              Analysis results will appear here once generated.
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
