import React, { useState, useMemo } from 'react';

interface AttentionViewerProps {
  attentionData: number[][][][]; // Layers -> Heads -> SeqLen_Query -> SeqLen_Key
  promptText: string;
}

// Simple tokenizer (split by space and some punctuation)
const simpleTokenize = (text: string): string[] => {
  if (!text) return [];
  // Basic split by space, and retain some punctuation as separate tokens or attached
  // This is a very rough approximation.
  return text.split(/\s+|(?=[.,;:!?])|(?<=[.,;:!?])/g).filter(token => token.trim() !== '');
};

const AttentionViewer: React.FC<AttentionViewerProps> = ({ attentionData, promptText }) => {
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(0);

  const displayTokens = useMemo(() => simpleTokenize(promptText), [promptText]);

  if (!attentionData || attentionData.length === 0) {
    return <p className="text-gray-600">Attention data is not available or is empty.</p>;
  }

  const numLayers = attentionData.length;
  const numHeads = attentionData[selectedLayer]?.length || 0;

  const handleLayerChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newLayer = parseInt(event.target.value, 10);
    setSelectedLayer(newLayer);
    setSelectedHead(0); // Reset head selection when layer changes
  };

  const handleHeadChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedHead(parseInt(event.target.value, 10));
  };

  const currentAttentionMatrix = attentionData[selectedLayer]?.[selectedHead];

  if (!currentAttentionMatrix || currentAttentionMatrix.length === 0 || currentAttentionMatrix.length !== displayTokens.length) {
    // Add a check for matrix-token length consistency if possible (might be off due to simple tokenization)
    console.warn("Token length vs attention matrix dimension mismatch or invalid data for current selection.",
                 "Tokens:", displayTokens.length,
                 "Matrix Dim1:", currentAttentionMatrix?.length,
                 "Matrix Dim2:", currentAttentionMatrix?.[0]?.length);
    // It's possible the server's tokenization differs. For now, we'll proceed if matrix exists.
    // A more robust solution would involve passing tokens from backend or using identical tokenizer.
  }

  // Normalize score for color intensity (0-1 range)
  const getNormalizedScore = (score: number, minScore: number, maxScore: number) => {
    if (maxScore === minScore) return 0.5; // Avoid division by zero, return mid-intensity
    return (score - minScore) / (maxScore - minScore);
  }

  // Find min/max scores in the current matrix for normalization (can be slow for large matrices)
  // For simplicity, we assume scores are generally positive and might not always be 0-1.
  // A more performant approach might pre-calculate this or assume a range.
  let minScore = 0;
  let maxScore = 1;
  if (currentAttentionMatrix) {
    minScore = Math.min(...currentAttentionMatrix.flat());
    maxScore = Math.max(...currentAttentionMatrix.flat());
  }


  return (
    <div className="p-4 bg-white shadow rounded">
      <h3 className="text-lg font-semibold mb-4 text-gray-700">Attention Viewer</h3>

      <div className="flex gap-4 mb-4">
        <div>
          <label htmlFor="layer-select" className="block text-sm font-medium text-gray-700 mb-1">Layer:</label>
          <select
            id="layer-select"
            value={selectedLayer}
            onChange={handleLayerChange}
            className="p-2 border rounded bg-white shadow-sm w-full sm:w-auto"
            disabled={numLayers === 0}
          >
            {Array.from({ length: numLayers }, (_, i) => (
              <option key={i} value={i}>Layer {i}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="head-select" className="block text-sm font-medium text-gray-700 mb-1">Head:</label>
          <select
            id="head-select"
            value={selectedHead}
            onChange={handleHeadChange}
            className="p-2 border rounded bg-white shadow-sm w-full sm:w-auto"
            disabled={numHeads === 0}
          >
            {Array.from({ length: numHeads }, (_, i) => (
              <option key={i} value={i}>Head {i}</option>
            ))}
          </select>
        </div>
      </div>

      {(!currentAttentionMatrix || displayTokens.length === 0) && (
        <p className="text-gray-500">Select a layer/head to view attention, or prompt seems empty.</p>
      )}

      {currentAttentionMatrix && displayTokens.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse border border-gray-300">
            <thead>
              <tr>
                <th className="p-2 border border-gray-300 bg-gray-100 sticky left-0 z-10">Token (Query)</th>
                {displayTokens.map((token, idx) => (
                  <th key={`col-header-${idx}`} className="p-1 sm:p-2 border border-gray-300 bg-gray-50 text-xs sm:text-sm whitespace-nowrap transform -rotate-45 min-w-[50px] h-[70px]">
                    <div className="translate-y-[-10px] translate-x-[5px] sm:translate-x-[10px]">
                      {token} ({idx})
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayTokens.map((queryToken, i) => (
                <tr key={`row-${i}`}>
                  <td className="p-1 sm:p-2 border border-gray-300 bg-gray-50 text-xs sm:text-sm font-medium whitespace-nowrap sticky left-0 z-10 min-w-[100px]">
                    {queryToken} ({i})
                  </td>
                  {displayTokens.map((keyToken, j) => {
                    // Ensure matrix is not ragged and indices are valid
                    const score = currentAttentionMatrix[i]?.[j] ?? 0;
                    const normalizedScore = getNormalizedScore(score, minScore, maxScore);
                    constbgColor = `rgba(59, 130, 246, ${normalizedScore.toFixed(2)})`; // Blue intensity

                    return (
                      <td
                        key={`cell-${i}-${j}`}
                        className="p-1 sm:p-2 border border-gray-200 text-center min-w-[50px]"
                        style={{ backgroundColor: bgColor, color: normalizedScore > 0.5 ? 'white' : 'black' }}
                        title={`Attention from "${queryToken}" (row ${i}) to "${keyToken}" (col ${j}): ${score.toFixed(4)}`}
                      >
                        <span className="text-xs">{score.toFixed(2)}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AttentionViewer;
