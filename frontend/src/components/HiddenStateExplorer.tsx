import React, { useState, useMemo } from 'react';

interface HiddenStateExplorerProps {
  hiddenStateData: number[][][]; // Layers -> Tokens -> HiddenSize
  promptText: string;
}

// Simple tokenizer (reuse or redefine if different logic needed)
const simpleTokenize = (text: string): string[] => {
  if (!text) return [];
  return text.split(/\s+|(?=[.,;:!?])|(?<=[.,;:!?])/g).filter(token => token.trim() !== '');
};

const HiddenStateExplorer: React.FC<HiddenStateExplorerProps> = ({ hiddenStateData, promptText }) => {
  const displayTokens = useMemo(() => simpleTokenize(promptText), [promptText]);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(displayTokens.length > 0 ? 0 : null);

  if (!hiddenStateData || hiddenStateData.length === 0) {
    return <p className="text-gray-600">Hidden state data is not available or is empty.</p>;
  }

  // Check for token mismatch (optional, based on how critical exact alignment is for this viz)
  if (displayTokens.length > 0 && hiddenStateData[0] && displayTokens.length !== hiddenStateData[0].length) {
    console.warn(
      `Token count (${displayTokens.length}) from frontend tokenizer ` +
      `does not match hidden state token dimension (${hiddenStateData[0].length}). ` +
      `Visualization might be misaligned or based on potentially incorrect token mapping.`
    );
  }

  const handleTokenChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTokenIndex(parseInt(event.target.value, 10));
  };

  let chartData: { layer: number; l2norm: number }[] = [];
  if (selectedTokenIndex !== null && hiddenStateData && hiddenStateData.length > 0) {
    hiddenStateData.forEach((layerData, layerIndex) => {
      if (layerData && layerData[selectedTokenIndex]) {
        const vector = layerData[selectedTokenIndex];
        const l2norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        chartData.push({ layer: layerIndex, l2norm });
      }
    });
  }

  // SVG Chart Constants
  const svgWidth = 500;
  const svgHeight = 300;
  const padding = 50;
  const chartWidth = svgWidth - 2 * padding;
  const chartHeight = svgHeight - 2 * padding;

  const maxL2Norm = chartData.length > 0 ? Math.max(...chartData.map(d => d.l2norm)) : 0;
  const minL2Norm = chartData.length > 0 ? Math.min(...chartData.map(d => d.l2norm)) : 0;

  const getX = (layerIndex: number) =>
    padding + (layerIndex / (Math.max(1, hiddenStateData.length -1 ))) * chartWidth;

  const getY = (l2norm: number) =>
    padding + chartHeight - ((l2norm - minL2Norm) / Math.max(1, maxL2Norm - minL2Norm)) * chartHeight;


  return (
    <div className="p-4 bg-white shadow rounded">
      <h3 className="text-lg font-semibold mb-4 text-gray-700">Hidden State Explorer</h3>

      <div className="mb-4">
        <label htmlFor="token-select" className="block text-sm font-medium text-gray-700 mb-1">Select Token:</label>
        <select
          id="token-select"
          value={selectedTokenIndex ?? ''}
          onChange={handleTokenChange}
          className="p-2 border rounded bg-white shadow-sm w-full sm:w-auto"
          disabled={displayTokens.length === 0}
        >
          {displayTokens.map((token, idx) => (
            <option key={idx} value={idx}>
              {token} (Index {idx})
            </option>
          ))}
        </select>
      </div>

      {selectedTokenIndex === null && <p className="text-gray-500">Please select a token to view its hidden state L2 norms across layers.</p>}

      {chartData.length > 0 && selectedTokenIndex !== null && (
        <div>
          <p className="text-sm text-gray-600 mb-2">
            Showing L2 Norm of hidden state for token: <strong>"{displayTokens[selectedTokenIndex]}"</strong> across layers.
          </p>
          <svg width={svgWidth} height={svgHeight} className="border rounded">
            {/* X Axis */}
            <line x1={padding} y1={svgHeight - padding} x2={svgWidth - padding} y2={svgHeight - padding} stroke="currentColor" />
            {hiddenStateData.map((_, layerIndex) => (
              <text key={`x-tick-${layerIndex}`} x={getX(layerIndex)} y={svgHeight - padding + 20} textAnchor="middle" className="text-xs">
                L{layerIndex}
              </text>
            ))}
            <text x={svgWidth / 2} y={svgHeight - padding + 40} textAnchor="middle" className="text-sm">Layer</text>


            {/* Y Axis */}
            <line x1={padding} y1={padding} x2={padding} y2={svgHeight - padding} stroke="currentColor" />
            {/* Simple Y-axis ticks (e.g., min, mid, max) */}
            {[minL2Norm, (minL2Norm + maxL2Norm) / 2, maxL2Norm].map((val, i) => (
                <text key={`y-tick-${i}`} x={padding - 10} y={getY(val)} textAnchor="end" dominantBaseline="middle" className="text-xs">
                    {val.toFixed(2)}
                </text>
            ))}
            <text transform={`translate(${padding - 30}, ${svgHeight/2}) rotate(-90)`} textAnchor="middle" className="text-sm">L2 Norm</text>

            {/* Data Path */}
            <path
              d={chartData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(d.layer)} ${getY(d.l2norm)}`).join(' ')}
              fill="none"
              stroke="rgb(59, 130, 246)"
              strokeWidth="2"
            />
            {/* Data Points */}
            {chartData.map((d) => (
              <circle
                key={`point-${d.layer}`}
                cx={getX(d.layer)}
                cy={getY(d.layer)}
                r="4"
                fill="rgb(59, 130, 246)"
                className="cursor-pointer"
              >
                <title>Layer {d.layer}: {d.l2norm.toFixed(4)}</title>
              </circle>
            ))}
          </svg>
        </div>
      )}
      {chartData.length === 0 && selectedTokenIndex !== null && (
        <p className="text-gray-500">No hidden state data available for the selected token/layer combination.</p>
      )}
    </div>
  );
};

export default HiddenStateExplorer;
