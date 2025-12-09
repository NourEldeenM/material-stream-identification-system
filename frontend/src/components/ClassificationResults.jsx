import './ClassificationResults.css'

function ClassificationResults({ result, isProcessing }) {
  const getMaterialIcon = (material) => {
    const icons = {
      glass: '🥃',
      paper: '📄',
      cardboard: '📦',
      plastic: '🥤',
      metal: '🔩',
      trash: '🗑️',
      unknown: '❓'
    }
    return icons[material] || '❓'
  }

  const getMaterialColor = (material) => {
    const colors = {
      glass: '#4FC3F7',
      paper: '#FFEB3B',
      cardboard: '#D7CCC8',
      plastic: '#FF7043',
      metal: '#9E9E9E',
      trash: '#66BB6A',
      unknown: '#9C27B0'
    }
    return colors[material] || '#9C27B0'
  }

  const getDisposalGuidance = (material) => {
    const guidance = {
      glass: 'Rinse and place in glass recycling bin. Remove caps and lids.',
      paper: 'Keep dry and clean. Place in paper recycling bin.',
      cardboard: 'Flatten boxes. Remove tape and labels. Place in cardboard recycling.',
      plastic: 'Check recycling number. Rinse containers. Place in plastic recycling bin.',
      metal: 'Rinse cans and containers. Place in metal recycling bin.',
      trash: 'Non-recyclable waste. Dispose in general waste bin.',
      unknown: 'Material not recognized. Please try again or dispose as general waste.'
    }
    return guidance[material] || 'Classification uncertain.'
  }

  return (
    <div className="classification-results">
      <h2>🎯 Classification Results</h2>

      {!result && !isProcessing && (
        <div className="results-placeholder">
          <div className="placeholder-icon">🔍</div>
          <p>Waiting for classification...</p>
          <p className="placeholder-hint">Capture an image to identify the waste material</p>
        </div>
      )}

      {isProcessing && (
        <div className="results-loading">
          <div className="loading-spinner"></div>
          <p>Analyzing waste material...</p>
          <p className="loading-hint">Processing with AI model</p>
        </div>
      )}

      {result && !isProcessing && (
        <div className="results-content">
          <div 
            className="material-result"
            style={{ borderColor: getMaterialColor(result.material) }}
          >
            <div 
              className="material-icon"
              style={{ background: getMaterialColor(result.material) }}
            >
              {getMaterialIcon(result.material)}
            </div>
            <div className="material-info">
              <h3 className="material-name">{result.material.toUpperCase()}</h3>
              <div className="confidence-bar">
                <div className="confidence-label">
                  <span>Confidence</span>
                  <span className="confidence-value">{(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ 
                      width: `${result.confidence * 100}%`,
                      background: getMaterialColor(result.material)
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="disposal-guidance">
            <h4>♻️ Disposal Guidance</h4>
            <p>{getDisposalGuidance(result.material)}</p>
          </div>

          <div className="result-metadata">
            <div className="metadata-item">
              <span className="metadata-label">Timestamp:</span>
              <span className="metadata-value">{result.timestamp}</span>
            </div>
          </div>

          <div className="classification-history-note">
            <p>💾 Result saved to classification history</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ClassificationResults
