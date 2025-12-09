import { useState } from 'react'
import './App.css'
import CameraFeed from './components/CameraFeed'
import ClassificationResults from './components/ClassificationResults'
import Header from './components/Header'

function App() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [classificationResult, setClassificationResult] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleStartCamera = () => {
    setIsStreaming(true)
  }

  const handleStopCamera = () => {
    setIsStreaming(false)
    setClassificationResult(null)
  }

  const handleCapture = async () => {
    setIsProcessing(true)
    
    // Simulate API call to backend ML model
    // In production, this would be: await fetch('/api/classify', { method: 'POST', body: imageData })
    setTimeout(() => {
      // Mock classification result for demonstration
      const materials = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
      const randomMaterial = materials[Math.floor(Math.random() * materials.length)]
      const confidence = (Math.random() * 0.4 + 0.6).toFixed(2) // Random confidence between 0.6 and 1.0
      
      setClassificationResult({
        material: randomMaterial,
        confidence: parseFloat(confidence),
        timestamp: new Date().toLocaleTimeString()
      })
      setIsProcessing(false)
    }, 1500)
  }

  return (
    <div className="app">
      <Header />
      
      <main className="main-content">
        <div className="container">
          <div className="camera-section">
            <CameraFeed 
              isStreaming={isStreaming}
              onStartCamera={handleStartCamera}
              onStopCamera={handleStopCamera}
              onCapture={handleCapture}
              isProcessing={isProcessing}
            />
          </div>

          <div className="results-section">
            <ClassificationResults 
              result={classificationResult}
              isProcessing={isProcessing}
            />
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>Material Stream Identification System - Automated Waste Classification</p>
      </footer>
    </div>
  )
}

export default App
