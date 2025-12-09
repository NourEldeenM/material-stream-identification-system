import { useRef, useEffect, useState } from 'react'
import './CameraFeed.css'

function CameraFeed({ isStreaming, onStartCamera, onStopCamera, onCapture, isProcessing }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [error, setError] = useState(null)
  const streamRef = useRef(null)

  useEffect(() => {
    let mounted = true

    const initCamera = async () => {
      if (!isStreaming) {
        // Stop camera
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop())
          streamRef.current = null
        }
        if (videoRef.current) {
          videoRef.current.srcObject = null
        }
        return
      }

      // Start camera
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'environment'
          } 
        })
        
        if (mounted && videoRef.current) {
          videoRef.current.srcObject = stream
          streamRef.current = stream
          setError(null)
        }
      } catch (err) {
        if (mounted) {
          console.error('Error accessing camera:', err)
          setError('Camera access denied. Please allow camera permissions.')
          onStopCamera()
        }
      }
    }

    void initCamera()

    return () => {
      mounted = false
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
    }
  }, [isStreaming, onStopCamera])

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    const context = canvas.getContext('2d')
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    
    // Convert canvas to blob for sending to backend
    canvas.toBlob((blob) => {
      if (blob) {
        onCapture(blob)
      }
    }, 'image/jpeg', 0.95)
  }

  return (
    <div className="camera-feed">
      <h2>📹 Live Camera Feed</h2>
      
      <div className="video-container">
        {!isStreaming && !error && (
          <div className="camera-placeholder">
            <div className="camera-icon">📷</div>
            <p>Click "Start Camera" to begin</p>
          </div>
        )}
        
        {error && (
          <div className="camera-error">
            <div className="error-icon">⚠️</div>
            <p>{error}</p>
          </div>
        )}
        
        <video 
          ref={videoRef}
          autoPlay
          playsInline
          className={isStreaming ? 'video-active' : 'video-hidden'}
        />
        
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      <div className="camera-controls">
        {!isStreaming ? (
          <button 
            className="btn btn-start"
            onClick={onStartCamera}
          >
            🎥 Start Camera
          </button>
        ) : (
          <>
            <button 
              className="btn btn-capture"
              onClick={captureImage}
              disabled={isProcessing}
            >
              {isProcessing ? '⏳ Processing...' : '📸 Capture & Identify'}
            </button>
            <button 
              className="btn btn-stop"
              onClick={onStopCamera}
              disabled={isProcessing}
            >
              ⏹️ Stop Camera
            </button>
          </>
        )}
      </div>

      <div className="camera-info">
        <p>💡 <strong>Tip:</strong> Position the waste material clearly in the frame for best results</p>
      </div>
    </div>
  )
}

export default CameraFeed
