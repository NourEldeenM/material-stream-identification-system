import { useCallback, useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import Labels from './Labels';

const videoConstraints = {
	width: 720,
	height: 480,
	facingMode: 'user',
};

export default function WebcamComponent({ classifier }) {
	const webcamRef = useRef(null);
	const [result, setResult] = useState('Unknown');
	const [isProcessing, setIsProcessing] = useState(false);

	function resizeImage(base64Image, maxWidth = 320, maxHeight = 240, quality = 0.7) {
		return new Promise((resolve) => {
			const img = new Image();
			img.onload = () => {
				const canvas = document.createElement('canvas');
				let width = img.width;
				let height = img.height;

				// Calculate new dimensions while maintaining aspect ratio
				if (width > height) {
					if (width > maxWidth) {
						height = (height * maxWidth) / width;
						width = maxWidth;
					}
				} else {
					if (height > maxHeight) {
						width = (width * maxHeight) / height;
						height = maxHeight;
					}
				}

				canvas.width = width;
				canvas.height = height;
				const ctx = canvas.getContext('2d');
				ctx.drawImage(img, 0, 0, width, height);

				// Convert to base64 with specified quality
				resolve(canvas.toDataURL('image/jpeg', quality));
			};
			img.src = base64Image;
		});
	}

	const captureFrame = useCallback(async () => {
		if (isProcessing) return; // Skip if already processing

		setIsProcessing(true);
		try {
			const image = webcamRef.current.getScreenshot();
			const resizedImage = await resizeImage(image, 320, 240, 0.7);

			const formData = new FormData();
			formData.append('image', resizedImage);

			const response = await axios.post(`http://localhost:8000/api/analyze/${classifier}`, formData);
			console.log(response);
			setResult(response.data['prediction']);
		} catch (error) {
			console.error('Error processing frame:', error);
		} finally {
			setIsProcessing(false);
		}
	}, [isProcessing, classifier]);

	useEffect(() => {
		const interval = setInterval(async () => {
			await captureFrame();
		}, 1000);
		return () => {
			clearInterval(interval);
		};
	}, [captureFrame]);

	return (
		<div className='flex flex-col items-center gap-8 p-8'>
			<Webcam
				audio={false}
				ref={webcamRef}
				screenshotFormat='image/jpeg'
				videoConstraints={videoConstraints}
				className='rounded-2xl shadow-2xl border-4 border-gray-300'
			/>
			<Labels prediction={result} classifier={classifier}/>
		</div>
	);
}
