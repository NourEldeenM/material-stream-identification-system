import { useCallback, useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const videoConstraints = {
	width: 720,
	height: 480,
	facingMode: 'user',
};

export default function WebcamComponent() {
	const webcamRef = useRef(null);
	const [result, setResult] = useState('unknown');

	async function captureFrame() {
		const image = webcamRef.current.getScreenshot();
		const formData = new FormData();
		formData.append('image', image);
		const response = await axios.post('http://localhost:8000/api/analyze', formData);
		console.log(response);
		setResult(response.data['prediction']);
	}

	useEffect(() => {
		const interval = setInterval(async () => {
			await captureFrame()
		}, 500)
		return () => {
			clearInterval(interval);
		}
	}, []);

	return (
		<>
			<Webcam
				audio={false}
				height={480}
				ref={webcamRef}
				screenshotFormat='image/jpeg'
				width={720}
				videoConstraints={videoConstraints}
			/>
			<p>{result}</p>
		</>
	);
}
