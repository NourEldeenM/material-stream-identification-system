import { useCallback, useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const videoConstraints = {
	width: 720,
	height: 480,
	facingMode: 'user',
};

export default function WebcamComponent({classifier}) {
	const webcamRef = useRef(null);
	const [result, setResult] = useState('unknown');

	async function captureFrame() {
		const image = webcamRef.current.getScreenshot();
		const formData = new FormData();
		formData.append('image', image);
		const response = await axios.post(`http://localhost:8000/api/analyze/${classifier}`, formData);
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
		<div className='flex flex-col items-center gap-8'>
			<Webcam
				audio={false}
				height={480}
				ref={webcamRef}
				screenshotFormat='image/jpeg'
				width={720}
				videoConstraints={videoConstraints}
				className='rounded-2xl'
			/>
			<p className='text-2xl uppercase'>{result}</p>
		</div>
	);
}
