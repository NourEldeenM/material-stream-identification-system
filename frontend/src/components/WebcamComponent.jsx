import { useCallback, useRef, useState } from 'react';
import Webcam from 'react-webcam';

const videoConstraints = {
	width: 720,
	height: 480,
	facingMode: 'user',
};

export default function WebcamComponent() {
	const webcamRef = useRef(null);
	const [imgSrc, setImgSrc] = useState(null);

	const capture = useCallback(() => {
		const imageSrc = webcamRef.current.getScreenshot();
		setImgSrc(imageSrc);
	}, [webcamRef]);

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
			<button onClick={capture}>Capture Photo</button>
			{imgSrc && <img src={imgSrc} />}
		</>
	);
}
