import WebcamComponent from './components/WebcamComponent';

export default function App() {
	return (
		<main className='flex py-20 justify-center m-auto gap-20'>
			<WebcamComponent classifier='knn' />
			<WebcamComponent classifier='svm' />
		</main>
	);
}
