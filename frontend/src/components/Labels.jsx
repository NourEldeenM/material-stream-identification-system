export default function Labels({ prediction, classifier }) {
	const commonStyle = 'shadow-lg scale-110 font-bold border-none';

	const classes = [
		{ name: 'Cardboard', activeClass: 'bg-amber-500 text-amber-900' + commonStyle },
		{ name: 'Glass', activeClass: 'bg-cyan-500 text-cyan-900' + commonStyle },
		{ name: 'Metal', activeClass: 'bg-gray-500 text-gray-900' + commonStyle },
		{ name: 'Paper', activeClass: 'bg-blue-500 text-blue-900' + commonStyle },
		{ name: 'Plastic', activeClass: 'bg-green-500 text-green-900' + commonStyle },
		{ name: 'Trash', activeClass: 'bg-red-500 text-red-900' + commonStyle },
		{ name: 'Unknown', activeClass: 'bg-gray-200 text-gray-900' + commonStyle },
	];

	return (
		<div className='w-full max-w-2xl'>
			<h2 className='text-xl font-semibold mb-4 text-center'>{classifier.toUpperCase()} Material Classification</h2>
			<div className='grid grid-cols-2 md:grid-cols-3 gap-4'>
				{classes.map((item) => {
					const isActive = prediction === item.name;
					return (
						<div
							key={item.name}
							className={`p-4 rounded-lg border-2 transition-all duration-300 ${
								isActive ? item.activeClass : 'bg-gray-100 text-gray-600 border-gray-300'
							}`}
						>
							<p className='text-center capitalize text-lg'>{item.name}</p>
						</div>
					);
				})}
			</div>
			{/* {prediction && (
				<div className='mt-6 text-center'>
					<p className='text-2xl font-bold uppercase tracking-wide'>
						Detected: <span className='text-blue-600'>{prediction}</span>
					</p>
				</div>
			)} */}
		</div>
	);
}
