# Material Stream Identification System

An Automated Material Stream Identification (MSI) System that efficiently automates sorting of post-consumer waste using fundamental Machine Learning (ML) techniques therefore achieving circular economy goals.

## 🎯 Project Overview

This system uses AI-powered computer vision to identify and classify waste materials in real-time, supporting the goals of circular economy through automated waste sorting. The system can classify materials into seven categories:

- 🥃 **Glass**
- 📄 **Paper**
- 📦 **Cardboard**
- 🥤 **Plastic**
- 🔩 **Metal**
- 🗑️ **Trash**
- ❓ **Unknown** (when classification confidence is low)

## 📁 Project Structure

```
material-stream-identification-system/
├── frontend/                 # React + Vite web application
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── Header.jsx           # Application header
│   │   │   ├── CameraFeed.jsx       # Live camera feed component
│   │   │   └── ClassificationResults.jsx  # Results display
│   │   ├── App.jsx          # Main application component
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── public/              # Static assets
│   ├── package.json         # Frontend dependencies
│   └── vite.config.js       # Vite configuration
├── backend/                 # Backend ML service (to be implemented)
│   └── [Future: ML model integration]
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v16 or higher)
- **npm** or **yarn**
- A modern web browser with camera support
- Camera permissions enabled

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NourEldeenM/material-stream-identification-system.git
   cd material-stream-identification-system
   ```

2. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5173`

### Building for Production

```bash
cd frontend
npm run build
```

The production-ready files will be in the `frontend/dist` directory.

## 🎨 Features

### Current Features (Frontend)

- ✅ **Live Camera Feed**: Real-time camera streaming with support for mobile and desktop
- ✅ **Capture & Classify**: Take snapshots for material identification
- ✅ **Visual Results**: Beautiful, color-coded classification results
- ✅ **Confidence Scores**: Display confidence levels for each classification
- ✅ **Disposal Guidance**: Helpful tips on how to properly dispose of each material type
- ✅ **Responsive Design**: Works on desktop, tablet, and mobile devices
- ✅ **Themed Interface**: Clean, modern UI that reflects the environmental purpose

### Planned Features (Backend - To Be Implemented)

- 🔄 **ML Model Integration**: Connect to SVM/k-NN classifiers
- 🔄 **Feature Extraction**: Process 2D/3D images into 1D feature vectors
- 🔄 **Data Augmentation**: Enhance training dataset
- 🔄 **Real-time Processing**: Stream processing for continuous classification
- 🔄 **Classification History**: Store and retrieve past classifications
- 🔄 **Model Performance Metrics**: Track accuracy and validation metrics

## 📖 Documentation

[Main Project Documentation](https://docs.google.com/document/d/1-nBlcD_kVZX6oGrfCbp0lg60r7hUdXLqMtu3Tv3RlrM/edit?usp=sharing)

## 🎯 Functional Requirements

1. Data Augmentation and Feature Extraction
2. Classifier Implementation (SVM, k-NN)
3. Interface with Live Camera (real-time) processing

## 📊 Non-Functional Requirements

1. Apply Data Augmentation techniques to increase training dataset by 30%
2. Try to achieve minimum validation accuracy of 0.85 across the six primary classes
3. Raw 2D and 3D images should be converted into 1D numerical feature vector
4. The SVM classifier must be designed to accept the extracted feature vector as input
5. The k-NN classifier must be designed to accept the extracted feature vector as input, and the weighting scheme (e.g., uniform, distance-based) for the classifier
6. There should be an `unknown` class that is selected if the model couldn't classify the data properly

## 🛠️ Technology Stack

### Frontend
- **React 19**: Modern UI framework
- **Vite**: Fast build tool and dev server
- **JavaScript (ES6+)**: Programming language
- **CSS3**: Styling with modern features
- **WebRTC**: Camera access via getUserMedia API

### Backend (To Be Implemented)
- **Python**: Primary programming language
- **Scikit-learn**: For SVM and k-NN classifiers
- **OpenCV**: Image processing
- **FastAPI/Flask**: REST API framework
- **NumPy**: Numerical computations

## 🎨 Design Philosophy

The interface is designed with the following principles:

- **Environmental Theme**: Blue and green gradients representing water and nature
- **Color-Coded Materials**: Each material type has a distinct color for quick recognition
- **Clear Hierarchy**: Important information is prominently displayed
- **Accessibility**: High contrast and readable fonts
- **Responsive**: Works seamlessly across all device sizes

## 🔌 API Integration (Future)

The frontend is prepared to integrate with a backend ML service. The expected API endpoint:

```javascript
POST /api/classify
Content-Type: multipart/form-data

Request Body:
{
  image: <blob>  // Captured image data
}

Response:
{
  material: "plastic",
  confidence: 0.92,
  timestamp: "2024-01-01T12:00:00Z"
}
```

## 📝 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Structure

The application follows a component-based architecture:

- **App.jsx**: Main application logic and state management
- **Header.jsx**: Branding and material type indicators
- **CameraFeed.jsx**: Camera control and image capture
- **ClassificationResults.jsx**: Display classification results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of an academic assignment focused on machine learning and computer vision applications for environmental sustainability.

## 👥 Team

- [Project Team Members]

## 🙏 Acknowledgments

- Thanks to all contributors who help make waste management more efficient
- Special thanks to the machine learning community for open-source tools and resources

## 📞 Support

For questions or issues, please open an issue in the GitHub repository.

---

**Note**: This project is currently in development. The frontend interface is complete, while the backend ML integration is planned for future implementation.
