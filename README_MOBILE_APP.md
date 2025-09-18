# Tire OCR Mobile App

A React Native mobile application that can analyze tire images and extract tire information using machine learning models. The app includes a camera interface for taking photos and displays extracted tire information including manufacturer, model, size, and DOT code.

## Features

- 📸 Camera integration for taking tire photos
- 🖼️ Photo library access for selecting existing images
- 🤖 Machine learning-based tire analysis
- 📊 Comprehensive tire information display
- 🏭 Plant information lookup
- 📱 Native iOS app that can be installed on iPhone

## Architecture

The mobile app consists of two main components:

1. **React Native Mobile App** (`mobile_app/`)
   - Cross-platform mobile interface
   - Camera and image picker integration
   - Results display and user interaction

2. **FastAPI Backend** (`backend/`)
   - REST API for tire analysis
   - Integration with existing ML models
   - Docker containerization for easy deployment

## Prerequisites

### For Mobile App Development
- Node.js (v16 or higher)
- React Native CLI
- Xcode (for iOS development)
- iOS Simulator or physical iPhone
- CocoaPods (for iOS dependencies)

### For Backend
- Python 3.9+
- Docker and Docker Compose
- Google Cloud Vision API credentials

## Setup Instructions

### 1. Backend Setup

#### Option A: Using Docker (Recommended)

1. **Start the backend service:**
   ```bash
   cd /Users/cathalodwyer/Desktop/Masters/Thesis/Tyre_OCR
   docker-compose up --build
   ```

   The API will be available at `http://localhost:8000`

#### Option B: Manual Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Set up Google Cloud credentials:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/ocrtyre-9369d891cdc1.json"
   ```

3. **Run the backend:**
   ```bash
   python main.py
   ```

### 2. Mobile App Setup

1. **Install Node.js dependencies:**
   ```bash
   cd mobile_app
   npm install
   ```

2. **Install iOS dependencies:**
   ```bash
   cd ios
   pod install
   cd ..
   ```

3. **Update API URL (if needed):**
   Edit `mobile_app/services/TireOCRService.ts` and update the `API_BASE_URL` if your backend is running on a different address.

### 3. Running the Mobile App

#### iOS Simulator
```bash
cd mobile_app
npx react-native run-ios
```

#### Physical iPhone
1. Open `mobile_app/ios/TireOCRApp.xcworkspace` in Xcode
2. Connect your iPhone via USB
3. Select your device in Xcode
4. Click the "Run" button

## Building for Production

### iOS App Store Build

1. **Open Xcode:**
   ```bash
   cd mobile_app/ios
   open TireOCRApp.xcworkspace
   ```

2. **Configure signing:**
   - Select your development team
   - Set bundle identifier
   - Configure provisioning profiles

3. **Build for release:**
   - Select "Any iOS Device" as target
   - Product → Archive
   - Follow the archive process to create an IPA file

### Alternative: Using React Native CLI

```bash
cd mobile_app
npx react-native run-ios --configuration Release
```

## Deployment Options

### 1. Local Development
- Backend runs on `localhost:8000`
- Mobile app connects to local backend
- Good for development and testing

### 2. Cloud Deployment
- Deploy backend to cloud service (AWS, Google Cloud, Heroku)
- Update mobile app API URL to point to cloud backend
- Mobile app can be distributed via TestFlight or App Store

### 3. Self-Hosted Server
- Deploy backend to your own server
- Update mobile app API URL
- Distribute mobile app via TestFlight or direct installation

## Configuration

### Backend Configuration
- Update `backend/main.py` to modify API behavior
- Environment variables can be set in `docker-compose.yml`
- Model paths are configured in the main script

### Mobile App Configuration
- Update `mobile_app/services/TireOCRService.ts` for API URL
- Modify `mobile_app/App.tsx` for UI changes
- Update `mobile_app/ios/TireOCRApp/Info.plist` for permissions

## API Endpoints

### Health Check
```
GET /health
```
Returns server status.

### Analyze Tire
```
POST /analyze-tire
Content-Type: multipart/form-data

Body: image file
```
Returns tire information in JSON format.

## Troubleshooting

### Common Issues

1. **Camera permission denied:**
   - Check Info.plist permissions
   - Ensure camera permission is granted in device settings

2. **Backend connection failed:**
   - Verify backend is running
   - Check API URL in TireOCRService.ts
   - Ensure network connectivity

3. **Build errors:**
   - Clean and rebuild: `cd ios && pod install && cd .. && npx react-native run-ios`
   - Clear Metro cache: `npx react-native start --reset-cache`

4. **Model loading errors:**
   - Ensure all model files are present
   - Check file permissions
   - Verify Google Cloud credentials

### Debug Mode
Enable debug logging by setting environment variables:
```bash
export DEBUG=1
```

## File Structure

```
mobile_app/
├── App.tsx                 # Main app component
├── services/
│   └── TireOCRService.ts   # API service
├── ios/                    # iOS-specific files
│   ├── Podfile
│   └── TireOCRApp/
│       └── Info.plist
├── package.json
└── README.md

backend/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
└── README.md

Dockerfile                  # Docker configuration
docker-compose.yml         # Docker Compose setup
```

## Security Considerations

1. **API Security:**
   - Add authentication if needed
   - Use HTTPS in production
   - Implement rate limiting

2. **Image Privacy:**
   - Images are processed server-side
   - Consider implementing image deletion after processing
   - Add user consent for data processing

3. **Network Security:**
   - Use secure connections
   - Validate all inputs
   - Implement proper error handling

## Performance Optimization

1. **Image Compression:**
   - Images are compressed before sending to API
   - Adjust quality settings in image picker

2. **Caching:**
   - Consider implementing result caching
   - Cache model predictions for similar images

3. **Background Processing:**
   - API calls are asynchronous
   - Loading indicators provide user feedback

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Ensure all dependencies are properly installed
4. Verify network connectivity and API availability

## License

This project is part of your Masters Thesis work. Please ensure proper attribution and licensing as required by your institution.
