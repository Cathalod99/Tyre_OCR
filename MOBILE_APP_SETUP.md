# Tire OCR Mobile App - Complete Setup Guide

This guide will help you set up and deploy the Tire OCR mobile application that can be installed on your iPhone.

## 🎯 What You'll Get

- **iPhone App**: A native iOS app that you can install on your iPhone
- **Camera Integration**: Take photos of tires directly in the app
- **AI Analysis**: Extract tire information using your existing ML models
- **Professional UI**: Clean, modern interface for displaying results

## 📋 Prerequisites

### Required Software
1. **Xcode** (from Mac App Store) - for building iOS apps
2. **Node.js** (v16+) - for React Native
3. **Docker Desktop** - for backend API
4. **CocoaPods** - for iOS dependencies

### Required Accounts
1. **Apple Developer Account** (free for personal use)
2. **Google Cloud Account** (for OCR API)

## 🚀 Quick Start (5 minutes)

### Step 1: Start the Backend
```bash
# Navigate to your project
cd /Users/cathalodwyer/Desktop/Masters/Thesis/Tyre_OCR

# Start the backend API
./start_mobile_app.sh
```

### Step 2: Install Mobile App Dependencies
```bash
# Install Node.js dependencies
cd mobile_app
npm install

# Install iOS dependencies
cd ios
pod install
cd ..
```

### Step 3: Run the App
```bash
# Run on iOS Simulator
npx react-native run-ios

# OR run on your iPhone (connected via USB)
# Open Xcode and run the project
```

## 📱 Installing on Your iPhone

### Method 1: Xcode (Recommended for Development)

1. **Connect your iPhone** via USB cable
2. **Open Xcode**:
   ```bash
   cd mobile_app/ios
   open TireOCRApp.xcworkspace
   ```
3. **Select your device** in Xcode
4. **Configure signing**:
   - Select your Apple ID
   - Choose "Automatically manage signing"
   - Select your development team
5. **Click Run** (▶️ button)

### Method 2: TestFlight (For Distribution)

1. **Build for Archive**:
   - In Xcode: Product → Archive
   - Wait for build to complete
2. **Upload to App Store Connect**:
   - Click "Distribute App"
   - Choose "App Store Connect"
   - Follow the upload process
3. **Add to TestFlight**:
   - Go to App Store Connect
   - Add internal testers
   - Install TestFlight app on your iPhone
   - Install the app from TestFlight

## 🔧 Detailed Setup

### Backend API Setup

The backend runs your existing tire OCR models via a REST API.

#### Option 1: Docker (Recommended)
```bash
# Start with Docker
docker-compose up --build

# Check if running
curl http://localhost:8000/health
```

#### Option 2: Manual Setup
```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"

# Run the API
python main.py
```

### Mobile App Setup

#### 1. Install Prerequisites
```bash
# Install Node.js (if not installed)
brew install node

# Install React Native CLI
npm install -g react-native-cli

# Install CocoaPods (if not installed)
sudo gem install cocoapods
```

#### 2. Install Dependencies
```bash
cd mobile_app

# Install Node.js packages
npm install

# Install iOS dependencies
cd ios
pod install
cd ..
```

#### 3. Configure API URL
Edit `mobile_app/services/TireOCRService.ts`:
```typescript
const API_BASE_URL = 'http://YOUR_SERVER_IP:8000';
```

For local development, use your computer's IP address instead of localhost.

## 🌐 Network Configuration

### For Local Development
1. **Find your computer's IP**:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```
2. **Update mobile app** to use your IP:
   ```typescript
   const API_BASE_URL = 'http://192.168.1.100:8000'; // Replace with your IP
   ```
3. **Ensure both devices are on same WiFi network**

### For Production Deployment
1. **Deploy backend to cloud** (AWS, Google Cloud, Heroku)
2. **Update API URL** in mobile app
3. **Build and distribute** the app

## 📱 App Features

### Camera Integration
- **Take Photos**: Direct camera access for tire photos
- **Photo Library**: Select existing images
- **Image Quality**: Automatic compression for optimal processing

### Tire Analysis
- **Manufacturer Detection**: Identifies tire brand
- **Model Recognition**: Extracts tire model
- **Size Information**: Tire dimensions (e.g., 205/55R16)
- **DOT Code**: Manufacturing date and plant information
- **Plant Details**: Factory name and country

### User Interface
- **Modern Design**: Clean, professional appearance
- **Loading Indicators**: Visual feedback during processing
- **Error Handling**: Clear error messages
- **Results Display**: Organized information presentation

## 🔍 Testing the App

### Test the Backend
```bash
# Run the test script
python test_api.py
```

### Test the Mobile App
1. **Take a photo** of a tire
2. **Tap "Analyze Tire"**
3. **Wait for results** (usually 5-15 seconds)
4. **Review extracted information**

## 🛠️ Troubleshooting

### Common Issues

#### "Cannot connect to server"
- **Check backend is running**: `curl http://localhost:8000/health`
- **Verify IP address** in TireOCRService.ts
- **Ensure same WiFi network**

#### "Camera permission denied"
- **Check Info.plist** has camera permissions
- **Grant permission** in iPhone Settings
- **Restart the app**

#### "Build failed"
- **Clean build**: `cd ios && pod install && cd ..`
- **Clear cache**: `npx react-native start --reset-cache`
- **Check Xcode logs** for specific errors

#### "No tire detected"
- **Ensure tire is clearly visible** in photo
- **Try different angles** or lighting
- **Check if tire has visible text**

### Debug Mode
Enable detailed logging:
```bash
# Backend
export DEBUG=1
python backend/main.py

# Mobile app
npx react-native run-ios --verbose
```

## 📦 Building for Distribution

### Development Build
```bash
npx react-native run-ios --configuration Debug
```

### Release Build
```bash
npx react-native run-ios --configuration Release
```

### App Store Build
1. **Open Xcode**: `open mobile_app/ios/TireOCRApp.xcworkspace`
2. **Select "Any iOS Device"**
3. **Product → Archive**
4. **Follow distribution wizard**

## 🔐 Security Considerations

### API Security
- **Use HTTPS** in production
- **Add authentication** if needed
- **Implement rate limiting**

### Data Privacy
- **Images processed server-side**
- **No permanent storage** of images
- **User consent** for data processing

## 📊 Performance Tips

### Image Optimization
- **Compress images** before sending
- **Use appropriate resolution** (1024x1024 max)
- **Consider image quality** vs processing time

### Network Optimization
- **Use WiFi** for better performance
- **Implement retry logic** for failed requests
- **Add offline capabilities** if needed

## 🆘 Getting Help

### Check Logs
```bash
# Backend logs
docker-compose logs -f

# Mobile app logs
npx react-native log-ios
```

### Common Solutions
1. **Restart everything**: Backend, mobile app, simulator
2. **Clear caches**: Metro, Xcode, CocoaPods
3. **Check permissions**: Camera, network, storage
4. **Verify network**: Same WiFi, correct IP address

### Support Resources
- **React Native Docs**: https://reactnative.dev/
- **iOS Development**: https://developer.apple.com/
- **FastAPI Docs**: https://fastapi.tiangolo.com/

## 🎉 Success!

Once everything is working, you'll have:
- ✅ A native iOS app on your iPhone
- ✅ Camera integration for tire photos
- ✅ AI-powered tire analysis
- ✅ Professional results display
- ✅ Easy installation and updates

The app will extract tire information including manufacturer, model, size, and DOT code, just like your desktop version but in a mobile-friendly format!
