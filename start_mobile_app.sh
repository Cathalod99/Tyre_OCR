#!/bin/bash

# Tire OCR Mobile App Startup Script
# This script starts the backend API and provides instructions for the mobile app

echo "🚀 Starting Tire OCR Mobile App System"
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if Google Cloud credentials exist
if [ ! -f "ocrtyre-9369d891cdc1.json" ]; then
    echo "⚠️  Warning: Google Cloud credentials file not found."
    echo "   Please ensure ocrtyre-9369d891cdc1.json is in the project root."
    echo "   The app may not work without proper OCR credentials."
fi

# Start the backend API
echo "🔧 Starting backend API..."
docker-compose up --build -d

# Wait for the API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 10

# Check if API is responding
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend API is running at http://localhost:8000"
else
    echo "❌ Backend API failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "📱 Mobile App Setup Instructions:"
echo "================================"
echo "1. Install Node.js (if not already installed):"
echo "   brew install node"
echo ""
echo "2. Install React Native CLI:"
echo "   npm install -g react-native-cli"
echo ""
echo "3. Install mobile app dependencies:"
echo "   cd mobile_app"
echo "   npm install"
echo ""
echo "4. Install iOS dependencies:"
echo "   cd ios"
echo "   pod install"
echo "   cd .."
echo ""
echo "5. Run the mobile app:"
echo "   npx react-native run-ios"
echo ""
echo "🔗 API Endpoints:"
echo "   Health Check: http://localhost:8000/health"
echo "   Analyze Tire: http://localhost:8000/analyze-tire"
echo ""
echo "📖 For detailed instructions, see README_MOBILE_APP.md"
echo ""
echo "🛑 To stop the backend: docker-compose down"
