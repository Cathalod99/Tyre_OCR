#!/bin/bash

# Tire OCR App Deployment Script
# This script helps you deploy your app to the cloud

echo "🚀 Tire OCR App Deployment Helper"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "mobile_web_app/index.html" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📋 Deployment Options:"
echo "1. Deploy Web App (Recommended - 15 minutes)"
echo "2. Deploy Native Mobile App (Advanced - 2+ hours)"
echo "3. View deployment guide"
echo ""

read -p "Choose an option (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🌐 Web App Deployment"
        echo "===================="
        echo ""
        echo "Step 1: Deploy Backend to Railway"
        echo "1. Go to https://railway.app"
        echo "2. Sign up with GitHub"
        echo "3. Click 'New Project' → 'Deploy from GitHub repo'"
        echo "4. Select your Tyre_OCR repository"
        echo "5. Railway will auto-detect the Dockerfile in backend/"
        echo "6. Add your Google Cloud credentials file"
        echo "7. Click 'Deploy'"
        echo ""
        echo "Step 2: Deploy Web App to Netlify"
        echo "1. Go to https://netlify.com"
        echo "2. Sign up with GitHub"
        echo "3. Click 'New site from Git'"
        echo "4. Select your Tyre_OCR repository"
        echo "5. Set base directory to 'mobile_web_app'"
        echo "6. Click 'Deploy site'"
        echo ""
        echo "Step 3: Update API URL"
        echo "1. Get your Railway backend URL"
        echo "2. Edit mobile_web_app/index.html"
        echo "3. Replace 'https://your-backend.railway.app' with your actual URL"
        echo "4. Redeploy on Netlify"
        echo ""
        echo "📖 For detailed instructions, see DEPLOYMENT_GUIDE.md"
        ;;
    2)
        echo ""
        echo "📱 Native Mobile App Deployment"
        echo "=============================="
        echo ""
        echo "iOS App Store:"
        echo "1. Open mobile_app/ios/TireOCRApp.xcworkspace in Xcode"
        echo "2. Update API URL in TireOCRService.ts"
        echo "3. Configure Apple Developer account"
        echo "4. Archive and upload to App Store Connect"
        echo ""
        echo "Google Play Store:"
        echo "1. Open mobile_app/android/ in Android Studio"
        echo "2. Update API URL in TireOCRService.ts"
        echo "3. Generate signed APK"
        echo "4. Upload to Google Play Console"
        echo ""
        echo "📖 For detailed instructions, see DEPLOYMENT_GUIDE.md"
        ;;
    3)
        echo ""
        echo "📖 Opening deployment guide..."
        if command -v open &> /dev/null; then
            open DEPLOYMENT_GUIDE.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open DEPLOYMENT_GUIDE.md
        else
            echo "Please open DEPLOYMENT_GUIDE.md in your text editor"
        fi
        ;;
    *)
        echo "❌ Invalid option. Please choose 1, 2, or 3."
        ;;
esac

echo ""
echo "✅ Deployment helper complete!"
echo "Need help? Check DEPLOYMENT_GUIDE.md for detailed instructions."
