# Tire OCR Mobile Web App

A Progressive Web App (PWA) that works on your iPhone to analyze tire images using your existing ML models.

## 🚀 Quick Start

### Step 1: Start the Backend API
```bash
# In Terminal 1 - Start your backend
cd /Users/cathalodwyer/Desktop/Masters/Thesis/Tyre_OCR/backend
python main.py
```

### Step 2: Start the Web App Server
```bash
# In Terminal 2 - Start the web app
cd /Users/cathalodwyer/Desktop/Masters/Thesis/Tyre_OCR/mobile_web_app
python start_server.py
```

### Step 3: Use on Your iPhone
1. **Make sure both devices are on the same WiFi network**
2. **Open Safari on your iPhone**
3. **Go to the URL shown in Terminal 2** (e.g., `http://192.168.1.100:3000`)
4. **Add to Home Screen** for app-like experience

## 📱 Features

- **Camera Integration** - Take photos directly in the browser
- **Drag & Drop** - Upload images from your photo library
- **Real-time Analysis** - Sends images to your backend API
- **Professional UI** - Clean, mobile-optimized interface
- **Tire Information Display** - Shows manufacturer, model, size, DOT code, and plant info
- **PWA Support** - Can be installed like a native app

## 🔧 How It Works

1. **Take Photo** - Use your iPhone's camera to photograph a tire
2. **Upload Image** - Image is sent to your backend API (localhost:8000)
3. **AI Processing** - Your existing ML models analyze the image:
   - YOLO detects the tire
   - Image enhancement with `warpPolar`
   - Google Cloud Vision OCR extracts text
   - ML models extract tire information
4. **Display Results** - App shows extracted information in a clean format

## 📊 Extracted Information

- **Manufacturer** - Tire brand (e.g., Michelin, Bridgestone)
- **Model** - Tire model name
- **Size** - Tire dimensions (e.g., 205/55R16)
- **Load/Speed Rating** - Load index and speed symbol
- **DOT Code** - Manufacturing date and plant information
- **Plant Details** - Factory name and country

## 🌐 Network Configuration

### For Local Development
- Backend runs on `localhost:8000`
- Web app runs on `localhost:3000` (or your local IP)
- Both devices must be on the same WiFi network

### For Production
- Deploy backend to cloud service
- Update `API_BASE_URL` in `index.html`
- Deploy web app to any web hosting service

## 📱 Installing as App on iPhone

1. **Open the web app** in Safari
2. **Tap the Share button** (square with arrow)
3. **Select "Add to Home Screen"**
4. **Tap "Add"**
5. **App icon appears** on your home screen
6. **Tap to open** like a native app

## 🛠️ Customization

### Change API URL
Edit `index.html` and update:
```javascript
const API_BASE_URL = 'http://YOUR_SERVER_IP:8000';
```

### Modify UI
- Edit the CSS in `index.html` for styling
- Modify the HTML structure for layout changes
- Update JavaScript for functionality changes

## 🔍 Troubleshooting

### "Cannot connect to server"
- Check that backend is running on port 8000
- Verify both devices are on same WiFi network
- Check firewall settings

### "Analysis failed"
- Ensure Google Cloud credentials are set up
- Check backend logs for errors
- Verify image quality and tire visibility

### "Page not loading"
- Check that web server is running on port 3000
- Try accessing from computer first
- Check network connectivity

## 📁 File Structure

```
mobile_web_app/
├── index.html          # Main web app file
├── start_server.py     # Python HTTP server
└── README.md          # This file
```

## 🎯 Benefits of Web App vs Native App

### Advantages:
- ✅ **No App Store** - Install directly from browser
- ✅ **Cross-platform** - Works on iPhone, Android, desktop
- ✅ **Easy updates** - Just refresh the page
- ✅ **No compilation** - Pure HTML/CSS/JavaScript
- ✅ **Instant deployment** - Just upload files

### Limitations:
- ❌ **Limited camera access** - Depends on browser permissions
- ❌ **No offline mode** - Requires internet connection
- ❌ **Browser dependent** - Some features vary by browser

## 🚀 Next Steps

1. **Test the app** with real tire images
2. **Customize the UI** to your preferences
3. **Deploy to production** if needed
4. **Add more features** like image history, sharing, etc.

## 📞 Support

If you encounter issues:
1. Check that both servers are running
2. Verify network connectivity
3. Check browser console for errors
4. Review backend logs for API issues

The web app provides a simple, effective way to use your tire OCR system on your iPhone without the complexity of native app development!
