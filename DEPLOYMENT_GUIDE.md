# 🚀 Tire OCR App Deployment Guide

This guide will help you deploy your Tire OCR app to the cloud so others can use it.

## 📋 Prerequisites

- GitHub account (✅ You have this)
- Railway account (free tier available)
- Netlify account (free tier available)
- Google Cloud credentials file (`ocrtyre.json`)

## 🎯 Deployment Options

### Option 1: Web App (Recommended - 15 minutes)
Deploy the Progressive Web App that works on any phone through the browser.

### Option 2: Native Mobile App (Advanced - 2+ hours)
Deploy to Apple App Store and Google Play Store.

---

## 🌐 Option 1: Deploy Web App (Easiest)

### Step 1: Deploy Backend API to Railway

1. **Go to [Railway.app](https://railway.app)**
2. **Sign up with GitHub**
3. **Click "New Project" → "Deploy from GitHub repo"**
4. **Select your `Tyre_OCR` repository**
5. **Railway will auto-detect the Dockerfile in `backend/`**
6. **Add Environment Variables:**
   - `GOOGLE_APPLICATION_CREDENTIALS`: Upload your `ocrtyre.json` file
   - `PORT`: Railway will set this automatically
7. **Click "Deploy"**
8. **Wait 5-10 minutes for deployment**
9. **Copy your Railway URL** (e.g., `https://your-app.railway.app`)

### Step 2: Deploy Web App to Netlify

1. **Go to [Netlify.com](https://netlify.com)**
2. **Sign up with GitHub**
3. **Click "New site from Git"**
4. **Select your `Tyre_OCR` repository**
5. **Set build settings:**
   - **Base directory:** `mobile_web_app`
   - **Build command:** (leave empty)
   - **Publish directory:** `mobile_web_app`
6. **Click "Deploy site"**
7. **Wait 2-3 minutes for deployment**
8. **Copy your Netlify URL** (e.g., `https://your-app.netlify.app`)

### Step 3: Update Web App to Use Your Backend

1. **In Netlify dashboard, go to your site**
2. **Click "Site settings" → "Environment variables"**
3. **Add new variable:**
   - **Key:** `REACT_APP_API_URL`
   - **Value:** Your Railway backend URL (e.g., `https://your-app.railway.app`)
4. **Redeploy the site**

### Step 4: Update the Web App Code

Edit `mobile_web_app/index.html` and update the API URL:

```javascript
// Change this line:
const API_BASE_URL = 'http://localhost:8000';

// To this (replace with your Railway URL):
const API_BASE_URL = 'https://your-app.railway.app';
```

### Step 5: Test Your Deployed App

1. **Open your Netlify URL on your phone**
2. **Take a photo of a tire**
3. **Check if it processes correctly**

---

## 📱 Option 2: Deploy Native Mobile App (Advanced)

### For iOS (Apple App Store)

1. **Install Xcode** (Mac only)
2. **Open `mobile_app/ios/TireOCRApp.xcworkspace`**
3. **Update API URL in `TireOCRService.ts`**
4. **Configure Apple Developer account**
5. **Archive and upload to App Store Connect**

### For Android (Google Play Store)

1. **Install Android Studio**
2. **Open `mobile_app/android/`**
3. **Update API URL in `TireOCRService.ts`**
4. **Generate signed APK**
5. **Upload to Google Play Console**

---

## 🔧 Configuration Details

### Backend Environment Variables

```bash
GOOGLE_APPLICATION_CREDENTIALS=/app/ocrtyre.json
PORT=8000
PYTHONPATH=/app
```

### Web App Environment Variables

```bash
REACT_APP_API_URL=https://your-backend.railway.app
```

---

## 🌍 Custom Domain (Optional)

### For Backend (Railway)
1. **Go to Railway dashboard**
2. **Click on your project**
3. **Go to "Settings" → "Domains"**
4. **Add custom domain**

### For Web App (Netlify)
1. **Go to Netlify dashboard**
2. **Click on your site**
3. **Go to "Domain management"**
4. **Add custom domain**

---

## 🚨 Important Security Notes

1. **Never commit `ocrtyre.json` to Git** (✅ Already in .gitignore)
2. **Use environment variables for sensitive data**
3. **Enable HTTPS** (Railway and Netlify do this automatically)
4. **Consider rate limiting** for production use

---

## 🧪 Testing Your Deployment

### Test Checklist:
- [ ] Backend health check: `https://your-backend.railway.app/health`
- [ ] Web app loads: `https://your-app.netlify.app`
- [ ] Camera works on mobile
- [ ] Image upload works
- [ ] OCR processing works
- [ ] Results display correctly

---

## 🔄 Updating Your App

### Backend Updates:
1. **Push changes to GitHub**
2. **Railway auto-deploys** (if connected to GitHub)

### Web App Updates:
1. **Push changes to GitHub**
2. **Netlify auto-deploys** (if connected to GitHub)

---

## 💰 Cost Estimate

### Free Tiers:
- **Railway:** 500 hours/month free
- **Netlify:** 100GB bandwidth/month free
- **Total:** $0/month for moderate use

### Paid Tiers (if needed):
- **Railway Pro:** $5/month
- **Netlify Pro:** $19/month

---

## 🆘 Troubleshooting

### Backend Issues:
- Check Railway logs in dashboard
- Verify environment variables
- Test locally first

### Web App Issues:
- Check Netlify logs
- Verify API URL is correct
- Test in different browsers

### Common Problems:
1. **CORS errors:** Backend allows all origins (✅ configured)
2. **Image upload fails:** Check file size limits
3. **OCR fails:** Verify Google Cloud credentials

---

## 📞 Support

If you encounter issues:
1. Check the logs in Railway/Netlify dashboards
2. Test locally first
3. Verify all environment variables are set
4. Check that your Google Cloud credentials are valid

---

## 🎉 Success!

Once deployed, your Tire OCR app will be accessible to anyone with the URL. Users can:
- Open the link on their phone
- Take photos of tires
- Get instant OCR analysis
- Add the app to their home screen (PWA)

**Your app is now live and ready for users! 🚀**
