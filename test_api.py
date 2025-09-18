#!/usr/bin/env python3
"""
Test script for the Tire OCR API
This script tests the API endpoints to ensure they're working correctly.
"""

import requests
import json
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint is working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_analyze_endpoint():
    """Test the analyze tire endpoint with a sample image"""
    print("\n🔍 Testing analyze tire endpoint...")
    
    # Look for a sample image
    sample_image = Path("sample_image/20240529_115909.jpg")
    if not sample_image.exists():
        print("⚠️  No sample image found. Skipping analyze test.")
        print("   Place a tire image at sample_image/20240529_115909.jpg to test")
        return True
    
    try:
        with open(sample_image, 'rb') as f:
            files = {'image': ('test_tire.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/analyze-tire", files=files, timeout=30)
        
        if response.status_code == 200:
            print("✅ Analyze tire endpoint is working")
            result = response.json()
            print("   Extracted information:")
            print(f"   - Manufacturer: {result.get('Manufacturer', 'N/A')}")
            print(f"   - Model: {result.get('Tire model', 'N/A')}")
            print(f"   - Size: {result.get('Tire size', 'N/A')}")
            print(f"   - DOT Code: {result.get('Scan TIN', {}).get('DOT Code', 'N/A')}")
            return True
        else:
            print(f"❌ Analyze tire endpoint failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Analyze tire endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Tire OCR API Test Suite")
    print("=" * 30)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Test analyze endpoint
    analyze_ok = test_analyze_endpoint()
    
    print("\n📊 Test Results:")
    print("=" * 15)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Analyze Tire: {'✅ PASS' if analyze_ok else '❌ FAIL'}")
    
    if health_ok and analyze_ok:
        print("\n🎉 All tests passed! The API is ready for the mobile app.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the logs and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
