#!/usr/bin/env python3
"""
Simple HTTP server to serve the mobile web app
Run this to make the web app accessible on your iPhone
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

# Get the directory where this script is located
DIRECTORY = Path(__file__).parent
PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers to allow requests from mobile devices
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def get_local_ip():
    """Get the local IP address of this machine"""
    import socket
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    print("🚀 Starting Tire OCR Mobile Web App Server")
    print("=" * 50)
    
    # Change to the web app directory
    os.chdir(DIRECTORY)
    
    # Get local IP address
    local_ip = get_local_ip()
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"✅ Server running at:")
            print(f"   Local: http://localhost:{PORT}")
            print(f"   Network: http://{local_ip}:{PORT}")
            print()
            print("📱 To use on your iPhone:")
            print(f"   1. Make sure your iPhone is on the same WiFi network")
            print(f"   2. Open Safari on your iPhone")
            print(f"   3. Go to: http://{local_ip}:{PORT}")
            print(f"   4. Add to Home Screen for app-like experience")
            print()
            print("🔧 Make sure your backend API is running on port 8000")
            print("   (Run: cd ../backend && python main.py)")
            print()
            print("Press Ctrl+C to stop the server")
            print("=" * 50)
            
            # Open browser automatically
            webbrowser.open(f"http://localhost:{PORT}")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Try a different port:")
            print(f"   python start_server.py --port 3001")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    # Check for port argument
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        try:
            PORT = int(sys.argv[2])
        except (IndexError, ValueError):
            print("❌ Invalid port number. Using default port 3000")
    
    main()
