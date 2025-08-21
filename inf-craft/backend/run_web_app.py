#!/usr/bin/env python3
"""
Runner script for the LLM Efficiency Web App
This script allows running the web app from the theory directory root
"""

import sys
import os

# Add the web_app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_app'))

# Import and run the web app
from web_app import app

if __name__ == "__main__":
    print("🌐 Starting LLM Efficiency Web App...")
    print("📱 Open your browser and go to: http://localhost:9000")
    print("🛑 Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=9000)
