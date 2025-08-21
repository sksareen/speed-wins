#!/usr/bin/env python3
"""
Setup script for AI Infinite Craft React App
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_node_version():
    """Check if Node.js is installed and has a compatible version"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"✅ Node.js version: {version}")
        return True
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        print("Please install Node.js from: https://nodejs.org/")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = '.env'
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("# OpenRouter API key for AI features\n")
            f.write("# Get your API key from: https://openrouter.ai/\n")
            f.write("VITE_OPENROUTER_API_KEY=your-openrouter-api-key-here\n")
        print("✅ Created .env file")
        print("⚠️  Please update the .env file with your OpenRouter API key")
    else:
        print("✅ .env file already exists")

def main():
    print("🚀 AI Infinite Craft React App Setup")
    print("=" * 50)
    
    # Check Node.js
    if not check_node_version():
        sys.exit(1)
    
    # Install dependencies
    if not run_command("npm install", "Installing dependencies"):
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Update the .env file with your OpenRouter API key")
    print("2. Run the development server:")
    print("   npm run dev")
    print("3. Open your browser and go to: http://localhost:3000")
    print("\nFor help, see README_AI_INFINITE_CRAFT.md")

if __name__ == "__main__":
    main()
