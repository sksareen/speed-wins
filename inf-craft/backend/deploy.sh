#!/bin/bash

# Deploy LLM Efficiency Game Backend to Fly.io

echo "🚀 Deploying LLM Efficiency Game Backend to Fly.io..."

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "❌ Fly CLI is not installed. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if user is logged in
if ! fly auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run: fly auth login"
    exit 1
fi

# Deploy the application
echo "📦 Building and deploying..."
fly deploy

# Check deployment status
echo "✅ Deployment complete!"
echo "🌐 Your backend is available at: https://llm-efficiency-game-backend.fly.dev"
echo "🔍 Health check: https://llm-efficiency-game-backend.fly.dev/health"
