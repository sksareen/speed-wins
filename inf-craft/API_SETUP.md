# OpenRouter API Setup Guide

## Current Status

The AI Infinite Craft app is now working in **Offline Mode** with predefined combinations. You can play the game immediately without any API key!

## To Enable AI Mode (Optional)

### 1. Get an OpenRouter API Key

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to your API keys section
4. Create a new API key
5. Copy the API key

### 2. Add the API Key

Edit the `.env` file in the project root:

```bash
# OpenRouter API key for AI features
# Get your API key from: https://openrouter.ai/
VITE_OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
```

### 3. Restart the Development Server

```bash
npm run dev
```

## Modes

### 💡 Offline Mode (Current)
- Uses predefined AI concept combinations
- Works immediately without API key
- Limited but educational combinations
- Status shows "💡 Offline Mode"

### 🤖 AI Mode (With API Key)
- Uses OpenRouter API with Claude 3.5 Sonnet
- Generates new AI concepts dynamically
- More combinations and creativity
- Status shows "🤖 AI Mode"

## Predefined Combinations (Offline Mode)

The app includes these combinations:
- **Data + Algorithm** → Machine Learning
- **Neural Network + Training** → Deep Learning
- **Attention + Neural Network** → Transformer
- **Compute + Gradient** → Optimization
- **Data + Compute** → Big Data
- **Algorithm + Gradient** → Gradient Descent
- **Neural Network + Gradient** → Backpropagation
- **Attention + Data** → Natural Language Processing
- **Training + Compute** → Distributed Training
- **Algorithm + Attention** → Reinforcement Learning

## Troubleshooting

### API Key Issues
- Make sure the API key starts with `sk-or-v1-`
- Check that you have credits in your OpenRouter account
- Verify the API key is correctly copied to `.env`

### 401 Unauthorized Error
- Your API key is invalid or expired
- You've run out of credits
- The API key format is incorrect

### Network Issues
- Check your internet connection
- OpenRouter might be temporarily unavailable
- The app will fall back to offline mode

## Cost Information

OpenRouter offers:
- **Free tier**: Limited requests per month
- **Paid plans**: More requests and faster models
- **Pay-as-you-go**: Pay only for what you use

The AI Infinite Craft game uses minimal tokens per request, so costs are typically very low.

## Support

If you encounter issues:
1. Check the browser console for error messages
2. Verify your API key is correct
3. Ensure you have sufficient credits
4. Try the offline mode if AI mode isn't working

The game works perfectly in offline mode, so you can enjoy it immediately! 🎮
