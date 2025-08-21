# LLM Efficiency InfiniteCraft Game

Learn LLM optimization concepts through discovery and combination! This game teaches efficiency techniques by letting you discover concepts organically through the same relationships that exist in actual research.

## 🎯 Project Overview

This project combines gamified learning with practical implementation to master LLM efficiency optimization. Play the game to understand concept relationships, then implement what you discover using the modular roadmap.

## 📁 Directory Structure

```
infinite-craft-project/
├── src/                    # React application source code
│   ├── components/
│   │   └── AiInfiniteCraft.tsx  # Main game component (35KB)
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── backend/                # Python backend applications
│   ├── web_app/           # Flask web application
│   ├── game/              # CLI game implementation
│   ├── run_web_app.py     # Web app runner
│   └── run_game.py        # Game runner
├── research/              # Research papers and analysis
├── docs/                  # Documentation and guides
├── data/                  # Concept definitions and data
├── scripts/               # Setup and utility scripts
├── logs/                  # Application logs
├── package.json           # Node.js dependencies
├── tsconfig.json          # TypeScript configuration
├── vite.config.ts         # Vite build configuration
├── tailwind.config.js     # Tailwind CSS configuration
├── index.html             # Main HTML file
├── README_AI_INFINITE_CRAFT.md
├── INFINITE_CRAFT_BEHAVIOR.md
├── API_SETUP.md
├── FIXES_APPLIED.md
└── SETUP_COMPLETE.md
```

## 🚀 Quick Start

### Frontend (React App)
```bash
npm install
# Create .env file with your OpenRouter API key
echo "VITE_OPENROUTER_API_KEY=your-api-key-here" > .env
npm run dev
```

### Backend (Python Web App)
```bash
cd backend/web_app
pip install -r requirements.txt
python web_app.py
```

### CLI Game
```bash
cd backend/game
python llm_efficiency_game.py
```

## 🔧 Setup Instructions

### Prerequisites
- Node.js and npm
- Python 3.x
- OpenRouter API key (optional, for AI features)

### Frontend Setup
1. Install dependencies: `npm install`
2. Create `.env` file with your OpenRouter API key (optional)
3. Run: `npm run dev`

### Backend Setup
1. Navigate to `backend/web_app/`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Run: `python web_app.py`

### API Configuration (Optional)
The app works in **Offline Mode** by default with predefined combinations. To enable AI-powered combinations:

1. Get an OpenRouter API key from [OpenRouter](https://openrouter.ai/)
2. Add it to your `.env` file: `VITE_OPENROUTER_API_KEY=your-api-key-here`
3. Restart the development server

**Modes:**
- **💡 Offline Mode** (Default): Uses predefined combinations, works immediately
- **🤖 AI Mode** (With API key): Uses OpenRouter API for dynamic combinations

## 🎮 How to Play

1. **Drag & Drop**: Drag AI concepts from the panel to the workspace
2. **Combine**: Drop one element onto another to create new combinations
3. **Discover**: AI generates new concepts based on your combinations
4. **Progress**: Track discoveries and unlock achievements

## 📚 Learning Resources

- **Learning Roadmap**: `docs/llm_efficiency_learning_roadmap.md`
- **Practical Guide**: `docs/llm_efficiency_practical_guide.md`
- **Research Papers**: `research/` directory
- **Concept Data**: `data/llm_efficiency_concepts.json`

## 🔬 Research Components

- **Main Paper**: `research/paper.md`
- **Dimensional Analysis**: `research/dimensions.md`
- **Model Documentation**: `research/models.md`
- **Resonance Theory**: `research/resonance.md`

## 🛠️ Development

### Frontend Development
- Built with React 18 + TypeScript
- Uses Vite for fast development
- Styled with Tailwind CSS
- Main component: `src/components/AiInfiniteCraft.tsx`

### Backend Development
- Flask web application in `backend/web_app/`
- CLI game in `backend/game/`
- Python-based with comprehensive requirements

### Data Management
- Concept definitions in `data/llm_efficiency_concepts.json`
- Extensible structure for adding new concepts

## 📊 Features

- **Interactive Gameplay**: Drag-and-drop interface
- **AI-Powered Combinations**: OpenRouter API integration
- **Progress Tracking**: Discovery and achievement system
- **Responsive Design**: Works on desktop and mobile
- **Search & Filter**: Find elements by name or category
- **Rarity System**: Common to Legendary concept tiers

## 🔍 Troubleshooting

### Common Issues
1. **API Key Problems**: Ensure OpenRouter API key is correctly set (optional)
2. **Port Conflicts**: Vite will auto-select next available port
3. **Build Issues**: Clear node_modules and reinstall if needed

### Offline Mode (Default)
- App works immediately without any API key
- Uses 10+ predefined AI concept combinations
- No setup required - just run `npm run dev`

### AI Mode (Optional)
- Requires valid OpenRouter API key
- Uses Claude 3.5 Sonnet for dynamic combinations
- Graceful fallback to offline mode if API fails

### Getting Help
- Check `API_SETUP.md` for detailed API configuration
- The app works perfectly in offline mode by default

## 📦 Portability

This entire directory is self-contained and can be:
- Copied to any location
- Shared as a complete project
- Deployed independently
- Used for development or production

## 📄 License

MIT License - feel free to use this project for learning and experimentation!

---

**Note**: This is the original, properly organized AI Infinite Craft project. All components are included and properly structured for easy deployment and development.
