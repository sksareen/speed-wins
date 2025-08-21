# LLM Efficiency InfiniteCraft Game

Learn LLM optimization concepts through discovery and combination! This game teaches efficiency techniques by letting you discover concepts organically through the same relationships that exist in actual research.

## 🎯 Project Overview

This project combines gamified learning with practical implementation to master LLM efficiency optimization. Play the game to understand concept relationships, then implement what you discover using the modular roadmap.

## 📁 Directory Structure

```
inf-craft/
├── src/                           # React frontend
│   ├── components/
│   │   ├── LLMEfficiencyGame.tsx  # Main efficiency game
│   │   └── AiInfiniteCraft.tsx    # Original game (legacy)
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── backend/                       # Python backend
│   ├── game/                      # Core game logic
│   │   └── llm_efficiency_game.py
│   ├── web_app/                   # Flask API
│   │   ├── game_api.py
│   │   └── requirements.txt
│   └── run_game.py                # Unified runner
├── data/                          # Game content
│   └── llm_efficiency_concepts_complete.json
├── research/                      # Learning resources
│   ├── llm-efficiency/
│   │   ├── modular_optimization_roadmap.md
│   │   └── game_based_learning_guide.md
│   ├── speed-wins.md             # Technical foundation
│   └── models.md                 # Intelligence scaffolding
├── docs/                         # Additional documentation
├── package.json                  # Node.js dependencies
├── tsconfig.json                 # TypeScript configuration
├── vite.config.ts               # Vite build configuration
└── tailwind.config.js           # Tailwind CSS configuration
```

## 🚀 Quick Start

### 1. Backend Setup (API Server)
```bash
cd backend
pip install flask flask-cors

# Run web API for React frontend
python run_game.py --mode web
# Server runs on http://localhost:5001
```

### 2. Frontend Setup (React Game)
```bash
# Install dependencies
npm install

# Start development server
npm run dev
# Frontend runs on http://localhost:5173
```

### 3. Terminal Game (Optional)
```bash
cd backend
python run_game.py --mode terminal
```

## 🔧 Prerequisites

- **Node.js** and npm (for React frontend)
- **Python 3.7+** (for Flask backend)
- No API keys required - works completely offline!

## 🎮 How to Play

### Game Mechanics
1. **Click or Drag**: Add concepts from sidebar to canvas
2. **Combine**: Drag elements close together to discover new concepts
3. **Remove**: Hover over canvas elements and click × to remove
4. **Progress**: Track score, level, and discovery progress

### Starting Concepts
- **Linear Algebra** (⭐) - Mathematical foundation
- **Calculus** (⭐) - Derivatives and optimization  
- **Probability** (⭐) - Statistical foundations

### Discovery Examples
- Linear Algebra + Calculus + Probability → **Neural Networks**
- Neural Networks + Linear Algebra → **Attention Mechanism**
- Attention + Memory Bandwidth → **FlashAttention**
- State Space Models + Gating → **Mamba**

## 📚 Learning Integration

### Game → Implementation Flow
1. **Discover concepts** through game combinations
2. **Understand relationships** between techniques
3. **Implement discoveries** using the modular roadmap
4. **Validate improvements** with measurement infrastructure

### Key Resources
- `research/llm-efficiency/modular_optimization_roadmap.md` - Implementation guide
- `research/llm-efficiency/game_based_learning_guide.md` - How to use game for learning
- `research/speed-wins.md` - Technical foundation (Speed Always Wins survey)
- `data/llm_efficiency_concepts_complete.json` - 100+ concept database

## 🛠️ Technical Details

### Frontend Stack
- **React 18** + TypeScript for robust UI
- **Vite** for fast development and building
- **Tailwind CSS** for responsive styling
- **Main Component**: `LLMEfficiencyGame.tsx`

### Backend Stack
- **Flask** API server with CORS support
- **Python game engine** with concept relationships
- **JSON database** with 100+ LLM efficiency concepts
- **Unified runner** for terminal or web modes

### Game Database
- Comprehensive concept definitions in `data/llm_efficiency_concepts_complete.json`
- Real research relationships between techniques
- Difficulty progression from basic math to production systems
- Extensible structure for adding new discoveries

## 📊 Features

### Core Gameplay
- **InfiniteCraft-style** drag-and-drop interface
- **Click to add** elements to canvas
- **Proximity-based combination** detection
- **Element removal** with hover controls
- **Responsive canvas** that adapts to screen size

### Learning Features  
- **100+ concepts** from basic math to production optimization
- **Real relationships** mirror actual research evolution
- **Progress tracking** by difficulty and category
- **Hint system** for guided discovery
- **Search and filter** by name, category, difficulty

### Integration Features
- **Module mapping** - each discovery connects to implementation guide
- **Validation checkpoints** for measuring improvements
- **Resource links** to papers and tutorials
- **Save/load** game progress

## 🔍 Troubleshooting

### Setup Issues
- **Port conflicts**: Backend uses 5001, frontend uses 5173
- **Python dependencies**: Run `pip install flask flask-cors`
- **Node issues**: Clear `node_modules` and reinstall if needed

### Game Issues
- **Elements not combining**: Drag closer together (< 80px distance)
- **Elements outside canvas**: Resize handling keeps them in bounds
- **Duplicates**: Each element has unique key, no duplication issues

### Performance
- **Large concept database**: Lazy loading and filtering optimize performance
- **Memory usage**: Canvas elements are efficiently managed
- **Responsive**: Works on desktop, tablet, and mobile

## 📦 Self-Contained Project

This project is completely self-contained:
- ✅ **No external APIs** required
- ✅ **Complete concept database** included
- ✅ **Full documentation** and guides
- ✅ **Ready to run** with minimal setup
- ✅ **Extensible** for adding new concepts

Perfect for learning LLM efficiency through discovery and hands-on implementation!

## 📄 License

MIT License - Use freely for learning, research, and development.
