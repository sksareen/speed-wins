# LLM Efficiency Game - Web Application

## 🚀 Overview

A beautiful, interactive web application for learning LLM efficiency concepts through discovery-based gameplay, powered by AI using OpenRouter. This transforms the command-line game into a modern, engaging web experience.

## ✨ Features

### 🎮 Interactive Gameplay
- **Discovery-Based Learning**: Combine concepts to unlock new ones
- **Real-Time Updates**: See your progress and available combinations instantly
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Progress Tracking**: Score, level, and concept discovery tracking

### 🤖 AI-Powered Features
- **Smart Explanations**: AI-generated explanations tailored to your level
- **Intelligent Hints**: Get helpful hints for concept combinations
- **Learning Path Suggestions**: AI-guided learning recommendations
- **Contextual Help**: Understand concepts in relation to what you already know

### 🔍 Advanced Features
- **Concept Search**: Find any concept in the database
- **Detailed Views**: Deep dive into concept properties and relationships
- **Save/Load**: Persist your learning progress
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🛠️ Quick Start

### Prerequisites
- Python 3.7 or higher
- OpenRouter API key (optional, for AI features)

### 1. Automatic Setup
```bash
# Run the setup script
python setup_web_app.py
```

### 2. Manual Setup
```bash
# Install dependencies
pip install flask requests python-dotenv

# Set environment variables
export OPENROUTER_API_KEY="your-api-key-here"  # Optional
export SECRET_KEY="your-secret-key-here"

# Run the application
python web_app.py
```

### 3. Access the Application
Open your browser and go to: **http://localhost:5000**

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project directory:

```env
# Flask secret key (change this in production)
SECRET_KEY=your-secret-key-change-in-production

# OpenRouter API key (optional - for AI features)
# Get your API key from: https://openrouter.ai/
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=1
```

### OpenRouter Setup
1. Go to [OpenRouter](https://openrouter.ai/)
2. Create an account and get your API key
3. Add the key to your `.env` file
4. The AI features will automatically activate

## 🎯 How to Play

### Getting Started
1. **Initialize Game**: Click "New Game" to start
2. **View Concepts**: See your discovered concepts in the left panel
3. **Find Combinations**: Available combinations appear in the right panel
4. **Combine Concepts**: Click on combinations to discover new concepts
5. **Get AI Help**: Use the "Hint" button for guidance

### Game Mechanics
- **Starting Concepts**: Linear Algebra, Calculus, Probability
- **Scoring**: Each concept gives points based on difficulty (1-5 stars)
- **Leveling**: Level up every 5 discoveries
- **Progression**: Unlock more complex concepts as you advance

### AI Features
- **Explanations**: Click on any concept for AI-generated explanations
- **Hints**: Get contextual hints for combinations
- **Search**: Find concepts by name or description
- **Learning Paths**: Get personalized learning recommendations

## 🏗️ Architecture

### Backend (Flask)
- **Game Logic**: Concept combination and discovery mechanics
- **AI Integration**: OpenRouter API for intelligent explanations
- **Session Management**: Save/load game state
- **REST API**: JSON endpoints for frontend communication

### Frontend (HTML/CSS/JavaScript)
- **Modern UI**: Responsive design with CSS Grid and Flexbox
- **Interactive Elements**: Smooth animations and transitions
- **Real-Time Updates**: Dynamic content without page reloads
- **Search Functionality**: Instant concept search

### AI Tutor
- **Contextual Explanations**: Tailored to user's current level
- **Smart Hints**: Guide without giving away answers
- **Learning Paths**: Personalized recommendations
- **Fallback Support**: Works without API key

## 📁 File Structure

```
theory/
├── web_app.py                 # Main Flask application
├── setup_web_app.py          # Setup script
├── llm_efficiency_concepts.json  # Concept database
├── templates/
│   └── index.html            # Main web interface
├── static/                   # Static assets (CSS, JS, images)
├── logs/                     # Application logs
├── .env                      # Environment configuration
└── README_Web_App.md         # This file
```

## 🔌 API Endpoints

### Game Management
- `POST /api/game/init` - Initialize new game session
- `GET /api/game/state` - Get current game state
- `POST /api/game/save` - Save game progress
- `POST /api/game/load` - Load saved game

### Gameplay
- `POST /api/game/combine` - Combine two concepts
- `POST /api/game/hint` - Get AI hint for combination
- `POST /api/game/explain` - Get AI explanation of concept
- `POST /api/game/path` - Get learning path suggestion

### Search
- `GET /api/game/search?q=<query>` - Search for concepts

## 🎨 UI Components

### Game Stats Panel
- **Score Display**: Current points earned
- **Level Indicator**: Current learning level
- **Concept Counter**: Number of discoveries
- **Combination Counter**: Available combinations

### Concept Cards
- **Gradient Backgrounds**: Visual appeal with color coding
- **Difficulty Stars**: 1-5 star rating system
- **Category Badges**: Concept classification
- **Hover Effects**: Interactive feedback

### Combination Cards
- **Formula Display**: Shows concept combination
- **Result Preview**: Possible discoveries
- **Hint Button**: AI-powered guidance
- **Click to Combine**: Interactive discovery

### Modals
- **Discovery Modal**: Celebrate new findings
- **Concept Details**: Deep dive into concepts
- **AI Explanations**: Markdown-formatted content
- **Responsive Design**: Works on all screen sizes

## 🤖 AI Integration

### OpenRouter Models
- **Claude 3.5 Sonnet**: Primary model for explanations
- **Context-Aware**: Considers user's current level
- **Personalized**: Adapts to learning progress
- **Fallback Support**: Works without API key

### AI Features
1. **Concept Explanations**
   - Tailored to user's level
   - Uses analogies and examples
   - Connects to known concepts
   - Includes practical insights

2. **Smart Hints**
   - Guides without spoiling
   - Uses metaphors and analogies
   - Encourages exploration
   - Contextual to combination

3. **Learning Paths**
   - Personalized recommendations
   - Acknowledges current knowledge
   - Suggests efficient next steps
   - Motivates continued learning

## 🚀 Deployment

### Local Development
```bash
python web_app.py
```

### Production Deployment
```bash
# Set production environment
export FLASK_ENV=production
export SECRET_KEY="your-secure-secret-key"

# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "web_app:app"]
```

## 🔧 Customization

### Adding New Concepts
1. Edit `llm_efficiency_concepts.json`
2. Add new concept with proper relationships
3. Restart the application

### Customizing AI Prompts
1. Modify the `AITutor` class in `web_app.py`
2. Adjust prompts for different AI behaviors
3. Test with different models

### Styling Changes
1. Edit CSS in `templates/index.html`
2. Modify color schemes and layouts
3. Add new animations and effects

## 🐛 Troubleshooting

### Common Issues

**Application won't start**
- Check Python version (3.7+ required)
- Verify all dependencies are installed
- Check if port 5000 is available

**AI features not working**
- Verify OpenRouter API key is set
- Check internet connection
- Review API key permissions

**Concepts not loading**
- Validate JSON syntax in concepts file
- Check file permissions
- Review error logs

**UI not displaying correctly**
- Clear browser cache
- Check browser compatibility
- Verify all static files are served

### Debug Mode
```bash
export FLASK_DEBUG=1
python web_app.py
```

### Logs
Check the `logs/` directory for application logs and error details.

## 📈 Performance

### Optimization Tips
- **Caching**: Implement Redis for session storage
- **CDN**: Use CDN for static assets
- **Database**: Move to PostgreSQL for production
- **Load Balancing**: Use multiple workers with Gunicorn

### Monitoring
- **Health Checks**: `/health` endpoint for monitoring
- **Metrics**: Track user engagement and learning progress
- **Error Tracking**: Monitor AI API failures and fallbacks

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Include error handling

## 📄 License

This project is open source. See the main project license for details.

## 🙏 Acknowledgments

- **OpenRouter**: For providing AI model access
- **Flask**: For the web framework
- **Font Awesome**: For icons
- **Inter Font**: For typography

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the main project documentation
3. Create an issue in the repository
4. Contact the development team

---

**Happy Learning! 🧠✨**
