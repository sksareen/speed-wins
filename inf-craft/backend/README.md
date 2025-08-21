# LLM Efficiency Game Backend

Flask API backend for the LLM Efficiency InfiniteCraft game.

## Features

- RESTful API for game state management
- Concept combination logic
- Progress tracking and statistics
- Documentation serving
- Session management

## Local Development

1. Install dependencies:
```bash
pip install -r web_app/requirements.txt
```

2. Run the development server:
```bash
python web_app/game_api.py
```

The server will be available at `http://localhost:5001`

## Deployment to Fly.io

### Prerequisites

1. Install Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login to Fly.io:
```bash
fly auth login
```

### Deploy

1. Navigate to the backend directory:
```bash
cd backend
```

2. Run the deployment script:
```bash
./deploy.sh
```

Or deploy manually:
```bash
fly deploy
```

### Configuration

The app is configured via `fly.toml`:
- App name: `llm-efficiency-game-backend`
- Primary region: `iad` (Washington DC)
- Port: `5001`
- Memory: `512MB`
- CPU: `1 shared core`

### Environment Variables

- `FLASK_APP`: Set to `web_app/game_api.py`
- `FLASK_ENV`: Set to `production`
- `PORT`: Automatically set by Fly.io

## API Endpoints

- `GET /health` - Health check
- `POST /api/session` - Create new game session
- `GET /api/game/state` - Get current game state
- `POST /api/combine` - Combine two concepts
- `GET /api/concepts` - Get all concepts
- `GET /api/concept/<id>` - Get concept details
- `GET /api/docs/<filename>` - Serve documentation
- `GET /api/statistics` - Get game statistics
- `POST /api/reset` - Reset game
- `POST /api/hint` - Get hint

## Health Check

The app includes a health check endpoint at `/health` that returns:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Monitoring

- Health checks run every 30 seconds
- Auto-scaling enabled (0-1 machines)
- HTTPS enforced
- Grace period: 10 seconds

## Troubleshooting

1. Check deployment status:
```bash
fly status
```

2. View logs:
```bash
fly logs
```

3. SSH into the machine:
```bash
fly ssh console
```

4. Restart the app:
```bash
fly apps restart
```
