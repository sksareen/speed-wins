#!/usr/bin/env python3
"""
Run LLM Efficiency Game - Terminal or Web API
"""

import sys
import os
import argparse

def run_terminal():
    """Run the terminal version of the game"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game'))
    from llm_efficiency_game import LLMEfficiencyGame
    
    # Use the complete concepts database
    game = LLMEfficiencyGame("../data/llm_efficiency_concepts_complete.json")
    game.run()

def run_web_api():
    """Run the web API server"""
    from web_app.game_api import app
    print("Starting LLM Efficiency Game API server...")
    print("Frontend should connect to http://localhost:5001")
    print("Game concepts loaded from complete database")
    app.run(debug=True, host='0.0.0.0', port=5001)

def main():
    parser = argparse.ArgumentParser(description='LLM Efficiency Game')
    parser.add_argument('--mode', choices=['terminal', 'web'], default='web',
                       help='Run mode: terminal for CLI, web for API server')
    
    args = parser.parse_args()
    
    if args.mode == 'terminal':
        run_terminal()
    else:
        run_web_api()

if __name__ == "__main__":
    main()
