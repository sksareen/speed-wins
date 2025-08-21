#!/usr/bin/env python3
"""
Runner script for the LLM Efficiency Game
This script allows running the game from the theory directory root
"""

import sys
import os

# Add the game directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game'))

# Import and run the game
from llm_efficiency_game import LLMEfficiencyGame

if __name__ == "__main__":
    # Create game instance with correct path
    game = LLMEfficiencyGame("data/llm_efficiency_concepts.json")
    game.run()
