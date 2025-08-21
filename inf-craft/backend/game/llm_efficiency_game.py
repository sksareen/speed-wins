#!/usr/bin/env python3
"""
LLM Efficiency Infinite Craft Game
A discovery-based learning game for understanding LLM efficiency concepts
"""

import json
import random
import time
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import os

@dataclass
class Concept:
    id: str
    name: str
    description: str
    difficulty: int
    category: str
    prerequisites: List[str]
    leads_to: List[str]
    properties: Dict
    resources: List[str]
    discovered: bool = False
    discovery_time: Optional[float] = None

@dataclass
class GameState:
    discovered_concepts: Set[str]
    available_combinations: List[Tuple[str, str]]
    score: int
    level: int
    start_time: float
    session_time: float

class LLMEfficiencyGame:
    def __init__(self, concepts_file: str = "../data/llm_efficiency_concepts.json"):
        """Initialize the game with concept database"""
        self.concepts_file = concepts_file
        self.concepts = {}
        self.relationships = {}
        self.categories = {}
        self.difficulty_levels = {}
        self.game_state = None
        self.load_concepts()
        self.initialize_game()
    
    def load_concepts(self):
        """Load concepts from JSON file"""
        try:
            with open(self.concepts_file, 'r') as f:
                data = json.load(f)
            
            # Load concepts
            for concept_id, concept_data in data['concepts'].items():
                self.concepts[concept_id] = Concept(
                    id=concept_id,
                    name=concept_data['name'],
                    description=concept_data['description'],
                    difficulty=concept_data['difficulty'],
                    category=concept_data['category'],
                    prerequisites=concept_data['prerequisites'],
                    leads_to=concept_data['leads_to'],
                    properties=concept_data['properties'],
                    resources=concept_data['resources']
                )
            
            self.relationships = data['relationships']
            self.categories = data['categories']
            self.difficulty_levels = data['difficulty_levels']
            
        except FileNotFoundError:
            print(f"Error: Could not find {self.concepts_file}")
            print("Please ensure the concepts file is in the same directory.")
            exit(1)
    
    def initialize_game(self):
        """Initialize the game state"""
        # Start with basic mathematical foundations
        starting_concepts = ['linear_algebra', 'calculus', 'probability']
        
        self.game_state = GameState(
            discovered_concepts=set(starting_concepts),
            available_combinations=[],
            score=0,
            level=1,
            start_time=time.time(),
            session_time=0
        )
        
        # Mark starting concepts as discovered
        for concept_id in starting_concepts:
            if concept_id in self.concepts:
                self.concepts[concept_id].discovered = True
                self.concepts[concept_id].discovery_time = time.time()
        
        self.update_available_combinations()
    
    def update_available_combinations(self):
        """Update available combinations based on discovered concepts"""
        discovered = self.game_state.discovered_concepts
        combinations = []
        
        # Find all possible combinations of discovered concepts
        discovered_list = list(discovered)
        for i in range(len(discovered_list)):
            for j in range(i + 1, len(discovered_list)):
                concept1 = discovered_list[i]
                concept2 = discovered_list[j]
                
                # Check if this combination can lead to new concepts
                new_concepts = self.get_combination_results(concept1, concept2)
                if new_concepts:
                    combinations.append((concept1, concept2, new_concepts))
        
        self.game_state.available_combinations = combinations
    
    def get_combination_results(self, concept1: str, concept2: str) -> List[str]:
        """Get concepts that can be discovered by combining two concepts"""
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return []
        
        c1 = self.concepts[concept1]
        c2 = self.concepts[concept2]
        
        # Find concepts that have both as prerequisites
        potential_results = []
        for concept_id, concept in self.concepts.items():
            if concept_id in self.game_state.discovered_concepts:
                continue
            
            prereqs = set(concept.prerequisites)
            if concept1 in prereqs and concept2 in prereqs:
                # Check if all other prerequisites are also discovered
                if prereqs.issubset(self.game_state.discovered_concepts):
                    potential_results.append(concept_id)
        
        return potential_results
    
    def combine_concepts(self, concept1: str, concept2: str) -> Optional[str]:
        """Combine two concepts and return the result"""
        if concept1 not in self.game_state.discovered_concepts or concept2 not in self.game_state.discovered_concepts:
            return None
        
        new_concepts = self.get_combination_results(concept1, concept2)
        if not new_concepts:
            return None
        
        # Select the concept with lowest difficulty
        result = min(new_concepts, key=lambda x: self.concepts[x].difficulty)
        
        # Discover the new concept
        self.discover_concept(result)
        return result
    
    def discover_concept(self, concept_id: str):
        """Discover a new concept"""
        if concept_id not in self.concepts:
            return
        
        concept = self.concepts[concept_id]
        if concept.discovered:
            return
        
        concept.discovered = True
        concept.discovery_time = time.time()
        self.game_state.discovered_concepts.add(concept_id)
        
        # Update score based on difficulty
        self.game_state.score += concept.difficulty * 10
        
        # Update level
        self.game_state.level = len(self.game_state.discovered_concepts) // 5 + 1
        
        # Update available combinations
        self.update_available_combinations()
        
        # Show discovery message
        self.show_discovery_message(concept)
    
    def show_discovery_message(self, concept: Concept):
        """Show a discovery message for a new concept"""
        print(f"\n🎉 DISCOVERED: {concept.name} 🎉")
        print(f"Category: {concept.category}")
        print(f"Difficulty: {'⭐' * concept.difficulty}")
        print(f"Description: {concept.description}")
        
        if concept.properties:
            print("Key Properties:")
            for key, value in concept.properties.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"Score: +{concept.difficulty * 10}")
        print("-" * 50)
    
    def show_available_combinations(self):
        """Show available combinations to the player"""
        if not self.game_state.available_combinations:
            print("\nNo new combinations available!")
            print("Try exploring different concepts or review what you've learned.")
            return
        
        print(f"\n🔬 Available Combinations ({len(self.game_state.available_combinations)}):")
        for i, (concept1, concept2, results) in enumerate(self.game_state.available_combinations):
            c1_name = self.concepts[concept1].name
            c2_name = self.concepts[concept2].name
            result_names = [self.concepts[r].name for r in results]
            print(f"{i+1}. {c1_name} + {c2_name} → {', '.join(result_names)}")
    
    def show_discovered_concepts(self):
        """Show all discovered concepts"""
        print(f"\n📚 Discovered Concepts ({len(self.game_state.discovered_concepts)}):")
        
        # Group by category
        by_category = defaultdict(list)
        for concept_id in self.game_state.discovered_concepts:
            concept = self.concepts[concept_id]
            by_category[concept.category].append(concept)
        
        for category, concepts in by_category.items():
            print(f"\n{category.upper()}:")
            for concept in sorted(concepts, key=lambda x: x.difficulty):
                print(f"  {concept.name} {'⭐' * concept.difficulty}")
    
    def show_game_stats(self):
        """Show current game statistics"""
        self.game_state.session_time = time.time() - self.game_state.start_time
        
        print(f"\n📊 Game Statistics:")
        print(f"Score: {self.game_state.score}")
        print(f"Level: {self.game_state.level}")
        print(f"Concepts Discovered: {len(self.game_state.discovered_concepts)}")
        print(f"Session Time: {self.game_state.session_time:.1f}s")
        
        # Show progress by difficulty
        by_difficulty = defaultdict(int)
        for concept_id in self.game_state.discovered_concepts:
            concept = self.concepts[concept_id]
            by_difficulty[concept.difficulty] += 1
        
        print(f"\nProgress by Difficulty:")
        for difficulty in sorted(by_difficulty.keys()):
            total = len(self.difficulty_levels.get(str(difficulty), []))
            discovered = by_difficulty[difficulty]
            print(f"  Level {difficulty}: {discovered}/{total} ({discovered/total*100:.1f}%)")
    
    def get_concept_info(self, concept_id: str):
        """Get detailed information about a concept"""
        if concept_id not in self.concepts:
            print(f"Concept '{concept_id}' not found.")
            return
        
        concept = self.concepts[concept_id]
        print(f"\n📖 {concept.name}")
        print(f"ID: {concept.id}")
        print(f"Category: {concept.category}")
        print(f"Difficulty: {'⭐' * concept.difficulty}")
        print(f"Description: {concept.description}")
        
        if concept.properties:
            print(f"\nProperties:")
            for key, value in concept.properties.items():
                if isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")
        
        if concept.prerequisites:
            prereq_names = [self.concepts[p].name for p in concept.prerequisites if p in self.concepts]
            print(f"\nPrerequisites: {', '.join(prereq_names)}")
        
        if concept.leads_to:
            lead_names = [self.concepts[l].name for l in concept.leads_to if l in self.concepts]
            print(f"Leads to: {', '.join(lead_names)}")
        
        if concept.resources:
            print(f"\nResources: {', '.join(concept.resources)}")
        
        if concept.discovered:
            print(f"\nDiscovered: Yes")
        else:
            print(f"\nDiscovered: No")
    
    def search_concepts(self, query: str):
        """Search for concepts by name or description"""
        query = query.lower()
        results = []
        
        for concept_id, concept in self.concepts.items():
            if (query in concept.name.lower() or 
                query in concept.description.lower() or
                query in concept.category.lower()):
                results.append(concept)
        
        if results:
            print(f"\n🔍 Search Results for '{query}' ({len(results)} found):")
            for concept in results:
                status = "✅" if concept.discovered else "❌"
                print(f"{status} {concept.name} ({concept.category}) {'⭐' * concept.difficulty}")
        else:
            print(f"\nNo concepts found matching '{query}'")
    
    def show_learning_path(self, target_concept: str):
        """Show the learning path to a specific concept"""
        if target_concept not in self.concepts:
            print(f"Concept '{target_concept}' not found.")
            return
        
        target = self.concepts[target_concept]
        if target.discovered:
            print(f"\n✅ {target.name} is already discovered!")
            return
        
        # Find the shortest path to the target
        path = self.find_shortest_path(target_concept)
        if not path:
            print(f"\n❌ No path found to {target.name}")
            return
        
        print(f"\n🛤️  Learning Path to {target.name}:")
        for i, concept_id in enumerate(path):
            concept = self.concepts[concept_id]
            status = "✅" if concept.discovered else "❌"
            print(f"{i+1}. {status} {concept.name} {'⭐' * concept.difficulty}")
    
    def find_shortest_path(self, target: str) -> List[str]:
        """Find shortest path to target concept using BFS"""
        if target in self.game_state.discovered_concepts:
            return [target]
        
        # BFS to find shortest path
        queue = [(concept_id, [concept_id]) for concept_id in self.game_state.discovered_concepts]
        visited = set(self.game_state.discovered_concepts)
        
        while queue:
            current, path = queue.pop(0)
            
            # Check if current concept leads to target
            if target in self.concepts[current].leads_to:
                return path + [target]
            
            # Add undiscovered concepts that current leads to
            for next_concept in self.concepts[current].leads_to:
                if next_concept not in visited and next_concept in self.concepts:
                    visited.add(next_concept)
                    queue.append((next_concept, path + [next_concept]))
        
        return []
    
    def show_menu(self):
        """Show the main game menu"""
        print(f"\n{'='*60}")
        print(f"🧠 LLM Efficiency Infinite Craft Game")
        print(f"{'='*60}")
        print(f"Score: {self.game_state.score} | Level: {self.game_state.level}")
        print(f"Concepts: {len(self.game_state.discovered_concepts)}")
        print(f"{'='*60}")
        print("1. 🔬 Show available combinations")
        print("2. 🧪 Combine concepts")
        print("3. 📚 View discovered concepts")
        print("4. 📊 Show game statistics")
        print("5. 🔍 Search concepts")
        print("6. 📖 Get concept details")
        print("7. 🛤️  Show learning path")
        print("8. 💾 Save game")
        print("9. 📂 Load game")
        print("0. 🚪 Exit")
        print(f"{'='*60}")
    
    def save_game(self, filename: str = "llm_efficiency_save.json"):
        """Save the current game state"""
        save_data = {
            "discovered_concepts": list(self.game_state.discovered_concepts),
            "score": self.game_state.score,
            "level": self.game_state.level,
            "start_time": self.game_state.start_time,
            "session_time": self.game_state.session_time
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Game saved to {filename}")
    
    def load_game(self, filename: str = "llm_efficiency_save.json"):
        """Load a saved game state"""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)
            
            self.game_state.discovered_concepts = set(save_data["discovered_concepts"])
            self.game_state.score = save_data["score"]
            self.game_state.level = save_data["level"]
            self.game_state.start_time = save_data["start_time"]
            self.game_state.session_time = save_data["session_time"]
            
            # Mark concepts as discovered
            for concept_id in self.game_state.discovered_concepts:
                if concept_id in self.concepts:
                    self.concepts[concept_id].discovered = True
            
            self.update_available_combinations()
            print(f"\n📂 Game loaded from {filename}")
            
        except FileNotFoundError:
            print(f"\n❌ Save file {filename} not found")
    
    def run(self):
        """Run the main game loop"""
        print("Welcome to the LLM Efficiency Infinite Craft Game!")
        print("Discover concepts by combining mathematical foundations and neural network principles.")
        print("Start with Linear Algebra, Calculus, and Probability to build your knowledge!")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice (0-9): ").strip()
                
                if choice == "1":
                    self.show_available_combinations()
                
                elif choice == "2":
                    self.show_available_combinations()
                    if self.game_state.available_combinations:
                        try:
                            combo_choice = input("\nEnter combination number: ").strip()
                            combo_idx = int(combo_choice) - 1
                            if 0 <= combo_idx < len(self.game_state.available_combinations):
                                concept1, concept2, _ = self.game_state.available_combinations[combo_idx]
                                result = self.combine_concepts(concept1, concept2)
                                if result:
                                    print(f"\n🎉 Successfully discovered: {self.concepts[result].name}")
                                else:
                                    print("\n❌ Combination failed")
                            else:
                                print("\n❌ Invalid combination number")
                        except ValueError:
                            print("\n❌ Please enter a valid number")
                
                elif choice == "3":
                    self.show_discovered_concepts()
                
                elif choice == "4":
                    self.show_game_stats()
                
                elif choice == "5":
                    query = input("\nEnter search query: ").strip()
                    self.search_concepts(query)
                
                elif choice == "6":
                    concept_id = input("\nEnter concept ID: ").strip()
                    self.get_concept_info(concept_id)
                
                elif choice == "7":
                    target = input("\nEnter target concept ID: ").strip()
                    self.show_learning_path(target)
                
                elif choice == "8":
                    filename = input("\nEnter save filename (default: llm_efficiency_save.json): ").strip()
                    if not filename:
                        filename = "llm_efficiency_save.json"
                    self.save_game(filename)
                
                elif choice == "9":
                    filename = input("\nEnter load filename (default: llm_efficiency_save.json): ").strip()
                    if not filename:
                        filename = "llm_efficiency_save.json"
                    self.load_game(filename)
                
                elif choice == "0":
                    print("\n👋 Thanks for playing! Keep exploring LLM efficiency concepts!")
                    break
                
                else:
                    print("\n❌ Invalid choice. Please enter a number between 0-9.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Game interrupted. Thanks for playing!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

def main():
    """Main function to run the game"""
    game = LLMEfficiencyGame()
    game.run()

if __name__ == "__main__":
    main()
