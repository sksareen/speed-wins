#!/usr/bin/env python3
"""
Flask API for LLM Efficiency InfiniteCraft Game
Provides REST endpoints for the React frontend
"""

from flask import Flask, jsonify, request, session, send_from_directory
from flask_cors import CORS
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple

# Add parent directory to path to import game module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.llm_efficiency_game import LLMEfficiencyGame, Concept, GameState

app = Flask(__name__)
app.secret_key = 'llm-efficiency-secret-key-2024'
CORS(app, supports_credentials=True)

# Store game instances per session
game_instances: Dict[str, LLMEfficiencyGame] = {}

def get_or_create_game(session_id: str) -> LLMEfficiencyGame:
    """Get existing game or create new one for session"""
    if session_id not in game_instances:
        game_instances[session_id] = LLMEfficiencyGame()
    return game_instances[session_id]

def concept_to_dict(concept: Concept) -> dict:
    """Convert Concept object to dictionary for JSON serialization"""
    return {
        'id': concept.id,
        'name': concept.name,
        'description': concept.description,
        'difficulty': concept.difficulty,
        'category': concept.category,
        'prerequisites': concept.prerequisites,
        'leads_to': concept.leads_to,
        'properties': concept.properties,
        'resources': concept.resources,
        'discovered': concept.discovered,
        'discovery_time': concept.discovery_time
    }

def get_rarity_from_difficulty(difficulty: int) -> str:
    """Map difficulty to rarity for frontend compatibility"""
    rarity_map = {
        1: 'common',
        2: 'uncommon',
        3: 'rare',
        4: 'epic',
        5: 'legendary'
    }
    return rarity_map.get(difficulty, 'common')

def get_emoji_for_category(category: str) -> str:
    """Get emoji representation for concept category"""
    emoji_map = {
        'mathematics': '📐',
        'deep_learning': '🧠',
        'transformer': '🔄',
        'attention': '👁️',
        'efficient_attention': '⚡',
        'linear_models': '📊',
        'ssm': '🌊',
        'moe': '🎭',
        'hybrid_models': '🔀',
        'optimization': '📈',
        'quantization': '🔢',
        'memory_optimization': '💾',
        'orchestration': '🎼',
        'training': '🏋️',
        'serving': '🚀',
        'deployment': '📦',
        'hardware': '💻',
        'distributed': '🌐',
        'frameworks': '🛠️',
        'production_models': '🏭',
        'measurement': '📏',
        'sparse_patterns': '🕸️',
        'alternative_architectures': '🏗️',
        'theory': '📚',
        'components': '🧩',
        'architectures': '🏛️',
        'scaling': '📊',
        'inference_optimization': '⚡',
        'operations': '⚙️',
        'phenomena': '✨'
    }
    return emoji_map.get(category, '🔧')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new game session"""
    session_id = str(uuid.uuid4())
    session['game_id'] = session_id
    game = get_or_create_game(session_id)
    
    return jsonify({
        'session_id': session_id,
        'message': 'New game session created',
        'starting_concepts': list(game.game_state.discovered_concepts)
    })

@app.route('/api/game/state')
def get_game_state():
    """Get current game state"""
    session_id = session.get('game_id')
    if not session_id:
        return jsonify({'error': 'No active session'}), 400
    
    game = get_or_create_game(session_id)
    
    # Get all discovered concepts with frontend-compatible format
    discovered_elements = []
    for concept_id in game.game_state.discovered_concepts:
        if concept_id in game.concepts:
            concept = game.concepts[concept_id]
            discovered_elements.append({
                'id': concept.id,
                'name': concept.name,
                'emoji': get_emoji_for_category(concept.category),
                'discovered': True,
                'category': concept.category,
                'rarity': get_rarity_from_difficulty(concept.difficulty),
                'description': concept.description,
                'properties': concept.properties,
                'difficulty': concept.difficulty
            })
    
    # Get available combinations
    available_combinations = []
    for combo in game.game_state.available_combinations:
        if len(combo) >= 3:  # (concept1, concept2, results)
            concept1, concept2, results = combo[0], combo[1], combo[2]
            available_combinations.append({
                'element1': concept1,
                'element2': concept2,
                'possible_results': results
            })
    
    return jsonify({
        'discovered_concepts': list(game.game_state.discovered_concepts),
        'discovered_elements': discovered_elements,
        'available_combinations': available_combinations,
        'score': game.game_state.score,
        'level': game.game_state.level,
        'total_concepts': len(game.concepts),
        'discovered_count': len(game.game_state.discovered_concepts)
    })

@app.route('/api/combine', methods=['POST'])
def combine_concepts():
    """Combine two concepts to discover new ones"""
    session_id = session.get('game_id')
    if not session_id:
        return jsonify({'error': 'No active session'}), 400
    
    data = request.json
    element1_id = data.get('element1')
    element2_id = data.get('element2')
    
    if not element1_id or not element2_id:
        return jsonify({'error': 'Two elements required'}), 400
    
    game = get_or_create_game(session_id)
    
    # Try to combine the concepts
    result = game.combine_concepts(element1_id, element2_id)
    
    if result:
        concept = game.concepts[result]
        return jsonify({
            'success': True,
            'discovered': True,
            'newElement': {
                'id': concept.id,
                'name': concept.name,
                'emoji': get_emoji_for_category(concept.category),
                'discovered': True,
                'category': concept.category,
                'rarity': get_rarity_from_difficulty(concept.difficulty),
                'description': concept.description,
                'properties': concept.properties,
                'difficulty': concept.difficulty
            },
            'message': f'Discovered {concept.name}!',
            'score': game.game_state.score,
            'level': game.game_state.level
        })
    else:
        # Check if this combination could lead to something
        potential_results = game.get_combination_results(element1_id, element2_id)
        if potential_results:
            # Missing prerequisites
            missing = []
            for potential in potential_results:
                prereqs = set(game.concepts[potential].prerequisites)
                missing_prereqs = prereqs - game.game_state.discovered_concepts
                missing.extend(missing_prereqs)
            
            if missing:
                missing_names = [game.concepts[m].name for m in missing if m in game.concepts]
                return jsonify({
                    'success': False,
                    'discovered': False,
                    'message': f'Missing prerequisites: {", ".join(missing_names[:3])}',
                    'hint': 'Try discovering the prerequisites first!'
                })
        
        return jsonify({
            'success': False,
            'discovered': False,
            'message': 'These concepts don\'t combine into anything new',
            'hint': 'Try different combinations!'
        })

@app.route('/api/concepts')
def get_all_concepts():
    """Get all concepts (discovered and undiscovered)"""
    session_id = session.get('game_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['game_id'] = session_id
    
    game = get_or_create_game(session_id)
    
    all_elements = []
    for concept_id, concept in game.concepts.items():
        all_elements.append({
            'id': concept.id,
            'name': concept.name,
            'emoji': get_emoji_for_category(concept.category),
            'discovered': concept_id in game.game_state.discovered_concepts,
            'category': concept.category,
            'rarity': get_rarity_from_difficulty(concept.difficulty),
            'description': concept.description if concept_id in game.game_state.discovered_concepts else '???',
            'properties': concept.properties if concept_id in game.game_state.discovered_concepts else {},
            'difficulty': concept.difficulty,
            'prerequisites': concept.prerequisites,
            'leads_to': concept.leads_to
        })
    
    return jsonify({
        'elements': all_elements,
        'categories': list(game.categories.keys()),
        'total': len(all_elements),
        'discovered': len([e for e in all_elements if e['discovered']])
    })

@app.route('/api/concept/<concept_id>')
def get_concept_details(concept_id):
    """Get detailed information about a specific concept"""
    session_id = session.get('game_id', str(uuid.uuid4()))
    game = get_or_create_game(session_id)
    
    if concept_id not in game.concepts:
        return jsonify({'error': 'Concept not found'}), 404
    
    concept = game.concepts[concept_id]
    is_discovered = concept_id in game.game_state.discovered_concepts
    
    # Get prerequisite and leads_to names
    prereq_info = []
    for prereq_id in concept.prerequisites:
        if prereq_id in game.concepts:
            prereq = game.concepts[prereq_id]
            prereq_info.append({
                'id': prereq_id,
                'name': prereq.name,
                'discovered': prereq_id in game.game_state.discovered_concepts
            })
    
    leads_to_info = []
    for lead_id in concept.leads_to:
        if lead_id in game.concepts:
            lead = game.concepts[lead_id]
            leads_to_info.append({
                'id': lead_id,
                'name': lead.name if is_discovered else '???',
                'discovered': lead_id in game.game_state.discovered_concepts
            })
    
    return jsonify({
        'id': concept.id,
        'name': concept.name,
        'emoji': get_emoji_for_category(concept.category),
        'discovered': is_discovered,
        'category': concept.category,
        'rarity': get_rarity_from_difficulty(concept.difficulty),
        'description': concept.description if is_discovered else 'Discover this concept to learn more',
        'properties': concept.properties if is_discovered else {},
        'difficulty': concept.difficulty,
        'prerequisites': prereq_info,
        'leads_to': leads_to_info if is_discovered else [],
        'resources': concept.resources if is_discovered else []
    })

@app.route('/api/learning-path/<target_concept>')
def get_learning_path(target_concept):
    """Get the learning path to a specific concept"""
    session_id = session.get('game_id', str(uuid.uuid4()))
    game = get_or_create_game(session_id)
    
    if target_concept not in game.concepts:
        return jsonify({'error': 'Concept not found'}), 404
    
    path = game.find_shortest_path(target_concept)
    
    path_details = []
    for concept_id in path:
        if concept_id in game.concepts:
            concept = game.concepts[concept_id]
            path_details.append({
                'id': concept.id,
                'name': concept.name,
                'discovered': concept_id in game.game_state.discovered_concepts,
                'difficulty': concept.difficulty,
                'category': concept.category
            })
    
    return jsonify({
        'target': target_concept,
        'path': path_details,
        'steps': len(path_details),
        'already_discovered': target_concept in game.game_state.discovered_concepts
    })

@app.route('/api/search')
def search_concepts():
    """Search for concepts by name or description"""
    query = request.args.get('q', '').lower()
    session_id = session.get('game_id', str(uuid.uuid4()))
    game = get_or_create_game(session_id)
    
    results = []
    for concept_id, concept in game.concepts.items():
        if (query in concept.name.lower() or 
            query in concept.description.lower() or
            query in concept.category.lower()):
            results.append({
                'id': concept.id,
                'name': concept.name,
                'category': concept.category,
                'discovered': concept_id in game.game_state.discovered_concepts,
                'difficulty': concept.difficulty,
                'match_type': 'name' if query in concept.name.lower() else 'description'
            })
    
    return jsonify({
        'query': query,
        'results': results[:20],  # Limit to 20 results
        'total': len(results)
    })

@app.route('/api/statistics')
def get_statistics():
    """Get game statistics and progress"""
    session_id = session.get('game_id', str(uuid.uuid4()))
    game = get_or_create_game(session_id)
    
    # Calculate progress by category
    category_progress = {}
    for category in game.categories:
        category_concepts = [c for c in game.concepts.values() if c.category == category]
        discovered = len([c for c in category_concepts if c.id in game.game_state.discovered_concepts])
        total = len(category_concepts)
        category_progress[category] = {
            'discovered': discovered,
            'total': total,
            'percentage': (discovered / total * 100) if total > 0 else 0
        }
    
    # Calculate progress by difficulty
    difficulty_progress = {}
    for difficulty in range(1, 6):
        diff_concepts = [c for c in game.concepts.values() if c.difficulty == difficulty]
        discovered = len([c for c in diff_concepts if c.id in game.game_state.discovered_concepts])
        total = len(diff_concepts)
        difficulty_progress[difficulty] = {
            'discovered': discovered,
            'total': total,
            'percentage': (discovered / total * 100) if total > 0 else 0
        }
    
    return jsonify({
        'score': game.game_state.score,
        'level': game.game_state.level,
        'total_concepts': len(game.concepts),
        'discovered_concepts': len(game.game_state.discovered_concepts),
        'overall_progress': (len(game.game_state.discovered_concepts) / len(game.concepts) * 100),
        'category_progress': category_progress,
        'difficulty_progress': difficulty_progress,
        'available_combinations': len(game.game_state.available_combinations)
    })

@app.route('/api/reset', methods=['POST'])
def reset_game():
    """Reset the game to initial state"""
    session_id = session.get('game_id')
    if session_id and session_id in game_instances:
        del game_instances[session_id]
    
    new_session_id = str(uuid.uuid4())
    session['game_id'] = new_session_id
    game = get_or_create_game(new_session_id)
    
    return jsonify({
        'message': 'Game reset successfully',
        'session_id': new_session_id,
        'starting_concepts': list(game.game_state.discovered_concepts)
    })

@app.route('/api/save', methods=['POST'])
def save_game():
    """Save current game state"""
    session_id = session.get('game_id')
    if not session_id:
        return jsonify({'error': 'No active session'}), 400
    
    game = get_or_create_game(session_id)
    
    save_data = {
        'session_id': session_id,
        'discovered_concepts': list(game.game_state.discovered_concepts),
        'score': game.game_state.score,
        'level': game.game_state.level,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file (you could also save to database)
    save_id = str(uuid.uuid4())[:8]
    filename = f'saves/game_save_{save_id}.json'
    os.makedirs('saves', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(save_data, f)
    
    return jsonify({
        'save_id': save_id,
        'message': 'Game saved successfully'
    })

@app.route('/api/load/<save_id>', methods=['POST'])
def load_game(save_id):
    """Load a saved game state"""
    filename = f'saves/game_save_{save_id}.json'
    
    if not os.path.exists(filename):
        return jsonify({'error': 'Save not found'}), 404
    
    with open(filename, 'r') as f:
        save_data = json.load(f)
    
    session_id = save_data['session_id']
    session['game_id'] = session_id
    
    game = get_or_create_game(session_id)
    game.game_state.discovered_concepts = set(save_data['discovered_concepts'])
    game.game_state.score = save_data['score']
    game.game_state.level = save_data['level']
    
    # Mark concepts as discovered
    for concept_id in game.game_state.discovered_concepts:
        if concept_id in game.concepts:
            game.concepts[concept_id].discovered = True
    
    game.update_available_combinations()
    
    return jsonify({
        'message': 'Game loaded successfully',
        'score': game.game_state.score,
        'level': game.game_state.level,
        'discovered_concepts': len(game.game_state.discovered_concepts)
    })

@app.route('/api/hint', methods=['POST'])
def get_hint():
    """Get a hint for what to combine next"""
    session_id = session.get('game_id')
    if not session_id:
        return jsonify({'error': 'No active session'}), 400
    
    game = get_or_create_game(session_id)
    
    if game.game_state.available_combinations:
        # Get a combination that leads to the easiest undiscovered concept
        best_combo = None
        min_difficulty = 6
        
        for combo in game.game_state.available_combinations[:10]:  # Check first 10 to avoid long search
            concept1, concept2, results = combo[0], combo[1], combo[2]
            for result_id in results:
                if result_id in game.concepts:
                    difficulty = game.concepts[result_id].difficulty
                    if difficulty < min_difficulty:
                        min_difficulty = difficulty
                        best_combo = (concept1, concept2, result_id)
        
        if best_combo:
            c1_name = game.concepts[best_combo[0]].name
            c2_name = game.concepts[best_combo[1]].name
            result_name = game.concepts[best_combo[2]].name
            
            return jsonify({
                'hint': f'Try combining {c1_name} with {c2_name}',
                'difficulty': min_difficulty,
                'potential_discovery': result_name
            })
    
    return jsonify({
        'hint': 'Keep exploring! Try combining concepts from different categories.',
        'tip': 'Mathematical foundations often combine with architectures to create new techniques.'
    })

@app.route('/api/docs/<filename>')
def get_documentation(filename):
    """Serve markdown documentation files"""
    # Define the path to the documentation files - pointing to /inf-craft/docs
    docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'docs')
    
    # Validate filename to prevent directory traversal
    allowed_files = [
        'speed-wins.md', 
        'modular_optimization_roadmap.md', 
        'game_based_learning_guide.md',
        'resonance.md',
        'models.md',
        'paper.md',
        'dimensions.md'
    ]
    if filename not in allowed_files:
        return jsonify({'error': 'File not found'}), 404
    
    file_path = os.path.join(docs_path, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)