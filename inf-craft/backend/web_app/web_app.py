#!/usr/bin/env python3
"""
Simple Flask web app for the LLM Efficiency Game
"""

from flask import Flask, render_template, send_from_directory, jsonify, request
import json
import os

app = Flask(__name__)

# Load concepts data
def load_concepts():
    try:
        with open('../data/llm_efficiency_concepts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"concepts": {}, "relationships": {}, "categories": {}, "difficulty_levels": {}}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/concepts')
def concepts():
    """Simple concepts view"""
    return render_template('llm_concepts_simple.html')

@app.route('/dashboard')
def dashboard():
    """Dense dashboard view"""
    return render_template('dense_llm_dashboard.html')

@app.route('/api/concepts')
def api_concepts():
    """API endpoint to get all concepts"""
    data = load_concepts()
    return jsonify(data)

@app.route('/api/concepts/<concept_id>')
def api_concept(concept_id):
    """API endpoint to get a specific concept"""
    data = load_concepts()
    if concept_id in data['concepts']:
        return jsonify(data['concepts'][concept_id])
    return jsonify({"error": "Concept not found"}), 404

@app.route('/api/categories')
def api_categories():
    """API endpoint to get all categories"""
    data = load_concepts()
    return jsonify(data['categories'])

if __name__ == '__main__':
    # Use port 9000 to avoid conflicts with AirPlay on macOS
    app.run(debug=True, host='0.0.0.0', port=9000)
