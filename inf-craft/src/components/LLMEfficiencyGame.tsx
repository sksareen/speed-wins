import React, { useState, useEffect, useRef } from 'react';
import { 
  Brain, Cpu, Zap, Search, BookOpen, Trophy, 
  Volume2, VolumeX, Star, RefreshCw, Save, 
  Upload, HelpCircle, TrendingUp, Layers, GitBranch
} from 'lucide-react';

// Type definitions
interface Element {
  id: string;
  name: string;
  emoji: string;
  discovered: boolean;
  category: string;
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
  description: string;
  properties?: Record<string, any>;
  difficulty?: number;
}

interface WorkspaceElement extends Element {
  x: number;
  y: number;
  key: string;
}

interface CombineResult {
  success: boolean;
  discovered: boolean;
  newElement?: Element;
  message: string;
  hint?: string;
  score?: number;
  level?: number;
}

interface GameStats {
  score: number;
  level: number;
  total_concepts: number;
  discovered_concepts: number;
  overall_progress: number;
  category_progress: Record<string, any>;
  difficulty_progress: Record<string, any>;
}

const API_BASE = 'http://localhost:5001/api';

const LLMEfficiencyGame: React.FC = () => {
  const [elements, setElements] = useState<Element[]>([]);
  const [workspaceElements, setWorkspaceElements] = useState<WorkspaceElement[]>([]);
  const [draggedElement, setDraggedElement] = useState<Element | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [combineMessage, setCombineMessage] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [gameStats, setGameStats] = useState<GameStats | null>(null);
  const [showTooltip, setShowTooltip] = useState<string | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<'discovered' | 'workspace' | 'stats'>('discovered');
  const [showHint, setShowHint] = useState(false);
  const [hint, setHint] = useState<string>('');
  const workspaceRef = useRef<HTMLDivElement>(null);

  const rarityColors = {
    common: 'bg-gray-100 border-gray-300 text-gray-800',
    uncommon: 'bg-green-50 border-green-300 text-green-800',
    rare: 'bg-blue-50 border-blue-300 text-blue-800',
    epic: 'bg-purple-50 border-purple-300 text-purple-800',
    legendary: 'bg-yellow-50 border-yellow-400 text-yellow-800'
  };

  const categoryIcons: Record<string, React.ReactNode> = {
    mathematics: <BookOpen className="w-4 h-4" />,
    deep_learning: <Brain className="w-4 h-4" />,
    attention: <Zap className="w-4 h-4" />,
    optimization: <TrendingUp className="w-4 h-4" />,
    hardware: <Cpu className="w-4 h-4" />,
    distributed: <GitBranch className="w-4 h-4" />,
    production: <Layers className="w-4 h-4" />
  };

  // Initialize game session
  useEffect(() => {
    initializeGame();
  }, []);

  const initializeGame = async () => {
    try {
      // Create new session
      const sessionRes = await fetch(`${API_BASE}/session`, {
        method: 'POST',
        credentials: 'include'
      });
      const sessionData = await sessionRes.json();
      setSessionId(sessionData.session_id);

      // Load game state
      await loadGameState();
    } catch (error) {
      console.error('Failed to initialize game:', error);
    }
  };

  const loadGameState = async () => {
    try {
      // Get all concepts
      const conceptsRes = await fetch(`${API_BASE}/concepts`, {
        credentials: 'include'
      });
      const conceptsData = await conceptsRes.json();
      setElements(conceptsData.elements.filter((e: Element) => e.discovered));

      // Get statistics
      const statsRes = await fetch(`${API_BASE}/statistics`, {
        credentials: 'include'
      });
      const statsData = await statsRes.json();
      setGameStats(statsData);
    } catch (error) {
      console.error('Failed to load game state:', error);
    }
  };

  const handleDragStart = (e: React.DragEvent, element: Element) => {
    setDraggedElement(element);
    if (soundEnabled) {
      // Play drag sound
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (!draggedElement || !workspaceRef.current) return;

    const rect = workspaceRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check for combination with existing elements
    const targetElement = workspaceElements.find(el => {
      const distance = Math.sqrt(Math.pow(el.x - x, 2) + Math.pow(el.y - y, 2));
      return distance < 50;
    });

    if (targetElement && targetElement.id !== draggedElement.id) {
      await combineElements(draggedElement, targetElement);
    } else {
      // Add to workspace
      const newWorkspaceElement: WorkspaceElement = {
        ...draggedElement,
        x,
        y,
        key: `${draggedElement.id}-${Date.now()}`
      };
      setWorkspaceElements([...workspaceElements, newWorkspaceElement]);
    }

    setDraggedElement(null);
  };

  const combineElements = async (element1: Element, element2: Element) => {
    setIsGenerating(true);
    setCombineMessage('Combining concepts...');

    try {
      const response = await fetch(`${API_BASE}/combine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          element1: element1.id,
          element2: element2.id
        })
      });

      const result: CombineResult = await response.json();

      if (result.success && result.newElement) {
        // Add new element to discovered list
        setElements(prev => {
          const exists = prev.some(el => el.id === result.newElement!.id);
          if (!exists) {
            return [...prev, result.newElement!];
          }
          return prev;
        });

        setCombineMessage(`🎉 Discovered: ${result.newElement.name}!`);
        
        // Update stats
        if (result.score !== undefined) {
          setGameStats(prev => prev ? {
            ...prev,
            score: result.score!,
            level: result.level!,
            discovered_concepts: prev.discovered_concepts + 1
          } : null);
        }

        // Add to workspace
        const newWorkspaceElement: WorkspaceElement = {
          ...result.newElement,
          x: Math.random() * 400 + 100,
          y: Math.random() * 300 + 100,
          key: `${result.newElement.id}-${Date.now()}`
        };
        setWorkspaceElements(prev => [...prev, newWorkspaceElement]);

        if (soundEnabled) {
          // Play success sound
        }
      } else {
        setCombineMessage(result.message);
        if (result.hint) {
          setHint(result.hint);
        }
      }
    } catch (error) {
      console.error('Combination failed:', error);
      setCombineMessage('Failed to combine elements');
    }

    setIsGenerating(false);
    setTimeout(() => setCombineMessage(''), 3000);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const getHintFromAPI = async () => {
    try {
      const response = await fetch(`${API_BASE}/hint`, {
        method: 'POST',
        credentials: 'include'
      });
      const data = await response.json();
      setHint(data.hint || data.tip);
      setShowHint(true);
    } catch (error) {
      console.error('Failed to get hint:', error);
    }
  };

  const resetGame = async () => {
    if (!confirm('Are you sure you want to reset your progress?')) return;

    try {
      await fetch(`${API_BASE}/reset`, {
        method: 'POST',
        credentials: 'include'
      });
      await initializeGame();
      setWorkspaceElements([]);
    } catch (error) {
      console.error('Failed to reset game:', error);
    }
  };

  const filteredElements = elements.filter(el => {
    const matchesSearch = el.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || el.category === selectedCategory;
    return matchesSearch && matchesCategory && el.discovered;
  });

  const categories = Array.from(new Set(elements.map(el => el.category)));

  return (
    <div className="h-screen bg-gray-950 text-white flex">
      {/* Left Panel - Discovered Elements */}
      <div className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            LLM Efficiency Lab
          </h2>
          
          {/* Stats Bar */}
          {gameStats && (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-gray-800 rounded px-2 py-1">
                <span className="text-gray-400">Score:</span> 
                <span className="ml-1 font-bold text-yellow-400">{gameStats.score}</span>
              </div>
              <div className="bg-gray-800 rounded px-2 py-1">
                <span className="text-gray-400">Level:</span> 
                <span className="ml-1 font-bold text-green-400">{gameStats.level}</span>
              </div>
              <div className="bg-gray-800 rounded px-2 py-1 col-span-2">
                <span className="text-gray-400">Progress:</span> 
                <span className="ml-1 font-bold text-blue-400">
                  {gameStats.discovered_concepts}/{gameStats.total_concepts} ({Math.round(gameStats.overall_progress)}%)
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Search and Filter */}
        <div className="p-3 space-y-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search concepts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-2 bg-gray-800 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            <option value="all">All Categories</option>
            {categories.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>

        {/* Elements List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {filteredElements.map(element => (
            <div
              key={element.id}
              draggable
              onDragStart={(e) => handleDragStart(e, element)}
              className={`p-3 rounded-lg border-2 cursor-move transition-all hover:scale-105 ${rarityColors[element.rarity]}`}
              onMouseEnter={() => setShowTooltip(element.id)}
              onMouseLeave={() => setShowTooltip(null)}
            >
              <div className="flex items-center gap-2">
                <span className="text-xl">{element.emoji}</span>
                <div className="flex-1">
                  <div className="font-semibold">{element.name}</div>
                  <div className="text-xs opacity-75 flex items-center gap-1">
                    {categoryIcons[element.category] || <Layers className="w-3 h-3" />}
                    {element.category}
                    {element.difficulty && (
                      <span className="ml-1">
                        {'⭐'.repeat(element.difficulty)}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              
              {showTooltip === element.id && (
                <div className="mt-2 p-2 bg-black/50 rounded text-xs">
                  {element.description}
                  {element.properties && Object.keys(element.properties).length > 0 && (
                    <div className="mt-1 text-gray-400">
                      {Object.entries(element.properties).slice(0, 2).map(([key, value]) => (
                        <div key={key}>{key}: {JSON.stringify(value)}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="p-3 border-t border-gray-800 flex gap-2">
          <button
            onClick={getHintFromAPI}
            className="flex-1 px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-1"
          >
            <HelpCircle className="w-4 h-4" />
            Hint
          </button>
          <button
            onClick={resetGame}
            className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-1"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col">
        {/* Workspace Header */}
        <div className="p-4 bg-gray-900 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h3 className="text-lg font-semibold">Combination Workspace</h3>
            {combineMessage && (
              <div className={`px-3 py-1 rounded-lg text-sm ${
                combineMessage.includes('🎉') ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'
              }`}>
                {combineMessage}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setWorkspaceElements([])}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            >
              Clear Workspace
            </button>
          </div>
        </div>

        {/* Workspace Area */}
        <div
          ref={workspaceRef}
          className="flex-1 relative bg-gray-950"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          style={{
            backgroundImage: 'radial-gradient(circle, #1a1a2e 1px, transparent 1px)',
            backgroundSize: '20px 20px'
          }}
        >
          {workspaceElements.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-600">
              <div className="text-center">
                <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-lg">Drag concepts here to combine them</p>
                <p className="text-sm mt-2">Drop concepts close together to discover new ones!</p>
              </div>
            </div>
          )}

          {workspaceElements.map((element) => (
            <div
              key={element.key}
              className={`absolute p-3 rounded-lg border-2 cursor-move transition-all hover:scale-110 hover:z-10 ${rarityColors[element.rarity]}`}
              style={{
                left: element.x - 60,
                top: element.y - 30,
                width: '120px'
              }}
              draggable
              onDragStart={(e) => handleDragStart(e, element)}
            >
              <div className="text-center">
                <div className="text-2xl mb-1">{element.emoji}</div>
                <div className="text-xs font-semibold">{element.name}</div>
              </div>
            </div>
          ))}

          {isGenerating && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
              <div className="bg-gray-800 rounded-lg p-6 flex items-center gap-3">
                <RefreshCw className="w-6 h-6 animate-spin text-purple-400" />
                <span>Discovering new concepts...</span>
              </div>
            </div>
          )}
        </div>

        {/* Hint Display */}
        {showHint && hint && (
          <div className="p-3 bg-purple-900/50 border-t border-purple-800">
            <div className="flex items-start gap-2">
              <HelpCircle className="w-5 h-5 text-purple-400 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm text-purple-200">{hint}</p>
                <button
                  onClick={() => setShowHint(false)}
                  className="text-xs text-purple-400 hover:text-purple-300 mt-1"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel - Stats and Info */}
      <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Trophy className="w-5 h-5 text-yellow-400" />
            Progress & Stats
          </h3>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {gameStats && (
            <div className="space-y-4">
              {/* Overall Progress */}
              <div className="bg-gray-800 rounded-lg p-3">
                <h4 className="text-sm font-semibold mb-2 text-gray-400">Overall Progress</h4>
                <div className="w-full bg-gray-700 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all"
                    style={{ width: `${gameStats.overall_progress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {gameStats.discovered_concepts} / {gameStats.total_concepts} concepts discovered
                </p>
              </div>

              {/* Category Progress */}
              <div className="bg-gray-800 rounded-lg p-3">
                <h4 className="text-sm font-semibold mb-2 text-gray-400">Category Progress</h4>
                <div className="space-y-2">
                  {Object.entries(gameStats.category_progress).slice(0, 5).map(([category, data]: [string, any]) => (
                    <div key={category}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="capitalize">{category.replace('_', ' ')}</span>
                        <span>{data.discovered}/{data.total}</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full transition-all"
                          style={{ width: `${data.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Difficulty Progress */}
              <div className="bg-gray-800 rounded-lg p-3">
                <h4 className="text-sm font-semibold mb-2 text-gray-400">Difficulty Progress</h4>
                <div className="space-y-2">
                  {Object.entries(gameStats.difficulty_progress).map(([difficulty, data]: [string, any]) => (
                    <div key={difficulty} className="flex items-center justify-between">
                      <span className="text-xs">{'⭐'.repeat(parseInt(difficulty))}</span>
                      <span className="text-xs">{data.discovered}/{data.total}</span>
                      <div className="flex-1 mx-2 bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-yellow-500 h-2 rounded-full transition-all"
                          style={{ width: `${data.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LLMEfficiencyGame;