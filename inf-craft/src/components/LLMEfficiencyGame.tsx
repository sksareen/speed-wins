import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Brain, Cpu, Zap, Search, BookOpen, Trophy, 
  Volume2, VolumeX, RefreshCw, 
  HelpCircle, TrendingUp, Layers, GitBranch,
  FileText, X
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
  // const [selectedTab] = useState<'discovered' | 'workspace' | 'stats'>('discovered');
  const [showHint, setShowHint] = useState(false);
  const [hint, setHint] = useState<string>('');
  const [showDocs, setShowDocs] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<string>('');
  const [docContent, setDocContent] = useState<string>('');
  const workspaceRef = useRef<HTMLDivElement>(null);

  const rarityColors = {
    common: 'bg-gray-100 border-gray-400 text-gray-900 hover:bg-gray-200',
    uncommon: 'bg-green-100 border-green-500 text-green-900 hover:bg-green-200',
    rare: 'bg-blue-100 border-blue-500 text-blue-900 hover:bg-blue-200',
    epic: 'bg-purple-100 border-purple-500 text-purple-900 hover:bg-purple-200',
    legendary: 'bg-amber-100 border-amber-500 text-amber-900 hover:bg-amber-200'
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

  const docFiles = [
    { name: 'Speed Wins', file: 'speed-wins.md', description: 'Optimization strategies for LLM efficiency' },
    { name: 'Modular Optimization Roadmap', file: 'modular_optimization_roadmap.md', description: 'Comprehensive roadmap for modular optimization' },
    { name: 'Game-Based Learning Guide', file: 'game_based_learning_guide.md', description: 'Guide for learning through gamification' },
    { name: 'Resonance', file: 'resonance.md', description: 'Consciousness as atmospheric tuning' },
    { name: 'Models', file: 'models.md', description: 'Intelligence scaffolding and modular AI' },
    { name: 'Paper', file: 'paper.md', description: 'Speed: consciousness as computational tuning' },
    { name: 'Dimensions', file: 'dimensions.md', description: '3D as a fold of 2D' }
  ];

  const openDocumentation = async (filename: string) => {
    try {
      const response = await fetch(`${API_BASE}/docs/${filename}`);
      if (response.ok) {
        const content = await response.text();
        setDocContent(content);
        setSelectedDoc(filename);
        setShowDocs(true);
      } else {
        console.error('Failed to load documentation');
      }
    } catch (error) {
      console.error('Error loading documentation:', error);
    }
  };

  // Initialize game session
  useEffect(() => {
    initializeGame();
  }, []);

  // Handle window resize to keep elements in bounds
  useEffect(() => {
    const handleResize = () => {
      if (!workspaceRef.current) return;
      
      const rect = workspaceRef.current.getBoundingClientRect();
      setWorkspaceElements(prev => prev.map(el => ({
        ...el,
        x: Math.max(60, Math.min(el.x, rect.width - 60)),
        y: Math.max(30, Math.min(el.y, rect.height - 30))
      })));
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const removeElementFromCanvas = (elementKey: string) => {
    setWorkspaceElements(prev => prev.filter(el => el.key !== elementKey));
  };

  const handleWorkspaceElementDrag = (e: React.DragEvent, element: WorkspaceElement) => {
    e.stopPropagation();
    setDraggedElement(element);
    
    // Store the initial mouse position relative to element
    const rect = e.currentTarget.getBoundingClientRect();
    const offsetX = e.clientX - rect.left - 60; // 60 is half width
    const offsetY = e.clientY - rect.top - 30; // 30 is half height
    
    e.dataTransfer.setData('offsetX', offsetX.toString());
    e.dataTransfer.setData('offsetY', offsetY.toString());
  };

  const moveWorkspaceElement = (elementKey: string, newX: number, newY: number) => {
    if (!workspaceRef.current) return;
    
    const rect = workspaceRef.current.getBoundingClientRect();
    const boundedX = Math.max(60, Math.min(newX, rect.width - 60));
    const boundedY = Math.max(30, Math.min(newY, rect.height - 30));
    
    setWorkspaceElements(prev => prev.map(el => 
      el.key === elementKey ? { ...el, x: boundedX, y: boundedY } : el
    ));
  };

  const initializeGame = async () => {
    try {
      // Create new session
      // const sessionRes = await fetch(`${API_BASE}/session`, {
      //   method: 'POST',
      //   credentials: 'include'
      // });
      // const sessionData = await sessionRes.json();
      // setSessionId(sessionData.session_id);

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
    e.dataTransfer.setData('text/plain', element.id);
    if (soundEnabled) {
      // Play drag sound
    }
  };

  const addElementToCanvas = (element: Element, x?: number, y?: number) => {
    if (!workspaceRef.current) return;
    
    const rect = workspaceRef.current.getBoundingClientRect();
    const canvasX = x ?? Math.random() * (rect.width - 120) + 60;
    const canvasY = y ?? Math.random() * (rect.height - 60) + 30;
    
    const newWorkspaceElement: WorkspaceElement = {
      ...element,
      x: Math.max(60, Math.min(canvasX, rect.width - 60)),
      y: Math.max(30, Math.min(canvasY, rect.height - 30)),
      key: `${element.id}-${Date.now()}-${Math.random()}`
    };
    
    setWorkspaceElements(prev => [...prev, newWorkspaceElement]);
  };

  const handleElementClick = (element: Element) => {
    addElementToCanvas(element);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (!draggedElement || !workspaceRef.current) return;

    const rect = workspaceRef.current.getBoundingClientRect();
    const offsetX = parseInt(e.dataTransfer.getData('offsetX') || '0');
    const offsetY = parseInt(e.dataTransfer.getData('offsetY') || '0');
    const x = e.clientX - rect.left - offsetX;
    const y = e.clientY - rect.top - offsetY;

    // Check if this is a workspace element being moved
    const draggedWorkspaceElement = workspaceElements.find(el => el.key === (draggedElement as WorkspaceElement).key);
    
    if (draggedWorkspaceElement) {
      // Check for combination with other workspace elements
      const targetElement = workspaceElements.find(el => {
        if (el.key === draggedWorkspaceElement.key) return false;
        const distance = Math.sqrt(Math.pow(el.x - x, 2) + Math.pow(el.y - y, 2));
        return distance < 80;
      });

      if (targetElement) {
        await combineElements(draggedElement, targetElement);
      } else {
        // Just move the element
        moveWorkspaceElement(draggedWorkspaceElement.key, x, y);
      }
    } else {
      // This is a new element from the sidebar
      // Check for combination with existing elements
      const targetElement = workspaceElements.find(el => {
        const distance = Math.sqrt(Math.pow(el.x - x, 2) + Math.pow(el.y - y, 2));
        return distance < 80;
      });

      if (targetElement) {
        await combineElements(draggedElement, targetElement);
      } else {
        // Add to workspace at drop position
        addElementToCanvas(draggedElement, x, y);
      }
    }

    setDraggedElement(null);
  };

  const combineElements = async (element1: Element, element2: Element) => {
    setIsGenerating(true);
    setCombineMessage('Combining concepts...');

    // Find workspace elements being combined
    const workspaceEl1 = workspaceElements.find(el => el.id === element1.id);
    const workspaceEl2 = workspaceElements.find(el => el.id === element2.id);

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
        
        // Refresh complete stats from backend to ensure accuracy
        await loadGameState();

        // Remove the two elements that were combined and add the new one
        const combinationX = workspaceEl1 && workspaceEl2 ? 
          (workspaceEl1.x + workspaceEl2.x) / 2 : 
          Math.random() * 400 + 200;
        const combinationY = workspaceEl1 && workspaceEl2 ? 
          (workspaceEl1.y + workspaceEl2.y) / 2 : 
          Math.random() * 300 + 150;

        setWorkspaceElements(prev => {
          // Remove the combined elements
          const filtered = prev.filter(el => 
            !(el.id === element1.id && el.key === workspaceEl1?.key) &&
            !(el.id === element2.id && el.key === workspaceEl2?.key)
          );
          
          // Add the new element at combination position
          const newElement: WorkspaceElement = {
            ...result.newElement!,
            x: combinationX,
            y: combinationY,
            key: `${result.newElement!.id}-${Date.now()}-${Math.random()}`
          };
          
          return [...filtered, newElement];
        });

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
    <div className="h-screen bg-slate-900 text-slate-100 flex">
      {/* Left Panel - Discovered Elements */}
      <div className="w-80 bg-slate-800 border-r border-slate-700 flex flex-col">
        <div className="p-4 border-b border-slate-700">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Brain className="w-5 h-5 text-indigo-400" />
            LLM Infinite Craft
          </h2>
        </div>

        {/* Search and Filter */}
        <div className="p-3 space-y-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search concepts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-2 bg-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-slate-600"
            />
          </div>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full px-3 py-2 bg-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-slate-600"
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
              onClick={() => handleElementClick(element)}
              className={`p-3 rounded-lg border-2 cursor-pointer transition-all hover:scale-105 active:scale-95 ${rarityColors[element.rarity]}`}
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
                <div className="mt-2 p-2 bg-slate-800/90 rounded text-xs text-slate-100 border border-slate-600">
                  {element.description}
                  {element.properties && Object.keys(element.properties).length > 0 && (
                    <div className="mt-1 text-slate-300">
                      {Object.entries(element.properties).slice(0, 2).map(([key, value]) => (
                        <div key={key}>{key}: {JSON.stringify(value)}</div>
                      ))}
                    </div>
                  )}
                  <div className="mt-1 text-emerald-400 text-xs">
                    Click to add to canvas • Drag to position
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="p-3 border-t border-slate-700 flex gap-2">
          <button
            onClick={getHintFromAPI}
            className="flex-1 px-3 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-1 text-white"
          >
            <HelpCircle className="w-4 h-4" />
            Hint
          </button>
          <button
            onClick={resetGame}
            className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-1 text-white"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="px-3 py-2 bg-slate-600 hover:bg-slate-500 rounded-lg transition-colors text-white"
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col">
        {/* Workspace Header */}
        <div className="p-4 bg-slate-800 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h3 className="text-lg font-semibold text-slate-100">Combination Workspace</h3>
            {combineMessage && (
              <div className={`px-3 py-1 rounded-lg text-sm ${
                combineMessage.includes('🎉') ? 'bg-emerald-900 text-emerald-200 border border-emerald-700' : 'bg-amber-900 text-amber-200 border border-amber-700'
              }`}>
                {combineMessage}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowDocs(true)}
              className="px-3 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 text-white"
            >
              <FileText className="w-4 h-4" />
              Docs
            </button>
            <button
              onClick={() => setWorkspaceElements([])}
              className="px-3 py-1 bg-slate-600 hover:bg-slate-500 rounded text-sm transition-colors text-white"
            >
              Clear Workspace
            </button>
          </div>
        </div>

        {/* Workspace Area */}
        <div
          ref={workspaceRef}
          className="flex-1 relative bg-slate-950"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          style={{
            backgroundImage: 'radial-gradient(circle, #334155 1px, transparent 1px)',
            backgroundSize: '20px 20px'
          }}
        >
          {workspaceElements.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-slate-500">
              <div className="text-center">
                <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-lg text-slate-300">Click or drag concepts here to combine them</p>
                <p className="text-sm mt-2 text-slate-400">• Click concepts in sidebar to add them</p>
                <p className="text-sm text-slate-400">• Drag elements close together to discover new ones</p>
                <p className="text-sm text-slate-400">• Hover over elements to remove them</p>
              </div>
            </div>
          )}

          {workspaceElements.map((element) => (
            <div
              key={element.key}
              className={`absolute p-3 rounded-lg border-2 cursor-move transition-all hover:scale-110 hover:z-10 group ${rarityColors[element.rarity]}`}
              style={{
                left: element.x - 60,
                top: element.y - 30,
                width: '120px'
              }}
              draggable
              onDragStart={(e) => handleWorkspaceElementDrag(e, element)}
            >
              {/* Remove button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeElementFromCanvas(element.key);
                }}
                className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 hover:bg-red-600 rounded-full text-white text-xs opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
              >
                ×
              </button>
              
              <div className="text-center">
                <div className="text-2xl mb-1">{element.emoji}</div>
                <div className="text-xs font-semibold leading-tight">{element.name}</div>
              </div>
            </div>
          ))}

          {isGenerating && (
            <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center">
              <div className="bg-slate-800 rounded-lg p-6 flex items-center gap-3 border border-slate-600">
                <RefreshCw className="w-6 h-6 animate-spin text-indigo-400" />
                <span className="text-slate-100">Discovering new concepts...</span>
              </div>
            </div>
          )}
        </div>

        {/* Hint Display */}
        {showHint && hint && (
          <div className="p-3 bg-indigo-900/50 border-t border-indigo-700">
            <div className="flex items-start gap-2">
              <HelpCircle className="w-5 h-5 text-indigo-400 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm text-indigo-200">{hint}</p>
                <button
                  onClick={() => setShowHint(false)}
                  className="text-xs text-indigo-400 hover:text-indigo-300 mt-1"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel - Stats and Info */}
      <div className="w-80 bg-slate-800 border-l border-slate-700 flex flex-col">
        <div className="p-4 border-b border-slate-700">
          <h3 className="text-lg font-semibold flex items-center gap-2 text-slate-100">
            <Trophy className="w-5 h-5 text-amber-400" />
            Progress & Stats
          </h3>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {gameStats && (
            <div className="space-y-4">
              {/* Game Stats */}
              <div className="bg-slate-700 rounded-lg p-3 border border-slate-600">
                <h4 className="text-sm font-semibold mb-2 text-slate-300">Game Stats</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Score:</span> 
                    <span className="font-bold text-amber-400">{gameStats.score}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Level:</span> 
                    <span className="font-bold text-emerald-400">{gameStats.level}</span>
                  </div>
                </div>
              </div>

              {/* Overall Progress */}
              <div className="bg-slate-700 rounded-lg p-3 border border-slate-600">
                <h4 className="text-sm font-semibold mb-2 text-slate-300">Overall Progress</h4>
                <div className="w-full bg-slate-600 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-indigo-500 to-sky-500 h-3 rounded-full transition-all"
                    style={{ width: `${gameStats.overall_progress}%` }}
                  />
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  {gameStats.discovered_concepts} / {gameStats.total_concepts} concepts discovered
                </p>
              </div>

              {/* Category Progress */}
              <div className="bg-slate-700 rounded-lg p-3 border border-slate-600">
                <h4 className="text-sm font-semibold mb-2 text-slate-300">Category Progress</h4>
                <div className="space-y-2">
                  {Object.entries(gameStats.category_progress).slice(0, 5).map(([category, data]: [string, any]) => (
                    <div key={category}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="capitalize text-slate-200">{category.replace('_', ' ')}</span>
                        <span className="text-slate-300">{data.discovered}/{data.total}</span>
                      </div>
                      <div className="w-full bg-slate-600 rounded-full h-2">
                        <div
                          className="bg-emerald-500 h-2 rounded-full transition-all"
                          style={{ width: `${data.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Difficulty Progress */}
              <div className="bg-slate-700 rounded-lg p-3 border border-slate-600">
                <h4 className="text-sm font-semibold mb-2 text-slate-300">Difficulty Progress</h4>
                <div className="space-y-2">
                  {Object.entries(gameStats.difficulty_progress).map(([difficulty, data]: [string, any]) => (
                    <div key={difficulty} className="flex items-center justify-between">
                      <span className="text-xs text-slate-200">{'⭐'.repeat(parseInt(difficulty))}</span>
                      <span className="text-xs text-slate-300">{data.discovered}/{data.total}</span>
                      <div className="flex-1 mx-2 bg-slate-600 rounded-full h-2">
                        <div
                          className="bg-amber-500 h-2 rounded-full transition-all"
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

              {/* Documentation Modal */}
        {showDocs && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-slate-800 rounded-lg w-full h-full max-w-6xl max-h-[90vh] flex flex-col border border-slate-600">
              {/* Modal Header */}
              <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-indigo-400" />
                  Documentation
                </h3>
                <button
                  onClick={() => setShowDocs(false)}
                  className="p-1 hover:bg-slate-700 rounded transition-colors"
                  aria-label="Close documentation"
                >
                  <X className="w-5 h-5 text-slate-400" />
                </button>
              </div>

              {/* Modal Content */}
              <div className="flex-1 flex min-h-0">
                {/* Sidebar with doc list */}
                <div className="w-64 border-r border-slate-700 p-4 overflow-y-auto">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">Available Documents</h4>
                  <div className="space-y-2">
                    {docFiles.map((doc) => (
                      <button
                        key={doc.file}
                        onClick={() => openDocumentation(doc.file)}
                        className={`w-full text-left p-3 rounded-lg transition-colors ${
                          selectedDoc === doc.file 
                            ? 'bg-indigo-600 text-white' 
                            : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                        }`}
                      >
                        <div className="font-medium text-sm">{doc.name}</div>
                        <div className="text-xs opacity-75 mt-1">{doc.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Document content */}
                <div className="flex-1 p-6 overflow-y-auto">
                  {docContent ? (
                                         <div className="prose prose-invert prose-slate max-w-none text-slate-200">
                       <ReactMarkdown 
                         remarkPlugins={[remarkGfm]}
                         components={{
                          h1: ({children}) => <h1 className="text-3xl font-bold text-slate-100 mb-4 border-b border-slate-600 pb-2">{children}</h1>,
                          h2: ({children}) => <h2 className="text-2xl font-bold text-slate-100 mb-3 mt-6">{children}</h2>,
                          h3: ({children}) => <h3 className="text-xl font-semibold text-slate-100 mb-2 mt-4">{children}</h3>,
                          h4: ({children}) => <h4 className="text-lg font-semibold text-slate-100 mb-2 mt-3">{children}</h4>,
                          p: ({children}) => <p className="text-slate-200 mb-3 leading-relaxed">{children}</p>,
                          ul: ({children}) => <ul className="list-disc list-inside text-slate-200 mb-3 space-y-1">{children}</ul>,
                          ol: ({children}) => <ol className="list-decimal list-inside text-slate-200 mb-3 space-y-1">{children}</ol>,
                          li: ({children}) => <li className="text-slate-200">{children}</li>,
                          code: ({children, className}) => {
                            const isInline = !className;
                            if (isInline) {
                              return <code className="bg-slate-700 text-emerald-300 px-1 py-0.5 rounded text-sm">{children}</code>;
                            }
                            return (
                              <pre className="bg-slate-900 text-slate-200 p-4 rounded-lg overflow-x-auto mb-4">
                                <code className="text-sm">{children}</code>
                              </pre>
                            );
                          },
                          blockquote: ({children}) => <blockquote className="border-l-4 border-indigo-500 pl-4 italic text-slate-300 mb-3">{children}</blockquote>,
                          strong: ({children}) => <strong className="font-bold text-slate-100">{children}</strong>,
                          em: ({children}) => <em className="italic text-slate-300">{children}</em>,
                          a: ({children, href}) => <a href={href} className="text-indigo-400 hover:text-indigo-300 underline">{children}</a>,
                          table: ({children}) => <div className="overflow-x-auto mb-4"><table className="min-w-full border border-slate-600">{children}</table></div>,
                          th: ({children}) => <th className="border border-slate-600 px-3 py-2 text-left bg-slate-700 text-slate-100">{children}</th>,
                          td: ({children}) => <td className="border border-slate-600 px-3 py-2 text-slate-200">{children}</td>,
                        }}
                      >
                        {docContent}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-slate-400">
                      <div className="text-center">
                        <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>Select a document to view its content</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
    </div>
  );
};

export default LLMEfficiencyGame;