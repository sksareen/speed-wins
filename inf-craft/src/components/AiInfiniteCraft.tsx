import React, { useState, useRef, useEffect } from 'react';
import { Brain, Cpu, Database, Zap, Search, BookOpen, Trophy, Volume2, VolumeX, Star } from 'lucide-react';

// Type definitions
interface Element {
  id: string;
  name: string;
  emoji: string;
  discovered: boolean;
  category: 'fundamental' | 'architecture' | 'process' | 'mathematical' | 'application' | 'advanced';
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
  description: string;
}

interface WorkspaceElement extends Element {
  x: number;
  y: number;
}

interface Achievement {
  id: string;
  name: string;
  emoji: string;
}

interface CombineHistory {
  id: number;
  element1: string;
  element2: string;
  result: string;
  rarity: string;
  timestamp: string;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  color: string;
}

const AiInfiniteCraft: React.FC = () => {
  const [elements, setElements] = useState<Element[]>([
    { id: 'data', name: 'Data', emoji: '📊', discovered: true, category: 'fundamental', rarity: 'common', description: 'Raw information used to train AI models' },
    { id: 'algorithm', name: 'Algorithm', emoji: '⚙️', discovered: true, category: 'fundamental', rarity: 'common', description: 'Step-by-step instructions for solving problems' },
    { id: 'neural-network', name: 'Neural Network', emoji: '🧠', discovered: true, category: 'architecture', rarity: 'common', description: 'Computing system inspired by biological neural networks' },
    { id: 'training', name: 'Training', emoji: '🏋️', discovered: true, category: 'process', rarity: 'common', description: 'Process of teaching an AI model to make predictions' },
    { id: 'compute', name: 'Compute', emoji: '💻', discovered: true, category: 'fundamental', rarity: 'common', description: 'Computational resources needed for AI operations' },
    { id: 'gradient', name: 'Gradient', emoji: '📈', discovered: true, category: 'mathematical', rarity: 'common', description: 'Mathematical concept used to optimize neural networks' },
    { id: 'attention', name: 'Attention', emoji: '👁️', discovered: true, category: 'architecture', rarity: 'uncommon', description: 'Mechanism for focusing on relevant parts of input' },
    { id: 'transformer', name: 'Transformer', emoji: '🔄', discovered: false, category: 'architecture', rarity: 'rare', description: 'Revolutionary architecture using self-attention' }
  ]);
  
  const [workspaceElements, setWorkspaceElements] = useState<WorkspaceElement[]>([]);
  const [draggedElement, setDraggedElement] = useState<Element | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [combineMessage, setCombineMessage] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [achievements, setAchievements] = useState<string[]>([]);
  const [combineHistory, setCombineHistory] = useState<CombineHistory[]>([]);
  const [showTooltip, setShowTooltip] = useState<string | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [rightPanelTab, setRightPanelTab] = useState('history');
  const workspaceRef = useRef<HTMLDivElement>(null);

  const categories = ['all', 'fundamental', 'architecture', 'process', 'mathematical', 'application', 'advanced'];
  const rarityColors = {
    common: 'bg-gray-100 border-gray-400 text-gray-900 hover:bg-gray-200',
    uncommon: 'bg-green-100 border-green-500 text-green-900 hover:bg-green-200',
    rare: 'bg-blue-100 border-blue-500 text-blue-900 hover:bg-blue-200',
    epic: 'bg-purple-100 border-purple-500 text-purple-900 hover:bg-purple-200',
    legendary: 'bg-amber-100 border-amber-500 text-amber-900 hover:bg-amber-200'
  };

  const rarityBadgeColors = {
    common: 'bg-gray-200 text-gray-800',
    uncommon: 'bg-green-200 text-green-800',
    rare: 'bg-blue-200 text-blue-800',
    epic: 'bg-purple-200 text-purple-800',
    legendary: 'bg-amber-200 text-amber-800'
  };

  // Predefined combinations for offline mode
  const predefinedCombinations: Record<string, Element> = {
    'data+algorithm': { id: 'machine-learning', name: 'Machine Learning', emoji: '🤖', discovered: true, category: 'application', rarity: 'uncommon', description: 'AI systems that learn from data' },
    'neural-network+training': { id: 'deep-learning', name: 'Deep Learning', emoji: '🧠', discovered: true, category: 'architecture', rarity: 'rare', description: 'Neural networks with multiple layers' },
    'attention+neural-network': { id: 'transformer', name: 'Transformer', emoji: '🔄', discovered: true, category: 'architecture', rarity: 'epic', description: 'Revolutionary architecture using self-attention' },
    'compute+gradient': { id: 'optimization', name: 'Optimization', emoji: '📈', discovered: true, category: 'mathematical', rarity: 'uncommon', description: 'Mathematical techniques for improving performance' },
    'data+compute': { id: 'big-data', name: 'Big Data', emoji: '📊', discovered: true, category: 'fundamental', rarity: 'uncommon', description: 'Large-scale data processing and analysis' },
    'algorithm+gradient': { id: 'gradient-descent', name: 'Gradient Descent', emoji: '📉', discovered: true, category: 'mathematical', rarity: 'common', description: 'Optimization algorithm for training models' },
    'neural-network+gradient': { id: 'backpropagation', name: 'Backpropagation', emoji: '🔄', discovered: true, category: 'process', rarity: 'rare', description: 'Algorithm for training neural networks' },
    'attention+data': { id: 'natural-language-processing', name: 'Natural Language Processing', emoji: '💬', discovered: true, category: 'application', rarity: 'rare', description: 'AI for understanding human language' },
    'training+compute': { id: 'distributed-training', name: 'Distributed Training', emoji: '🌐', discovered: true, category: 'process', rarity: 'epic', description: 'Training AI models across multiple machines' },
    'algorithm+attention': { id: 'reinforcement-learning', name: 'Reinforcement Learning', emoji: '🎮', discovered: true, category: 'application', rarity: 'legendary', description: 'AI learning through trial and error' },
    // Add more combinations for common cases
    'data+training': { id: 'supervised-learning', name: 'Supervised Learning', emoji: '📚', discovered: true, category: 'process', rarity: 'common', description: 'Learning from labeled examples' },
    'algorithm+compute': { id: 'algorithmic-complexity', name: 'Algorithmic Complexity', emoji: '⚡', discovered: true, category: 'mathematical', rarity: 'uncommon', description: 'Study of computational efficiency' },
    'neural-network+data': { id: 'neural-training', name: 'Neural Training', emoji: '🎯', discovered: true, category: 'process', rarity: 'common', description: 'Training neural networks on data' },
    'gradient+training': { id: 'gradient-based-learning', name: 'Gradient-Based Learning', emoji: '📊', discovered: true, category: 'process', rarity: 'uncommon', description: 'Learning using gradient optimization' },
    'attention+gradient': { id: 'attention-optimization', name: 'Attention Optimization', emoji: '🎯', discovered: true, category: 'mathematical', rarity: 'rare', description: 'Optimizing attention mechanisms' },
    'compute+training': { id: 'computational-training', name: 'Computational Training', emoji: '💻', discovered: true, category: 'process', rarity: 'common', description: 'Training using computational resources' },
    'data+gradient': { id: 'data-optimization', name: 'Data Optimization', emoji: '📈', discovered: true, category: 'process', rarity: 'uncommon', description: 'Optimizing data processing' },
    'algorithm+training': { id: 'algorithm-training', name: 'Algorithm Training', emoji: '🏋️', discovered: true, category: 'process', rarity: 'common', description: 'Training algorithms on data' },
    'neural-network+compute': { id: 'neural-computation', name: 'Neural Computation', emoji: '🧮', discovered: true, category: 'fundamental', rarity: 'uncommon', description: 'Computational aspects of neural networks' },
    'attention+training': { id: 'attention-training', name: 'Attention Training', emoji: '👁️', discovered: true, category: 'process', rarity: 'rare', description: 'Training attention mechanisms' }
  };

  // Check for achievements
  useEffect(() => {
    checkAchievements();
  }, [elements]);

  const checkAchievements = () => {
    const discoveredCount = elements.filter(el => el.discovered).length;
    const newAchievements: Achievement[] = [];

    if (discoveredCount >= 10 && !achievements.includes('explorer')) {
      newAchievements.push({ id: 'explorer', name: 'AI Explorer', emoji: '🔍' });
    }
    if (discoveredCount >= 25 && !achievements.includes('researcher')) {
      newAchievements.push({ id: 'researcher', name: 'AI Researcher', emoji: '👨‍🔬' });
    }

    const hasLegendary = elements.some(el => el.discovered && el.rarity === 'legendary');
    if (hasLegendary && !achievements.includes('legendary-finder')) {
      newAchievements.push({ id: 'legendary-finder', name: 'Legendary Finder', emoji: '⭐' });
    }

    if (newAchievements.length > 0) {
      setAchievements(prev => [...prev, ...newAchievements.map(a => a.id)]);
      newAchievements.forEach(achievement => {
        setCombineMessage(`🏆 ${achievement.name}`);
        setTimeout(() => setCombineMessage(''), 3000);
      });
    }
  };

  const createParticles = (x: number, y: number) => {
    const newParticles = Array.from({ length: 6 }, () => ({
      id: Math.random(),
      x, y,
      vx: (Math.random() - 0.5) * 3,
      vy: (Math.random() - 0.5) * 3,
      life: 1,
      color: ['#3b82f6', '#8b5cf6', '#10b981'][Math.floor(Math.random() * 3)]
    }));
    
    setParticles(prev => [...prev, ...newParticles]);
    setTimeout(() => {
      setParticles(prev => prev.filter(p => !newParticles.includes(p)));
    }, 800);
  };

  const generateNewElement = async (element1: Element, element2: Element): Promise<Element> => {
    setIsGenerating(true);
    setCombineMessage(`🔬 ${element1.name} + ${element2.name}`);
    
    const discoveredElements = elements.filter(el => el.discovered).map(el => el.name).join(', ');
    
    // Check if API key is available
    const apiKey = import.meta.env.VITE_OPENROUTER_API_KEY;
    const useOfflineMode = !apiKey || apiKey === 'your-openrouter-api-key-here';
    
    // First, check if we have a predefined combination
    const combinationKey = `${element1.name.toLowerCase()}+${element2.name.toLowerCase()}`;
    const reverseKey = `${element2.name.toLowerCase()}+${element1.name.toLowerCase()}`;
    
    const predefinedResult = predefinedCombinations[combinationKey] || predefinedCombinations[reverseKey];
    
    if (predefinedResult) {
      // Use predefined combination
      const newElement: Element = {
        ...predefinedResult,
        id: predefinedResult.id,
        discovered: true
      };

      const exists = elements.find(el => el.id === newElement.id || el.name === newElement.name);
      if (!exists) {
        setElements(prev => [...prev, newElement]);
        setCombineHistory(prev => [{
          id: Date.now(),
          element1: element1.name,
          element2: element2.name,
          result: newElement.name,
          rarity: newElement.rarity,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev.slice(0, 19)]);
        
        setCombineMessage(`✨ ${newElement.name} ${newElement.emoji}`);
        
        if (workspaceRef.current) {
          const rect = workspaceRef.current.getBoundingClientRect();
          createParticles(rect.width / 2, rect.height / 2);
        }
        
        setIsGenerating(false);
        setTimeout(() => setCombineMessage(''), 3000);
        return newElement;
      } else {
        // Element already exists, return it anyway for the workspace
        setCombineMessage(`🔄 ${exists.name} ${exists.emoji}`);
        setIsGenerating(false);
        setTimeout(() => setCombineMessage(''), 3000);
        return exists;
      }
    }
    
    // If no predefined combination and we have AI, try to generate something new
    if (!useOfflineMode) {
      try {
        // Use OpenRouter API for truly new combinations
        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`,
            "HTTP-Referer": window.location.origin,
            "X-Title": "AI Infinite Craft"
          },
          body: JSON.stringify({
            model: "anthropic/claude-3.5-sonnet",
            max_tokens: 250,
            messages: [
              { 
                role: "user", 
                content: `AI Infinite Craft game engine. Create a logical AI/ML concept from combining these elements.

Known elements: ${discoveredElements}

Combine: "${element1.name}" + "${element2.name}"

JSON response only:
{
  "name": "New AI Concept",
  "emoji": "🤖",
  "category": "fundamental|architecture|process|mathematical|application|advanced", 
  "rarity": "common|uncommon|rare|epic|legendary",
  "description": "Brief educational description"
}

Make it more advanced than inputs when logical. Avoid duplicates.`
              }
            ]
          })
        });

        if (!response.ok) {
          throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        if (!data.choices || !data.choices[0] || !data.choices[0].message) {
          throw new Error('Invalid API response format');
        }
        
        let responseText = data.choices[0].message.content.trim().replace(/```json\s*|\s*```/g, '');
        
        let result;
        try {
          result = JSON.parse(responseText);
        } catch (e) {
          const jsonMatch = responseText.match(/\{[\s\S]*\}/);
          if (jsonMatch) result = JSON.parse(jsonMatch[0]);
          else throw new Error('Invalid response format');
        }

        if (result.name && result.emoji) {
          const newElement: Element = {
            id: result.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, ''),
            name: result.name,
            emoji: result.emoji,
            discovered: true,
            category: result.category || 'application',
            rarity: result.rarity || 'common',
            description: result.description || 'An AI concept'
          };

          const exists = elements.find(el => el.id === newElement.id || el.name === newElement.name);
          if (!exists) {
            setElements(prev => [...prev, newElement]);
            setCombineHistory(prev => [{
              id: Date.now(),
              element1: element1.name,
              element2: element2.name,
              result: newElement.name,
              rarity: newElement.rarity,
              timestamp: new Date().toLocaleTimeString()
            }, ...prev.slice(0, 19)]);
            
            setCombineMessage(`✨ ${newElement.name} ${newElement.emoji}`);
            
            if (workspaceRef.current) {
              const rect = workspaceRef.current.getBoundingClientRect();
              createParticles(rect.width / 2, rect.height / 2);
            }
            
            setIsGenerating(false);
            setTimeout(() => setCombineMessage(''), 3000);
            return newElement;
          } else {
            setCombineMessage(`🔄 ${exists.name} ${exists.emoji}`);
            setIsGenerating(false);
            setTimeout(() => setCombineMessage(''), 3000);
            return exists;
          }
        }
      } catch (error) {
        console.error('API Error:', error);
        if (error instanceof Error && error.message.includes('401')) {
          setCombineMessage('❌ Invalid API key');
        } else if (error instanceof Error && error.message.includes('API request failed')) {
          setCombineMessage('❌ API request failed');
        } else {
          setCombineMessage('❌ Combination failed');
        }
      }
    }
    
    // Fallback: create a simple combination name
    const fallbackName = `${element1.name} ${element2.name}`;
    const fallbackElement: Element = {
      id: fallbackName.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, ''),
      name: fallbackName,
      emoji: '🔗',
      discovered: true,
      category: 'application',
      rarity: 'common',
      description: `Combination of ${element1.name} and ${element2.name}`
    };

    const exists = elements.find(el => el.id === fallbackElement.id || el.name === fallbackElement.name);
    if (!exists) {
      setElements(prev => [...prev, fallbackElement]);
      setCombineHistory(prev => [{
        id: Date.now(),
        element1: element1.name,
        element2: element2.name,
        result: fallbackElement.name,
        rarity: fallbackElement.rarity,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 19)]);
    }
    
    setCombineMessage(`🔗 ${fallbackElement.name} ${fallbackElement.emoji}`);
    setIsGenerating(false);
    setTimeout(() => setCombineMessage(''), 3000);
    return exists || fallbackElement;
  };

  const handleDragStart = (e: React.DragEvent, element: Element) => {
    setDraggedElement(element);
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    
    if (!draggedElement || !workspaceRef.current) return;

    const rect = workspaceRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newWorkspaceElement: WorkspaceElement = {
      ...draggedElement,
      id: `workspace-${Date.now()}-${Math.random()}`,
      x, y
    };

    setWorkspaceElements(prev => [...prev, newWorkspaceElement]);
    setDraggedElement(null);
  };

  const handleWorkspaceElementDrag = (e: React.DragEvent, workspaceElement: WorkspaceElement) => {
    e.dataTransfer.setData('workspaceElement', JSON.stringify(workspaceElement));
  };

  const handleWorkspaceElementDrop = async (e: React.DragEvent, targetElement: WorkspaceElement) => {
    e.preventDefault();
    e.stopPropagation();

    const sourceData = e.dataTransfer.getData('workspaceElement');
    if (!sourceData) return;

    const sourceElement: WorkspaceElement = JSON.parse(sourceData);
    if (sourceElement.id === targetElement.id) return;

    const newElement = await generateNewElement(sourceElement, targetElement);
    
    // Always add the new element to the workspace, even if it's a duplicate
    if (newElement) {
      setWorkspaceElements(prev => {
        // Remove the two elements that were combined
        const filtered = prev.filter(el => 
          el.id !== sourceElement.id && el.id !== targetElement.id
        );
        // Add the new combined element
        return [...filtered, {
          ...newElement,
          id: `workspace-${Date.now()}-${Math.random()}`,
          x: (sourceElement.x + targetElement.x) / 2,
          y: (sourceElement.y + targetElement.y) / 2
        }];
      });
    }
  };

  const filteredElements = elements.filter(el => {
    if (!el.discovered) return false;
    const matchesSearch = el.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || el.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const discoveredCount = elements.filter(el => el.discovered).length;
  const stats = {
    common: elements.filter(el => el.discovered && el.rarity === 'common').length,
    uncommon: elements.filter(el => el.discovered && el.rarity === 'uncommon').length,
    rare: elements.filter(el => el.discovered && el.rarity === 'rare').length,
    epic: elements.filter(el => el.discovered && el.rarity === 'epic').length,
    legendary: elements.filter(el => el.discovered && el.rarity === 'legendary').length
  };

  return (
    <div className="h-screen bg-slate-900 text-slate-100 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800 border-b border-slate-700 px-4 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-indigo-400" />
            <h1 className="text-xl font-bold text-slate-100">
              AI Infinite Craft
            </h1>
            <div className="text-sm text-slate-300">
              {discoveredCount} elements
            </div>
          </div>
          
          <div className="flex items-center gap-2 text-xs">
            {Object.entries(stats).map(([rarity, count]) => count > 0 && (
              <div key={rarity} className={`${rarityBadgeColors[rarity as keyof typeof rarityBadgeColors]} px-2 py-1 rounded-full text-xs font-medium`}>
                {rarity[0].toUpperCase()}: {count}
              </div>
            ))}
            
            {/* API Status Indicator */}
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${
              import.meta.env.VITE_OPENROUTER_API_KEY && import.meta.env.VITE_OPENROUTER_API_KEY !== 'your-openrouter-api-key-here'
                ? 'bg-green-100 text-green-700 border border-green-200'
                : 'bg-yellow-100 text-yellow-700 border border-yellow-200'
            }`}>
              {import.meta.env.VITE_OPENROUTER_API_KEY && import.meta.env.VITE_OPENROUTER_API_KEY !== 'your-openrouter-api-key-here'
                ? '🤖 AI Mode'
                : '💡 Offline Mode'
              }
            </div>
            
            <button 
              onClick={() => setSoundEnabled(!soundEnabled)} 
              className="p-1.5 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            >
              {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </button>
            
            <button 
              onClick={() => setWorkspaceElements([])} 
              className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-medium transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
        
        {/* Status Message */}
        {combineMessage && (
          <div className="mt-2 text-center">
            <div className="inline-block bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-xs font-medium border border-indigo-200">
              {combineMessage}
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex gap-3 p-3 overflow-hidden">
        {/* Left Panel - Elements */}
        <div className="w-80 bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-sm flex flex-col">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-4 h-4 text-slate-400" />
            <span className="font-semibold text-sm text-slate-100">Elements</span>
            <div className="ml-auto text-xs text-slate-400">{filteredElements.length}</div>
          </div>
          
          {/* Search & Filter */}
          <div className="space-y-2 mb-3">
            <div className="relative">
              <Search className="w-3 h-3 absolute left-2 top-1/2 transform -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-7 pr-2 py-1.5 border border-slate-600 rounded-lg text-xs text-slate-100 placeholder-slate-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 outline-none bg-slate-700"
              />
            </div>
            
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full px-2 py-1.5 border border-slate-600 rounded-lg text-xs text-slate-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 outline-none bg-slate-700"
              aria-label="Filter by category"
            >
              {categories.map(cat => (
                <option key={cat} value={cat}>
                  {cat.charAt(0).toUpperCase() + cat.slice(1)}
                </option>
              ))}
            </select>
          </div>
          
          {/* Elements Grid */}
          <div className="flex-1 overflow-y-auto">
            <div className="grid grid-cols-2 gap-1.5">
              {filteredElements.map(element => (
                <div
                  key={element.id}
                  draggable
                  onDragStart={(e) => handleDragStart(e, element)}
                  onMouseEnter={() => setShowTooltip(element.id)}
                  onMouseLeave={() => setShowTooltip(null)}
                  className={`${rarityColors[element.rarity]} p-2 rounded-lg border cursor-grab active:cursor-grabbing hover:shadow-md transition-all relative group`}
                >
                  <div className="flex flex-col items-center text-center">
                    <span className="text-lg mb-1">{element.emoji}</span>
                    <div className="text-xs font-medium leading-tight">{element.name}</div>
                    <div className="text-xs opacity-70 capitalize">{element.rarity[0]}</div>
                  </div>
                  
                  {/* Tooltip */}
                  {showTooltip === element.id && (
                    <div className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-1 p-2 bg-slate-800 text-slate-100 border border-slate-600 rounded-lg shadow-xl w-48 text-xs">
                      <div className="font-bold mb-1">{element.name}</div>
                      <div className="text-slate-300 text-xs">{element.description}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Center - Workspace */}
        <div className="flex-1 bg-slate-950 border border-slate-700 rounded-lg border-dashed relative overflow-hidden">
          <div className="absolute top-3 left-3 z-10">
            <h2 className="text-sm font-semibold flex items-center text-slate-300">
              <Cpu className="w-4 h-4 mr-1" />
              Laboratory
            </h2>
          </div>

          <div
            ref={workspaceRef}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="w-full h-full relative"
          >
            {/* Particles */}
            {particles.map(particle => (
              <div
                key={particle.id}
                className="absolute w-1.5 h-1.5 rounded-full pointer-events-none animate-ping"
                style={{
                  left: particle.x,
                  top: particle.y,
                  backgroundColor: particle.color,
                  animationDuration: '0.8s'
                }}
              />
            ))}

            {workspaceElements.map(element => (
              <div
                key={element.id}
                draggable
                onDragStart={(e) => handleWorkspaceElementDrag(e, element)}
                onDragOver={handleDragOver}
                onDrop={(e) => handleWorkspaceElementDrop(e, element)}
                className={`absolute ${rarityColors[element.rarity]} p-2 rounded-lg border cursor-grab active:cursor-grabbing hover:shadow-lg transition-all`}
                style={{
                  left: element.x - 50,
                  top: element.y - 20,
                  width: '100px'
                }}
              >
                <div className="flex flex-col items-center text-center">
                  <span className="text-sm mb-1">{element.emoji}</span>
                  <div className="text-xs font-medium leading-tight">{element.name}</div>
                </div>
              </div>
            ))}

            {workspaceElements.length === 0 && !isGenerating && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center text-slate-400">
                  <Database className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Drag elements here to combine</p>
                </div>
              </div>
            )}

            {isGenerating && (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80">
                <div className="text-center">
                  <Zap className="w-8 h-8 mx-auto mb-2 animate-spin text-indigo-400" />
                  <p className="text-sm text-slate-300">Processing...</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel */}
        <div className="w-72 bg-white border border-gray-200 rounded-lg shadow-sm flex flex-col">
          {/* Tab Headers */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setRightPanelTab('history')}
              className={`flex-1 px-3 py-2 text-xs font-medium border-r border-gray-200 ${rightPanelTab === 'history' ? 'bg-blue-50 text-blue-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
            >
              <BookOpen className="w-3 h-3 inline mr-1" />
              History
            </button>
            <button
              onClick={() => setRightPanelTab('achievements')}
              className={`flex-1 px-3 py-2 text-xs font-medium border-r border-gray-200 ${rightPanelTab === 'achievements' ? 'bg-blue-50 text-blue-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
            >
              <Trophy className="w-3 h-3 inline mr-1" />
              Goals
            </button>
            <button
              onClick={() => setRightPanelTab('guide')}
              className={`flex-1 px-3 py-2 text-xs font-medium ${rightPanelTab === 'guide' ? 'bg-blue-50 text-blue-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
            >
              <Star className="w-3 h-3 inline mr-1" />
              Guide
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 p-3 overflow-y-auto">
            {rightPanelTab === 'history' && (
              <div className="space-y-2">
                {combineHistory.length === 0 ? (
                  <div className="text-center text-gray-400 text-xs">
                    <p>No discoveries yet</p>
                  </div>
                ) : (
                  combineHistory.map(entry => (
                    <div key={entry.id} className="bg-gray-50 border border-gray-200 p-2 rounded-lg text-xs">
                      <div className={`font-medium mb-1 ${rarityBadgeColors[entry.rarity as keyof typeof rarityBadgeColors]} px-2 py-0.5 rounded text-xs inline-block`}>
                        {entry.result}
                      </div>
                      <div className="text-gray-600 text-xs">
                        {entry.element1} + {entry.element2}
                      </div>
                      <div className="text-gray-400 text-xs">{entry.timestamp}</div>
                    </div>
                  ))
                )}
              </div>
            )}

            {rightPanelTab === 'achievements' && (
              <div className="space-y-2">
                <div className="text-xs text-gray-500 mb-2">Progress Goals:</div>
                
                <div className="space-y-1.5">
                  <div className={`p-2 rounded-lg border text-xs ${discoveredCount >= 10 ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                    <div className="flex items-center justify-between">
                      <span>🔍 AI Explorer</span>
                      <span className="text-xs">{Math.min(discoveredCount, 10)}/10</span>
                    </div>
                    <div className="text-xs text-gray-500">Discover 10 elements</div>
                  </div>
                  
                  <div className={`p-2 rounded-lg border text-xs ${discoveredCount >= 25 ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                    <div className="flex items-center justify-between">
                      <span>👨‍🔬 AI Researcher</span>
                      <span className="text-xs">{Math.min(discoveredCount, 25)}/25</span>
                    </div>
                    <div className="text-xs text-gray-500">Discover 25 elements</div>
                  </div>
                  
                  <div className={`p-2 rounded-lg border text-xs ${stats.legendary > 0 ? 'bg-yellow-50 border-yellow-200' : 'bg-gray-50 border-gray-200'}`}>
                    <div className="flex items-center justify-between">
                      <span>⭐ Legendary Finder</span>
                      <span className="text-xs">{stats.legendary > 0 ? '✓' : '✗'}</span>
                    </div>
                    <div className="text-xs text-gray-500">Find a legendary element</div>
                  </div>
                </div>
              </div>
            )}

            {rightPanelTab === 'guide' && (
              <div className="space-y-3 text-xs">
                <div>
                  <div className="font-medium text-blue-600 mb-1">🎯 How to Play:</div>
                  <div className="text-gray-600 space-y-1">
                    <p>• Drag elements from left panel to workspace</p>
                    <p>• Drop one element onto another to combine</p>
                    <p>• Discover new AI concepts and technologies</p>
                  </div>
                </div>
                
                <div>
                  <div className="font-medium text-purple-600 mb-1">⭐ Rarity System:</div>
                  <div className="space-y-1">
                    {Object.entries(rarityBadgeColors).map(([rarity, colors]) => (
                      <div key={rarity} className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded ${colors}`}></div>
                        <span className="capitalize text-gray-600">{rarity}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <div className="font-medium text-green-600 mb-1">💡 Tips:</div>
                  <div className="text-gray-600 space-y-1">
                    <p>• Try unexpected combinations</p>
                    <p>• More advanced = higher rarity</p>
                    <p>• Search and filter to find elements</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AiInfiniteCraft;
