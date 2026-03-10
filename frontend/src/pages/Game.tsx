/**
 * Game.tsx — Main Dashboard Page
 * ================================
 * The primary play & train view. Composes three columns:
 *
 *   Left column:   ConfigPanel (data gen + training controls)
 *                   Metrics (live loss chart)
 *                   ModelRegistry (model arena with Red/White assignment)
 *
 *   Center column: Interactive Board (drag-and-drop checkers)
 *                   Mode label + epsilon slider
 *                   Turn indicators (Red / White active)
 *
 *   Right column:  BrainVisualizer (CNN win probability bars)
 *                   How to Play rules
 *
 * State managed here:
 *   - boardState / currentTurn: the game grid and whose turn it is
 *   - blackModelId / whiteModelId: which AI model plays each side (null = human)
 *   - epsilon: randomness for AI move selection
 *   - cnnProbabilities: latest CNN output for the brain visualizer
 *   - gameOver: winner ID when the game ends (null while active)
 */
import { useState, useEffect } from 'react';
import { Info, RotateCcw, Play, Pause } from 'lucide-react';
import { Board } from '../components/Board';
import { ConfigPanel } from '../components/ConfigPanel';
import { BrainVisualizer } from '../components/BrainVisualizer';
import { Metrics } from '../components/Metrics';
import { ModelRegistry } from '../components/ModelRegistry';
import { GameMonitor } from '../components/GameMonitor';
import type { MaterialSnapshot } from '../components/GameMonitor';

// The main Dashboard view for playing and training
export const Game = () => {
    // Helper to generate the initial board
    const getInitialGrid = () => {
        const grid = Array(8).fill(null).map(() => Array(8).fill(0));
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 8; c++) {
                if ((r + c) % 2 !== 0) grid[r][c] = 1; // Black
            }
        }
        for (let r = 5; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                if ((r + c) % 2 !== 0) grid[r][c] = 2; // White
            }
        }
        return grid;
    };

    // Global State for the UI
    const [boardState, setBoardState] = useState<number[][]>(getInitialGrid);
    const [currentTurn, setCurrentTurn] = useState<number>(2); // White (2) moves first
    const [blackModelId, setBlackModelId] = useState<string | null>(null);
    const [whiteModelId, setWhiteModelId] = useState<string | null>(null);
    const [dualProbabilities, setDualProbabilities] = useState<{
        red_eval: { p_black: number, p_white: number } | null;
        white_eval: { p_black: number, p_white: number } | null;
    } | null>(null);
    const [gameOver, setGameOver] = useState<number | null>(null);
    const [epsilon, setEpsilon] = useState(0.0);
    const [searchDepth, setSearchDepth] = useState(1);
    const [isPlaying, setIsPlaying] = useState(false);
    const [moveHistory, setMoveHistory] = useState<MaterialSnapshot[]>([]);

    // Track material after every board state change
    useEffect(() => {
        let redScore = 0, whiteScore = 0;
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const p = boardState[r][c];
                if (p === 1) redScore += 1.0;
                else if (p === 3) redScore += 1.5;
                else if (p === 2) whiteScore += 1.0;
                else if (p === 4) whiteScore += 1.5;
            }
        }
        setMoveHistory(prev => [...prev, { redScore, whiteScore }]);
    }, [boardState]);


    const handleReset = () => {
        setBoardState(getInitialGrid());
        setCurrentTurn(2);
        setDualProbabilities(null);
        setGameOver(null);
        setIsPlaying(false); // Stop play on reset
        setMoveHistory([]);  // Clear history on reset
    };

    const handleModelAssign = (side: 'black' | 'white', modelId: string | null) => {
        if (side === 'black') setBlackModelId(modelId);
        else setWhiteModelId(modelId);
    };

    // Poll dual probabilities on every UI update or state change
    useEffect(() => {
        if (!blackModelId && !whiteModelId) {
            setDualProbabilities(null);
            return;
        }

        const fetchProbabilities = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        board_state: boardState,
                        current_turn: currentTurn,
                        red_model_id: blackModelId,
                        white_model_id: whiteModelId
                    })
                });
                const data = await response.json();
                setDualProbabilities(data);
            } catch (err) {
                console.error("Evaluate Error:", err);
            }
        };

        fetchProbabilities();
    }, [boardState, currentTurn, blackModelId, whiteModelId]);

    // Determine play mode for display
    const blackIsAI = blackModelId !== null;
    const whiteIsAI = whiteModelId !== null;
    const modeLabel = blackIsAI && whiteIsAI ? 'AI vs AI' :
        !blackIsAI && !whiteIsAI ? 'Human vs Human' :
            blackIsAI ? 'AI (Red) vs Human (White)' : 'Human (Red) vs AI (White)';

    return (
        <div className="dashboard-grid">
            {/* Left Column: Controls & Configuration */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <ConfigPanel />
                <Metrics />
                <ModelRegistry
                    blackModelId={blackModelId}
                    whiteModelId={whiteModelId}
                    onAssign={handleModelAssign}
                />
            </div>

            {/* Center Column: The Checkers Board */}
            <div className="panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', marginBottom: '1rem' }}>
                    <h2 style={{ margin: 0, borderBottom: 'none', paddingBottom: 0 }}>Interactive Board</h2>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button onClick={() => setIsPlaying(!isPlaying)} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: isPlaying ? 'var(--accent-red)' : 'var(--accent-green)', padding: '0.5rem 1rem', fontSize: '0.875rem', color: 'white' }}>
                            {isPlaying ? <><Pause size={16} /> Pause</> : <><Play size={16} /> Play</>}
                        </button>
                        <button onClick={handleReset} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: 'var(--bg-secondary)', padding: '0.5rem 1rem', fontSize: '0.875rem' }}>
                            <RotateCcw size={16} /> Reset
                        </button>
                    </div>
                </div>
                <div className="edu-note" style={{ width: '100%', marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div><strong>Mode:</strong> {modeLabel}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginLeft: 'auto', fontSize: '0.8rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <label style={{ whiteSpace: 'nowrap' }}>Depth: {searchDepth}</label>
                            <input type="range" min="1" max="8" step="1" value={searchDepth} onChange={(e) => setSearchDepth(Number(e.target.value))} style={{ width: '80px' }} />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <label style={{ whiteSpace: 'nowrap' }}>ε {epsilon.toFixed(2)}</label>
                            <input type="range" min="0" max="0.3" step="0.01" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} style={{ width: '80px' }} />
                        </div>
                    </div>
                </div>

                <Board
                    grid={boardState}
                    currentTurn={currentTurn}
                    blackModelId={blackModelId}
                    whiteModelId={whiteModelId}
                    epsilon={epsilon}
                    searchDepth={searchDepth}
                    isPlaying={isPlaying}
                    setGrid={setBoardState}
                    setCurrentTurn={setCurrentTurn}
                    gameOver={gameOver}
                    setGameOver={setGameOver}
                />

                <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <div className={`status-indicator ${currentTurn === 1 ? 'active' : ''}`} style={{ backgroundColor: 'var(--piece-red)' }} /> Red {currentTurn === 1 ? "(Active)" : ""}
                    <div className={`status-indicator ${currentTurn === 2 ? 'active' : ''}`} style={{ backgroundColor: 'var(--piece-white)' }} /> White {currentTurn === 2 ? "(Active)" : ""}
                </div>

                <GameMonitor grid={boardState} currentTurn={currentTurn} moveHistory={moveHistory} />
            </div>

            {/* Right Column: AI Brain Visualization & Rules */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <BrainVisualizer dualProbabilities={dualProbabilities} redAssigned={blackModelId !== null} whiteAssigned={whiteModelId !== null} />

                <div className="panel">
                    <h2><Info size={20} /> How to Play</h2>
                    <ul style={{
                        listStyleType: 'disc',
                        paddingLeft: '1.5rem',
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.75rem',
                        fontSize: '0.9rem'
                    }}>
                        <li><strong>Mode:</strong> Assign models to sides in the Model Arena panel. Unassigned sides are human-controlled.</li>
                        <li><strong>Turn Order:</strong> <em>White</em> always moves first.</li>
                        <li><strong>Direction:</strong> Red starts at the top and moves <em>down</em>. White starts at the bottom and moves <em>up</em>.</li>
                        <li><strong>Movement:</strong> Pieces move diagonally forward one square at a time.</li>
                        <li><strong>Jumping:</strong> If an opponent's piece is diagonally in front of you with an empty space behind it, you <em>must</em> jump it.</li>
                        <li><strong>Kings:</strong> Reaching the far edge crowns your piece a King (K). Kings can move diagonally forward <em>and</em> backward.</li>
                        <li><strong>Winning:</strong> Capture all opponent pieces or block them from making a legal move.</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};
