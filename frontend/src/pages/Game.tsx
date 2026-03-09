import { useState } from 'react';
import { Info, RotateCcw } from 'lucide-react';
import { Board } from '../components/Board';
import { ConfigPanel } from '../components/ConfigPanel';
import { BrainVisualizer } from '../components/BrainVisualizer';
import { Metrics } from '../components/Metrics';

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
    const [humanColor, setHumanColor] = useState<number>(1); // Player defaults to Black (1)
    const [cnnProbabilities, setCnnProbabilities] = useState<{ p_black: number, p_white: number } | null>(null);
    const [gameOver, setGameOver] = useState<number | null>(null);

    const handleReset = () => {
        setBoardState(getInitialGrid());
        setCurrentTurn(2); // Reset to White's turn
        setCnnProbabilities(null);
        setGameOver(null);
    };

    return (
        <div className="dashboard-grid">
            {/* Left Column: Controls & Configuration */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <ConfigPanel />
                <Metrics />
            </div>

            {/* Center Column: The Checkers Board */}
            <div className="panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', marginBottom: '1rem' }}>
                    <h2 style={{ margin: 0, borderBottom: 'none', paddingBottom: 0 }}>Interactive Board</h2>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <select
                            value={humanColor}
                            onChange={(e) => setHumanColor(parseInt(e.target.value))}
                            style={{ padding: '0.5rem', borderRadius: '0.25rem', backgroundColor: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--glass-border)' }}
                        >
                            <option value={1}>Play as Black</option>
                            <option value={2}>Play as White</option>
                        </select>
                        <button onClick={handleReset} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: 'var(--bg-secondary)', padding: '0.5rem 1rem', fontSize: '0.875rem' }}>
                            <RotateCcw size={16} /> Reset Game
                        </button>
                    </div>
                </div>
                <div className="edu-note" style={{ width: '100%', marginTop: 0, marginBottom: '1.5rem' }}>
                    <strong>Educational Note:</strong> Drag pieces to play. Checkers rules enforce mandatory jumps.
                    The AI will respond immediately after your turn.
                </div>

                <Board
                    grid={boardState}
                    currentTurn={currentTurn}
                    humanColor={humanColor}
                    setGrid={setBoardState}
                    setCurrentTurn={setCurrentTurn}
                    setCnnProbabilities={setCnnProbabilities}
                    gameOver={gameOver}
                    setGameOver={setGameOver}
                />

                <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <div className={`status-indicator ${currentTurn === 1 ? 'active' : ''}`} style={{ backgroundColor: 'var(--piece-black)' }} /> Black {currentTurn === 1 ? "(Active)" : ""}
                    <div className={`status-indicator ${currentTurn === 2 ? 'active' : ''}`} style={{ backgroundColor: 'var(--piece-white)' }} /> White {currentTurn === 2 ? "(Active)" : ""}
                </div>
            </div>

            {/* Right Column: AI Brain Visualization & Rules */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <BrainVisualizer probabilities={cnnProbabilities} />

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
                        <li><strong>Your Color:</strong> You are playing as {humanColor === 1 ? 'Black' : 'White'}.</li>
                        <li><strong>Turn Order:</strong> <em>White</em> always moves first.</li>
                        <li><strong>Direction:</strong> Black starts at the top and moves <em>down</em>. White starts at the bottom and moves <em>up</em>.</li>
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
