/**
 * GameMonitor.tsx — Game Statistics Panel
 * ========================================
 * Displays instantaneous and cumulative game statistics below the board.
 *
 * Instantaneous: piece counts, material advantage, legal moves for current player.
 * Cumulative: move count, material balance history chart.
 */
import React, { useMemo } from 'react';
import { BarChart3 } from 'lucide-react';

// Piece constants matching backend
const BLACK = 1, WHITE = 2, BLACK_KING = 3, WHITE_KING = 4;

interface MaterialSnapshot {
    redScore: number;
    whiteScore: number;
}

interface GameMonitorProps {
    grid: number[][];
    currentTurn: number;
    moveHistory: MaterialSnapshot[];
}

/** Count pieces and compute material scores from the grid */
function computeStats(grid: number[][]) {
    let redPieces = 0, whitePieces = 0, redKings = 0, whiteKings = 0;
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const p = grid[r][c];
            if (p === BLACK) redPieces++;
            else if (p === WHITE) whitePieces++;
            else if (p === BLACK_KING) redKings++;
            else if (p === WHITE_KING) whiteKings++;
        }
    }
    const redScore = redPieces * 1.0 + redKings * 1.5;
    const whiteScore = whitePieces * 1.0 + whiteKings * 1.5;
    return { redPieces, whitePieces, redKings, whiteKings, redScore, whiteScore };
}

/** Count legal moves for a given player (simplified client-side) */
function countLegalMoves(grid: number[][], turn: number): number {
    const isBlack = (p: number) => p === BLACK || p === BLACK_KING;
    const isWhite = (p: number) => p === WHITE || p === WHITE_KING;
    const isKing = (p: number) => p === BLACK_KING || p === WHITE_KING;
    const ownPiece = turn === 1 ? isBlack : isWhite;
    const enemyPiece = turn === 1 ? isWhite : isBlack;
    const inBounds = (r: number, c: number) => r >= 0 && r < 8 && c >= 0 && c < 8;

    let jumps = 0, simples = 0;

    // Determine directions: regular pieces move one way, kings move both
    const fwdDirs = turn === 1 ? [1] : [-1]; // Red goes down, White goes up
    const allDirs = [-1, 1];

    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            if (!ownPiece(grid[r][c])) continue;
            const dirs = isKing(grid[r][c]) ? allDirs : fwdDirs;
            for (const dr of dirs) {
                for (const dc of [-1, 1]) {
                    // Jump?
                    const mr = r + dr, mc = c + dc;
                    const jr = r + 2 * dr, jc = c + 2 * dc;
                    if (inBounds(jr, jc) && enemyPiece(grid[mr][mc]) && grid[jr][jc] === 0) {
                        jumps++;
                    }
                    // Simple move
                    if (inBounds(mr, mc) && grid[mr][mc] === 0) {
                        simples++;
                    }
                }
            }
        }
    }
    // If jumps exist, only jumps are legal (mandatory capture)
    return jumps > 0 ? jumps : simples;
}

/** Tiny inline SVG sparkline chart */
const MaterialChart: React.FC<{ history: MaterialSnapshot[] }> = ({ history }) => {
    if (history.length < 2) return <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center', padding: '0.5rem' }}>Play moves to see material chart</div>;

    const W = 320, H = 60, PAD = 2;
    const balances = history.map(s => s.redScore - s.whiteScore);
    const maxAbs = Math.max(1, ...balances.map(Math.abs));

    const points = balances.map((b, i) => {
        const x = PAD + (i / (balances.length - 1)) * (W - 2 * PAD);
        const y = H / 2 - (b / maxAbs) * (H / 2 - PAD);
        return `${x},${y}`;
    }).join(' ');

    return (
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
            {/* Zero line */}
            <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="var(--border-color)" strokeWidth="1" strokeDasharray="4,3" />
            {/* Material curve */}
            <polyline fill="none" stroke="var(--accent-blue)" strokeWidth="2" points={points} />
            {/* Labels */}
            <text x={W - PAD} y={PAD + 8} textAnchor="end" fontSize="8" fill="var(--piece-red)" opacity={0.6}>Red ↑</text>
            <text x={W - PAD} y={H - PAD} textAnchor="end" fontSize="8" fill="var(--text-muted)" opacity={0.6}>White ↑</text>
        </svg>
    );
};

export const GameMonitor: React.FC<GameMonitorProps> = ({ grid, currentTurn, moveHistory }) => {
    const stats = useMemo(() => computeStats(grid), [grid]);
    const legalMoves = useMemo(() => countLegalMoves(grid, currentTurn), [grid, currentTurn]);
    const advantage = stats.redScore - stats.whiteScore;

    return (
        <div className="panel" style={{ marginTop: '1rem' }}>
            <h2 style={{ fontSize: '1rem', marginBottom: '0.75rem' }}><BarChart3 size={18} /> Game Monitor</h2>

            {/* Instantaneous stats grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
                {/* Pieces */}
                <div style={{ textAlign: 'center', padding: '0.5rem', borderRadius: '0.5rem', backgroundColor: 'var(--bg-primary)' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Pieces</div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', fontSize: '0.9rem', fontWeight: 600 }}>
                        <span style={{ color: 'var(--piece-red)' }}>{stats.redPieces}+{stats.redKings}K</span>
                        <span style={{ color: 'var(--text-muted)' }}>vs</span>
                        <span style={{ color: 'var(--text-secondary)' }}>{stats.whitePieces}+{stats.whiteKings}K</span>
                    </div>
                </div>

                {/* Material advantage */}
                <div style={{ textAlign: 'center', padding: '0.5rem', borderRadius: '0.5rem', backgroundColor: 'var(--bg-primary)' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Material</div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 700, color: advantage > 0 ? 'var(--piece-red)' : advantage < 0 ? 'var(--accent-blue)' : 'var(--text-muted)' }}>
                        {advantage > 0 ? `+${advantage.toFixed(1)} Red` : advantage < 0 ? `+${Math.abs(advantage).toFixed(1)} White` : 'Equal'}
                    </div>
                </div>

                {/* Legal moves & move count */}
                <div style={{ textAlign: 'center', padding: '0.5rem', borderRadius: '0.5rem', backgroundColor: 'var(--bg-primary)' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Moves</div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                        <span>{legalMoves} legal</span>
                        <span style={{ color: 'var(--text-muted)', margin: '0 0.25rem' }}>·</span>
                        <span style={{ color: 'var(--text-muted)' }}>#{moveHistory.length}</span>
                    </div>
                </div>
            </div>

            {/* Material balance chart */}
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Material Balance Over Time</div>
            <MaterialChart history={moveHistory} />
        </div>
    );
};

export type { MaterialSnapshot };
