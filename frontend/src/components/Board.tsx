/**
 * Board.tsx — Interactive Checkers Board with Drag-and-Drop
 * ==========================================================
 * Renders the 8×8 checkers board and handles all move logic.
 *
 * Component hierarchy:
 *   Board (main) → DroppableSquare (64 cells) → DraggablePiece (where pieces exist)
 *
 * Play modes (determined by which model IDs are non-null):
 *   - Human vs Human:  Both sides draggable, no AI moves
 *   - Human vs AI:     One side draggable, AI auto-moves on its turn
 *   - AI vs AI:        No pieces draggable, both sides auto-move with 500ms delay
 *
 * AI inference pipeline:
 *   1. useEffect fires when currentTurn changes and belongs to an AI side
 *   2. POST /api/infer with board state, model_id, and epsilon
 *   3. Response contains the best move and CNN win probabilities
 *   4. Move is applied to the grid with a 500ms delay for visibility
 *   5. Turn switches, triggering the next AI's move (if AI vs AI)
 *
 * Human move pipeline:
 *   1. Player drags a piece (DraggablePiece) to a target (DroppableSquare)
 *   2. POST /api/validate_move checks legality (including mandatory jumps)
 *   3. If valid, move is applied locally and turn switches
 *
 * Piece constants match the Python backend: EMPTY=0, BLACK=1, WHITE=2, etc.
 */

import {
    DndContext,
    useSensor,
    useSensors,
    PointerSensor,
    type DragEndEvent,
    useDraggable,
    useDroppable
} from '@dnd-kit/core';
import { useEffect, useRef } from 'react';

// Piece constants — must match backend/engine/board.py
const EMPTY = 0;
const BLACK = 1;      // Regular red piece (called "black" internally)
const WHITE = 2;      // Regular white piece
const BLACK_KING = 3; // Promoted red piece (king)
const WHITE_KING = 4; // Promoted white piece (king)

interface BoardProps {
    grid: number[][];
    currentTurn: number;
    blackModelId: string | null;
    whiteModelId: string | null;
    epsilon: number;
    searchDepth: number;
    isPlaying: boolean;
    setGrid: (grid: number[][]) => void;
    setCurrentTurn: (turn: number) => void;
    setCnnProbabilities: (probs: any) => void;
    gameOver: number | null;
    setGameOver: (winner: number | null) => void;
}

// ----------------------------------------------------
// Draggable Piece Component
// ----------------------------------------------------
const DraggablePiece = ({ id, piece, r, c, isDraggable }: { id: string, piece: number, r: number, c: number, isDraggable: boolean }) => {
    const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
        id: id,
        data: { r, c, piece }, // Pass location via drag data
        disabled: !isDraggable
    });

    const style = transform ? {
        transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`,
        zIndex: 999,
    } : undefined;

    let bgColor = 'transparent';
    let border = 'none';
    if (piece === BLACK || piece === BLACK_KING) {
        bgColor = 'var(--piece-red)';
        border = `2px solid ${piece === BLACK_KING ? 'gold' : '#922b21'}`;
    } else if (piece === WHITE || piece === WHITE_KING) {
        bgColor = 'var(--piece-white)';
        border = `2px solid ${piece === WHITE_KING ? 'gold' : '#cbd5e1'}`;
    }

    return (
        <div
            ref={setNodeRef}
            style={{
                ...style,
                width: '80%',
                height: '80%',
                borderRadius: '50%',
                backgroundColor: bgColor,
                border,
                boxShadow: isDragging ? '0 10px 25px rgba(0,0,0,0.5)' : '0 4px 6px rgba(0,0,0,0.3)',
                cursor: isDraggable ? (isDragging ? 'grabbing' : 'grab') : 'default',
                opacity: isDraggable ? 1 : 0.8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
            }}
            {...listeners}
            {...attributes}
        >
            {/* Crown indicator for Kings */}
            {(piece === BLACK_KING || piece === WHITE_KING) && (
                <span style={{ color: 'gold', fontSize: '1.2rem', fontWeight: 'bold' }}>K</span>
            )}
        </div>
    );
};

// ----------------------------------------------------
// Droppable Square Component
// ----------------------------------------------------
const DroppableSquare = ({ r, c, piece, currentTurn, isHumanTurn }: { r: number, c: number, piece: number, currentTurn: number, isHumanTurn: boolean }) => {
    const isPlayable = (r + c) % 2 === 1;
    const id = `square-${r}-${c}`;

    // A piece is draggable if it belongs to the current turn AND it's the human's turn
    const belongsToCurrentTurn = (piece === currentTurn) || (piece === currentTurn + 2);
    const isDraggable = isPlayable && belongsToCurrentTurn && isHumanTurn;

    const { isOver, setNodeRef } = useDroppable({
        id: id,
        data: { r, c },
        disabled: !isPlayable
    });

    const bgColor = isPlayable ? 'var(--board-dark)' : 'var(--board-light)';
    const highlight = isOver && isPlayable ? 'rgba(59, 130, 246, 0.5)' : 'transparent';

    return (
        <div
            ref={setNodeRef}
            style={{
                width: '100%',
                height: '100%',
                backgroundColor: bgColor,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                position: 'relative',
                boxShadow: `inset 0 0 0 4px ${highlight}`,
                transition: 'box-shadow 0.2s ease',
            }}
        >
            {piece !== EMPTY && (
                <DraggablePiece id={`piece-${r}-${c}`} piece={piece} r={r} c={c} isDraggable={isDraggable} />
            )}
        </div>
    );
};


// ----------------------------------------------------
// Main Board Component
// ----------------------------------------------------
export const Board: React.FC<BoardProps> = ({
    grid, currentTurn, blackModelId, whiteModelId, epsilon, searchDepth, isPlaying, setGrid, setCurrentTurn, setCnnProbabilities, gameOver, setGameOver
}) => {
    const sensors = useSensors(
        useSensor(PointerSensor, { activationConstraint: { distance: 5 } })
    );

    // Determine if the current turn belongs to a human
    const isCurrentTurnAI = (currentTurn === 1 && blackModelId !== null) || (currentTurn === 2 && whiteModelId !== null);
    const isHumanTurn = !isCurrentTurnAI && !gameOver;

    const handleDragEnd = async (event: DragEndEvent) => {
        const { active, over } = event;
        if (!over) return;

        const startData = active.data.current;
        const endData = over.data.current;

        if (!startData || !endData) return;
        if (startData.r === endData.r && startData.c === endData.c) return;

        // Prevent human from moving if it's AI's turn or game over
        if (!isHumanTurn) return;

        try {
            const valRes = await fetch('http://localhost:8000/api/validate_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    board_state: grid,
                    current_turn: currentTurn,
                    start_r: startData.r,
                    start_c: startData.c,
                    end_r: endData.r,
                    end_c: endData.c
                })
            });
            const valData = await valRes.json();

            if (!valData.is_valid) return;

            const newGrid = grid.map(row => [...row]);
            newGrid[startData.r][startData.c] = EMPTY;
            newGrid[endData.r][endData.c] = startData.piece;

            if (valData.jumped_pieces) {
                for (const [r, c] of valData.jumped_pieces) {
                    newGrid[r][c] = EMPTY;
                }
            }

            if (startData.piece === BLACK && endData.r === 7) newGrid[endData.r][endData.c] = BLACK_KING;
            if (startData.piece === WHITE && endData.r === 0) newGrid[endData.r][endData.c] = WHITE_KING;

            setGrid(newGrid);
            const nextTurn = currentTurn === 1 ? 2 : 1;
            setCurrentTurn(nextTurn);
        } catch (err) {
            console.error("Validation Error:", err);
        }
    };

    const isInferencingRef = useRef(false);

    useEffect(() => {
        if (gameOver || !isPlaying) return;

        // Determine the model for the current turn
        const modelId = currentTurn === 1 ? blackModelId : whiteModelId;

        // Only proceed if the current side has an AI model assigned
        if (!modelId || isInferencingRef.current) return;

        isInferencingRef.current = true;

        const fetchAIMove = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        board_state: grid,
                        current_turn: currentTurn,
                        depth: searchDepth,
                        model_id: modelId,
                        epsilon: epsilon
                    })
                });

                const data = await response.json();
                setCnnProbabilities(data.cnn_probabilities);

                if (data.move) {
                    const aiGrid = grid.map(row => [...row]);
                    const [startR, startC] = data.move.start;
                    const [endR, endC] = data.move.end;
                    const aiPiece = aiGrid[startR][startC];

                    aiGrid[startR][startC] = EMPTY;
                    aiGrid[endR][endC] = aiPiece;

                    if (aiPiece === BLACK && endR === 7) aiGrid[endR][endC] = BLACK_KING;
                    if (aiPiece === WHITE && endR === 0) aiGrid[endR][endC] = WHITE_KING;

                    if (data.move.jumped_pieces && data.move.jumped_pieces.length > 0) {
                        for (const [r, c] of data.move.jumped_pieces) {
                            aiGrid[r][c] = EMPTY;
                        }
                    }

                    // Delay so moves are visible to the human observer
                    setTimeout(() => {
                        setGrid(aiGrid);
                        setCurrentTurn(currentTurn === 1 ? 2 : 1);
                        isInferencingRef.current = false;
                        if (data.game_over) {
                            setGameOver(data.game_over);
                        }
                    }, 500);
                } else {
                    isInferencingRef.current = false;
                    if (data.game_over) {
                        setGameOver(data.game_over);
                    }
                }

            } catch (err) {
                console.error("Move Error:", err);
                isInferencingRef.current = false;
            }
        };

        fetchAIMove();
    }, [currentTurn, blackModelId, whiteModelId, grid, setGrid, setCurrentTurn, setCnnProbabilities, gameOver, setGameOver, isPlaying]);

    // Game over display logic
    const getGameOverText = () => {
        if (!gameOver) return null;
        const winnerName = gameOver === 1 ? 'Red' : 'White';
        const blackIsAI = blackModelId !== null;
        const whiteIsAI = whiteModelId !== null;

        if (blackIsAI && whiteIsAI) {
            return { title: `${winnerName} Wins!`, subtitle: 'AI vs AI match concluded.' };
        } else if (!blackIsAI && !whiteIsAI) {
            return { title: `${winnerName} Wins!`, subtitle: 'Player vs Player match concluded.' };
        } else {
            const humanSide = blackIsAI ? 2 : 1;
            const humanWon = gameOver === humanSide;
            return {
                title: humanWon ? 'You Win!' : 'AI Wins!',
                subtitle: `${winnerName} takes the game.`
            };
        }
    };

    const gameOverInfo = getGameOverText();

    return (
        <div style={{
            width: '400px',
            height: '400px',
            border: '4px solid var(--glass-border)',
            borderRadius: '0.5rem',
            overflow: 'hidden',
            boxShadow: '0 20px 25px -5px rgba(0,0,0,0.2)',
            position: 'relative'
        }}>
            {gameOver && gameOverInfo && (
                <div style={{
                    position: 'absolute',
                    top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    zIndex: 1000,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    color: 'white',
                    backdropFilter: 'blur(4px)'
                }}>
                    <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem', color: '#4ade80' }}>
                        {gameOverInfo.title}
                    </h2>
                    <p style={{ margin: 0 }}>{gameOverInfo.subtitle}</p>
                </div>
            )}
            <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(8, 1fr)',
                    gridTemplateRows: 'repeat(8, 1fr)',
                    width: '100%',
                    height: '100%'
                }}>
                    {grid.map((row, r) =>
                        row.map((piece, c) => (
                            <DroppableSquare key={`sq-${r}-${c}`} r={r} c={c} piece={piece} currentTurn={currentTurn} isHumanTurn={isHumanTurn} />
                        ))
                    )}
                </div>
            </DndContext>
        </div>
    );
};
