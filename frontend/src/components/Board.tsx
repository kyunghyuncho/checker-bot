
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

// Represents standard Python representation
const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;
const BLACK_KING = 3;
const WHITE_KING = 4;

interface BoardProps {
    grid: number[][];
    currentTurn: number;
    humanColor: number;
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
        bgColor = 'var(--piece-black)';
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
                cursor: isDraggable ? (isDragging ? 'grabbing' : 'grab') : 'not-allowed',
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
const DroppableSquare = ({ r, c, piece, currentTurn, humanColor }: { r: number, c: number, piece: number, currentTurn: number, humanColor: number }) => {
    const isPlayable = (r + c) % 2 === 1;
    const id = `square-${r}-${c}`;

    // Human plays whichever color is selected.
    const isHumanPiece = piece === humanColor || piece === humanColor + 2;
    const isDraggable = isPlayable && isHumanPiece && currentTurn === humanColor;

    const { isOver, setNodeRef } = useDroppable({
        id: id,
        data: { r, c },
        disabled: !isPlayable // Only dark squares are valid drop zones
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
    grid, currentTurn, humanColor, setGrid, setCurrentTurn, setCnnProbabilities, gameOver, setGameOver
}) => {
    const sensors = useSensors(
        useSensor(PointerSensor, { activationConstraint: { distance: 5 } })
    );

    const handleDragEnd = async (event: DragEndEvent) => {
        const { active, over } = event;
        if (!over) return; // Dropped outside valid square

        const startData = active.data.current;
        const endData = over.data.current;

        if (!startData || !endData) return;

        // Prevent dropping on the same square
        if (startData.r === endData.r && startData.c === endData.c) return;

        // Prevent human from moving if it's AI's turn or game over
        if (currentTurn !== humanColor || gameOver) return;

        // Ask Python backend if this is a legal Checkers move
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

            if (!valData.is_valid) {
                // Move was rejected (e.g., missed a mandatory jump, or moved backwards)
                return;
            }

            // 1. UI Update (Client-side fast feedback using validated data)
            const newGrid = grid.map(row => [...row]);

            // Move Piece
            newGrid[startData.r][startData.c] = EMPTY;
            newGrid[endData.r][endData.c] = startData.piece;

            // Remove jumped pieces identified by the engine
            if (valData.jumped_pieces) {
                for (const [r, c] of valData.jumped_pieces) {
                    newGrid[r][c] = EMPTY;
                }
            }

            // Kinging logic Check
            if (startData.piece === BLACK && endData.r === 7) newGrid[endData.r][endData.c] = BLACK_KING;
            if (startData.piece === WHITE && endData.r === 0) newGrid[endData.r][endData.c] = WHITE_KING;

            setGrid(newGrid);

            // Swap turn
            const nextTurn = currentTurn === 1 ? 2 : 1;
            setCurrentTurn(nextTurn);
        } catch (err) {
            console.error("Validation Error:", err);
        }
    };

    const isInferencingRef = useRef(false);

    useEffect(() => {
        // AI's turn
        if (currentTurn !== humanColor && !isInferencingRef.current) {
            isInferencingRef.current = true;

            const fetchAIMove = async () => {
                try {
                    const response = await fetch('http://localhost:8000/api/infer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            board_state: grid,
                            current_turn: currentTurn,
                            depth: 4
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

                        // Short delay makes the AI's move visible to a human
                        setTimeout(() => {
                            setGrid(aiGrid);
                            setCurrentTurn(humanColor);
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
        }
    }, [currentTurn, humanColor, grid, setGrid, setCurrentTurn, setCnnProbabilities, gameOver, setGameOver]);

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
            {gameOver && (
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
                    <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem', color: gameOver === humanColor ? '#4ade80' : '#f87171' }}>
                        {gameOver === humanColor ? 'You Win!' : 'AI Wins!'}
                    </h2>
                    <p style={{ margin: 0 }}>{gameOver === 1 ? 'Black' : 'White'} takes the game.</p>
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
                            <DroppableSquare key={`sq-${r}-${c}`} r={r} c={c} piece={piece} currentTurn={currentTurn} humanColor={humanColor} />
                        ))
                    )}
                </div>
            </DndContext>
        </div>
    );
};
