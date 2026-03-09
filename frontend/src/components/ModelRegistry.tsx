/**
 * ModelRegistry.tsx — AI Model Arena Panel
 * ==========================================
 * Lists all trained models and lets the user assign them to play as Red or White.
 *
 * Each model row shows:
 *   - 🔴 radio button — assign this model to play as Red (player 1)
 *   - Model metadata (ID, epochs, architecture, loss)
 *   - ⚪ radio button — assign this model to play as White (player 2)
 *   - 🗑 delete button
 *
 * Selection logic:
 *   - Click a radio to assign; click again to deselect (toggle behavior)
 *   - The same model CAN be assigned to both sides (plays itself)
 *   - A status bar at the bottom shows the current Red/White assignment (AI or Human)
 *
 * Refreshes automatically via WebSocket when new models are trained (models_updated event).
 */
import { useEffect, useState } from 'react';
import { Box, Trash2 } from 'lucide-react';

interface ModelMeta {
    id: string;
    created_at: string;
    epochs_trained: number;
    hidden_dims: number;
    num_conv_layers: number;
    dropout_rate: number;
    learning_rate: number;
    batch_size: number;
    final_train_loss?: number;
    final_val_loss?: number;
}

interface ModelRegistryProps {
    blackModelId: string | null;
    whiteModelId: string | null;
    onAssign: (side: 'black' | 'white', modelId: string | null) => void;
}

export const ModelRegistry: React.FC<ModelRegistryProps> = ({ blackModelId, whiteModelId, onAssign }) => {
    const [models, setModels] = useState<ModelMeta[]>([]);

    const fetchModels = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/models');
            const data = await res.json();
            setModels(data.models);
        } catch (e) {
            console.error("Failed to fetch models:", e);
        }
    };

    useEffect(() => {
        fetchModels();
        const ws = new WebSocket('ws://localhost:8000/ws/metrics');
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'models_updated') {
                fetchModels();
            }
        };
        return () => ws.close();
    }, []);

    const handleDelete = async (modelId: string) => {
        try {
            await fetch(`http://localhost:8000/api/models/${modelId}`, { method: 'DELETE' });
            setModels(prev => prev.filter(m => m.id !== modelId));
            // Clear assignment if this model was assigned
            if (blackModelId === modelId) onAssign('black', null);
            if (whiteModelId === modelId) onAssign('white', null);
        } catch (e) {
            console.error("Failed to delete model:", e);
        }
    };

    const handleToggle = (side: 'black' | 'white', modelId: string) => {
        const currentId = side === 'black' ? blackModelId : whiteModelId;
        if (currentId === modelId) {
            // Deselect
            onAssign(side, null);
        } else {
            onAssign(side, modelId);
        }
    };

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    };

    const radioStyle = (active: boolean, color: string): React.CSSProperties => ({
        width: '20px',
        height: '20px',
        borderRadius: '50%',
        border: `2px solid ${active ? color : 'var(--glass-border)'}`,
        backgroundColor: active ? color : 'transparent',
        cursor: 'pointer',
        flexShrink: 0,
        transition: 'all 0.15s ease',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    });

    return (
        <div className="panel">
            <h2><Box size={20} /> Model Arena</h2>

            {/* Column headers */}
            {models.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', padding: '0 0.75rem' }}>
                    <div style={{ width: '20px', textAlign: 'center', fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 600 }}>🔴</div>
                    <div style={{ flex: 1, fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Model</div>
                    <div style={{ width: '20px', textAlign: 'center', fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 600 }}>⚪</div>
                    <div style={{ width: '22px' }} />
                </div>
            )}

            {models.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textAlign: 'center', padding: '1rem 0' }}>
                    No models trained yet. Train one above.
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    {models.map(m => {
                        const isBlack = m.id === blackModelId;
                        const isWhite = m.id === whiteModelId;
                        const isAssigned = isBlack || isWhite;
                        return (
                            <div
                                key={m.id}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.5rem',
                                    padding: '0.5rem 0.75rem',
                                    borderRadius: '0.5rem',
                                    backgroundColor: isAssigned ? 'rgba(59, 130, 246, 0.1)' : 'var(--bg-primary)',
                                    border: `1px solid ${isAssigned ? 'var(--accent-blue)' : 'var(--glass-border)'}`,
                                    transition: 'all 0.2s ease',
                                }}
                            >
                                {/* Black radio */}
                                <div
                                    style={radioStyle(isBlack, 'var(--piece-red)')}
                                    onClick={() => handleToggle('black', m.id)}
                                    title="Assign to Red"
                                />

                                {/* Model info */}
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                                        {m.num_conv_layers}L / {m.hidden_dims}d / dr{m.dropout_rate}
                                    </div>
                                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                                        <span>{m.epochs_trained}ep</span>
                                        {m.final_train_loss !== undefined && <span>T:{m.final_train_loss.toFixed(3)}</span>}
                                        {m.final_val_loss !== undefined && <span>V:{m.final_val_loss.toFixed(3)}</span>}
                                        <span>{formatDate(m.created_at)}</span>
                                    </div>
                                </div>

                                {/* White radio */}
                                <div
                                    style={radioStyle(isWhite, 'var(--piece-white)')}
                                    onClick={() => handleToggle('white', m.id)}
                                    title="Assign to White"
                                />

                                {/* Delete */}
                                <button
                                    onClick={() => handleDelete(m.id)}
                                    style={{
                                        background: 'none', border: 'none', padding: '0.25rem',
                                        color: 'var(--text-muted)', cursor: 'pointer', flexShrink: 0,
                                        borderRadius: '0.25rem', display: 'flex'
                                    }}
                                    title="Delete model"
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Status bar */}
            <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                <span>🔴 {blackModelId ? 'AI' : 'Human'}</span>
                <span>vs</span>
                <span>⚪ {whiteModelId ? 'AI' : 'Human'}</span>
            </div>
        </div>
    );
};
