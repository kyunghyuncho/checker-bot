import { useEffect, useState } from 'react';
import { Box, Trash2, CheckCircle } from 'lucide-react';

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

export const ModelRegistry = () => {
    const [models, setModels] = useState<ModelMeta[]>([]);
    const [activeModelId, setActiveModelId] = useState<string | null>(null);

    const fetchModels = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/models');
            const data = await res.json();
            setModels(data.models);
            setActiveModelId(data.active_model_id);
        } catch (e) {
            console.error("Failed to fetch models:", e);
        }
    };

    useEffect(() => {
        fetchModels();

        // Listen for model updates via WebSocket
        const ws = new WebSocket('ws://localhost:8000/ws/metrics');
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'models_updated') {
                fetchModels();
            }
        };
        return () => ws.close();
    }, []);

    const handleSelect = async (modelId: string) => {
        try {
            const res = await fetch(`http://localhost:8000/api/models/${modelId}/select`, { method: 'POST' });
            const data = await res.json();
            setActiveModelId(data.active_model_id);
        } catch (e) {
            console.error("Failed to select model:", e);
        }
    };

    const handleDelete = async (modelId: string) => {
        try {
            const res = await fetch(`http://localhost:8000/api/models/${modelId}`, { method: 'DELETE' });
            const data = await res.json();
            setActiveModelId(data.active_model_id);
            setModels(prev => prev.filter(m => m.id !== modelId));
        } catch (e) {
            console.error("Failed to delete model:", e);
        }
    };

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    };

    return (
        <div className="panel">
            <h2><Box size={20} /> Model Registry</h2>

            {models.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textAlign: 'center', padding: '1rem 0' }}>
                    No models trained yet. Use the Configuration panel to train one.
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {models.map(m => {
                        const isActive = m.id === activeModelId;
                        return (
                            <div
                                key={m.id}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.5rem',
                                    padding: '0.6rem 0.75rem',
                                    borderRadius: '0.5rem',
                                    backgroundColor: isActive ? 'rgba(59, 130, 246, 0.15)' : 'var(--bg-primary)',
                                    border: `1px solid ${isActive ? 'var(--accent-blue)' : 'var(--glass-border)'}`,
                                    transition: 'all 0.2s ease',
                                    cursor: 'pointer',
                                }}
                                onClick={() => handleSelect(m.id)}
                            >
                                {/* Active indicator */}
                                <CheckCircle
                                    size={16}
                                    style={{
                                        color: isActive ? 'var(--accent-green)' : 'var(--bg-tertiary)',
                                        flexShrink: 0
                                    }}
                                />

                                {/* Model info */}
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                                        {m.num_conv_layers}L / {m.hidden_dims}d / dr{m.dropout_rate}
                                    </div>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                        <span>{m.epochs_trained}ep</span>
                                        {m.final_train_loss !== undefined && <span>T:{m.final_train_loss.toFixed(3)}</span>}
                                        {m.final_val_loss !== undefined && <span>V:{m.final_val_loss.toFixed(3)}</span>}
                                        <span>{formatDate(m.created_at)}</span>
                                    </div>
                                </div>

                                {/* Delete button */}
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleDelete(m.id); }}
                                    style={{
                                        background: 'none',
                                        border: 'none',
                                        padding: '0.25rem',
                                        color: 'var(--text-muted)',
                                        cursor: 'pointer',
                                        flexShrink: 0,
                                        borderRadius: '0.25rem',
                                        display: 'flex'
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
        </div>
    );
};
