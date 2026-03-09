import { useState, useEffect } from 'react';
import { Settings, Play, Database } from 'lucide-react';

export const ConfigPanel = () => {
    const [numGames, setNumGames] = useState(10);
    const [depth, setDepth] = useState(4);
    const [epsilon, setEpsilon] = useState(0.1);
    const [epochs, setEpochs] = useState(5);
    const [lr, setLr] = useState(0.001);
    const [statusText, setStatusText] = useState<string>('');
    const [errorMsg, setErrorMsg] = useState<string>('');

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/metrics');
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'status') {
                setStatusText(msg.message);
                if (msg.message === "Data Generation Complete!" || msg.message === "Training Complete!") {
                    // Optional: clear status after a few seconds
                    setTimeout(() => setStatusText(''), 3000);
                }
            } else if (msg.type === 'metric') {
                setStatusText(`Training: Epoch ${msg.epoch + 1}/${epochs}`);
            }
        };
        return () => ws.close();
    }, [epochs]);

    const handleGenerate = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_games: numGames, depth, epsilon })
            });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(JSON.stringify(errData.detail) || 'Failed to trigger generation');
            }
            setStatusText(`Started generation...`);
            setErrorMsg('');
        } catch (e: any) {
            console.error(e);
            setErrorMsg(`Error: ${e.message || e}`);
            setStatusText('');
        }
    };

    const handleTrain = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs, learning_rate: lr, hidden_dims: 64 })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to start training');
            }
            setStatusText(`Started training...`);
            setErrorMsg('');
        } catch (e: any) {
            console.error(e);
            setErrorMsg(`Error: ${e.message || e}`);
            setStatusText('');
        }
    };

    return (
        <div className="panel">
            <h2><Settings size={20} /> Configuration</h2>

            {(statusText || errorMsg) && (
                <div style={{
                    marginBottom: '1.5rem',
                    padding: '0.75rem',
                    borderRadius: '0.5rem',
                    backgroundColor: errorMsg ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                    border: `1px solid ${errorMsg ? 'var(--accent-red)' : 'var(--accent-green)'}`,
                    color: errorMsg ? 'var(--accent-red)' : 'var(--accent-green)',
                    fontSize: '0.9rem',
                    textAlign: 'center'
                }}>
                    <strong>{errorMsg ? "Error: " : "Status: "}</strong>
                    {errorMsg || statusText}
                </div>
            )}

            <div style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
                    1. Data Generation
                </h3>
                <div className="input-group">
                    <label>Games to Generate <span>{numGames}</span></label>
                    <input type="range" min="1" max="50" value={numGames} onChange={(e) => setNumGames(Number(e.target.value))} />
                </div>
                <div className="input-group">
                    <label>Search Depth <span>{depth}</span></label>
                    <input type="range" min="1" max="8" value={depth} onChange={(e) => setDepth(Number(e.target.value))} />
                </div>
                <div className="input-group">
                    <label>Epsilon (Randomness) <span>{epsilon}</span></label>
                    <input type="range" min="0" max="0.5" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} />
                </div>
                <button className="primary" onClick={handleGenerate} style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem' }}>
                    <Database size={18} /> Generate Data
                </button>
            </div>

            <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '1.5rem' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
                    2. Model Training
                </h3>
                <div className="input-group">
                    <label>Epochs <span>{epochs}</span></label>
                    <input type="range" min="1" max="20" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
                </div>
                <div className="input-group">
                    <label>Learning Rate <span>{lr}</span></label>
                    <input type="number" step="0.001" value={lr} onChange={(e) => setLr(Number(e.target.value))} />
                </div>
                <button className="primary" onClick={handleTrain} style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem', background: 'linear-gradient(135deg, var(--accent-green), #059669)' }}>
                    <Play size={18} /> Start Training
                </button>
            </div>
        </div>
    );
};
