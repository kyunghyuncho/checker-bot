/**
 * ConfigPanel.tsx — Data Generation & Model Training Controls
 * =============================================================
 * Left-column panel with two collapsible sections:
 *
 *   1. Data Generation — controls for self-play game generation:
 *      - num_games, search depth, epsilon (randomness)
 *      - Triggers POST /api/generate in the background
 *
 *   2. Model Training — architecture and optimization hyperparameters:
 *      - Architecture: conv layers, hidden dims, dropout
 *      - Optimization: epochs, learning rate, batch size
 *      - Early stopping: validation split, patience
 *      - Triggers POST /api/train in the background
 *
 * A status bar shows real-time progress received via WebSocket.
 */
import { useState, useEffect } from 'react';
import { Settings, Play, Database, ChevronDown, ChevronRight, Square } from 'lucide-react';

export const ConfigPanel = () => {
    const [numGames, setNumGames] = useState(10);
    const [depth, setDepth] = useState(4);
    const [epsilon, setEpsilon] = useState(0.1);
    const [discountFactor, setDiscountFactor] = useState(0.0);
    const [epochs, setEpochs] = useState(20);
    const [lr, setLr] = useState(0.001);
    const [hiddenDims, setHiddenDims] = useState(64);
    const [numConvLayers, setNumConvLayers] = useState(2);
    const [dropoutRate, setDropoutRate] = useState(0.2);
    const [batchSize, setBatchSize] = useState(32);
    const [valSplit, setValSplit] = useState(0.2);
    const [patience, setPatience] = useState(5);
    const [statusText, setStatusText] = useState<string>('');
    const [errorMsg, setErrorMsg] = useState<string>('');
    const [genOpen, setGenOpen] = useState(false);
    const [trainOpen, setTrainOpen] = useState(false);
    const [isTraining, setIsTraining] = useState(false);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/metrics');
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'training_started') {
                setIsTraining(true);
                setStatusText(msg.message);
                setErrorMsg('');
            } else if (msg.type === 'status') {
                setStatusText(msg.message);
                if (msg.message === "Data Generation Complete!" || msg.message === "Training Complete!") {
                    if (msg.message === "Training Complete!") setIsTraining(false);
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
                body: JSON.stringify({ num_games: numGames, depth, epsilon, discount_factor: discountFactor })
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
                body: JSON.stringify({
                    epochs,
                    learning_rate: lr,
                    hidden_dims: hiddenDims,
                    num_conv_layers: numConvLayers,
                    dropout_rate: dropoutRate,
                    batch_size: batchSize,
                    val_split: valSplit,
                    patience
                })
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

    const handleStopTraining = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/train/stop', {
                method: 'POST'
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to stop training');
            }
            setStatusText('Sent stop signal...');
            setErrorMsg('');
            // We do NOT set isTraining(false) yet. 
            // We wait for the "Training Complete!" status from the websocket 
            // to confirm PyTorch Lightning actually terminated the loop and saved the model.
        } catch (e: any) {
            console.error(e);
            setErrorMsg(`Error stopping: ${e.message || e}`);
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
                <h3
                    onClick={() => setGenOpen(!genOpen)}
                    style={{ fontSize: '1rem', marginBottom: genOpen ? '0.75rem' : 0, color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', userSelect: 'none' }}
                >
                    {genOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    1. Data Generation
                </h3>
                {genOpen && (<>
                    <div className="input-group">
                        <label>Games to Generate <span>{numGames}</span></label>
                        <input type="range" min="1" max="500" value={numGames} onChange={(e) => setNumGames(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Search Depth <span>{depth}</span></label>
                        <input type="range" min="1" max="8" value={depth} onChange={(e) => setDepth(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Epsilon (Randomness) <span>{epsilon}</span></label>
                        <input type="range" min="0" max="0.5" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Label Discount γ <span>{discountFactor.toFixed(2)}</span></label>
                        <input type="range" min="0" max="1" step="0.05" value={discountFactor} onChange={(e) => setDiscountFactor(Number(e.target.value))} />
                    </div>
                    <button className="primary" onClick={handleGenerate} style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem' }}>
                        <Database size={18} /> Generate Data
                    </button>
                </>)}
            </div>

            <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '1.5rem' }}>
                <h3
                    onClick={() => setTrainOpen(!trainOpen)}
                    style={{ fontSize: '1rem', marginBottom: trainOpen ? '0.75rem' : 0, color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', userSelect: 'none' }}
                >
                    {trainOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    2. Model Training
                </h3>
                {trainOpen && (<>

                    <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Architecture</h4>
                    <div className="input-group">
                        <label>Conv Layers <span>{numConvLayers}</span></label>
                        <input type="range" min="1" max="5" value={numConvLayers} onChange={(e) => setNumConvLayers(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Hidden Dims <span>{hiddenDims}</span></label>
                        <input type="range" min="16" max="256" step="16" value={hiddenDims} onChange={(e) => setHiddenDims(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Dropout Rate <span>{dropoutRate}</span></label>
                        <input type="range" min="0" max="0.5" step="0.05" value={dropoutRate} onChange={(e) => setDropoutRate(Number(e.target.value))} />
                    </div>

                    <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', marginTop: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Optimization</h4>
                    <div className="input-group">
                        <label>Max Epochs <span>{epochs}</span></label>
                        <input type="range" min="1" max="100" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Learning Rate <span>{lr}</span></label>
                        <input type="number" step="0.0001" value={lr} onChange={(e) => setLr(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Batch Size <span>{batchSize}</span></label>
                        <input type="range" min="8" max="128" step="8" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
                    </div>

                    <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', marginTop: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Early Stopping</h4>
                    <div className="input-group">
                        <label>Validation Split <span>{(valSplit * 100).toFixed(0)}%</span></label>
                        <input type="range" min="0.1" max="0.4" step="0.05" value={valSplit} onChange={(e) => setValSplit(Number(e.target.value))} />
                    </div>
                    <div className="input-group">
                        <label>Patience <span>{patience}</span></label>
                        <input type="range" min="0" max="20" value={patience} onChange={(e) => setPatience(Number(e.target.value))} />
                    </div>

                    {!isTraining ? (
                        <button className="primary" onClick={handleTrain} style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem', background: 'linear-gradient(135deg, var(--accent-green), #059669)' }}>
                            <Play size={18} /> Start Training
                        </button>
                    ) : (
                        <button className="primary" onClick={handleStopTraining} style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem', background: 'linear-gradient(135deg, var(--accent-red), #b91c1c)' }}>
                            <Square size={18} /> Stop Training
                        </button>
                    )}
                </>)}
            </div>
        </div>
    );
};
