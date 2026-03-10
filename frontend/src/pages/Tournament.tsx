/**
 * Tournament.tsx — AI Tournament Page
 * =====================================
 * Lets N trained models compete in M random games.
 * Displays live progress, final rankings, and head-to-head results.
 */
import { useState, useEffect, useRef } from 'react';
import { Trophy, Play } from 'lucide-react';

interface RankingEntry {
    model_id: string;
    name: string;
    wins: number;
    losses: number;
    draws: number;
    total: number;
    win_rate: number;
    rating?: number;
}

interface ProgressUpdate {
    game: number;
    total: number;
    red: string;
    white: string;
    result: string;
    moves: number;
    standings: RankingEntry[];
}

interface H2HEntry {
    red_id: string;
    white_id: string;
    red_wins: number;
    white_wins: number;
    draws: number;
}

interface ModelMeta {
    hidden_dims?: number;
    num_conv_layers?: number;
    dropout_rate?: number;
    learning_rate?: number;
    epochs_trained?: number;
    batch_size?: number;
    final_train_loss?: number;
    final_val_loss?: number;
}

const ModelTooltip = ({ name, meta }: { name: string; meta?: ModelMeta }) => {
    if (!meta) return <>{name}</>;
    return (
        <span style={{ position: 'relative', cursor: 'help', borderBottom: '1px dotted var(--text-muted)' }} className="model-tooltip">
            {name}
            <span className="model-tooltip-content">
                <strong>{name}</strong>
                <br />Layers: {meta.num_conv_layers ?? '?'} × {meta.hidden_dims ?? '?'}ch
                <br />Dropout: {meta.dropout_rate ?? '?'}
                <br />LR: {meta.learning_rate ?? '?'}
                <br />Epochs: {meta.epochs_trained ?? '?'}
                <br />Batch: {meta.batch_size ?? '?'}
                {meta.final_train_loss != null && <><br />Train loss: {meta.final_train_loss}</>}
                {meta.final_val_loss != null && <><br />Val loss: {meta.final_val_loss}</>}
            </span>
        </span>
    );
};

export const Tournament = () => {
    const [numGames, setNumGames] = useState(20);
    const [depth, setDepth] = useState(2);
    const [temperature, setTemperature] = useState(1.0);
    const [maxMoves, setMaxMoves] = useState(200);
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState<ProgressUpdate | null>(null);
    const [rankings, setRankings] = useState<RankingEntry[]>([]);
    const [h2h, setH2h] = useState<Record<string, H2HEntry>>({});
    const [modelMeta, setModelMeta] = useState<Record<string, ModelMeta>>({});
    const [liveStandings, setLiveStandings] = useState<RankingEntry[]>([]);
    const [log, setLog] = useState<string[]>([]);
    const [error, setError] = useState('');
    const logRef = useRef<HTMLDivElement>(null);

    // Restore results from sessionStorage on mount
    useEffect(() => {
        try {
            const saved = sessionStorage.getItem('tournament_results');
            if (saved) {
                const data = JSON.parse(saved);
                if (data.rankings) setRankings(data.rankings);
                if (data.h2h) setH2h(data.h2h);
                if (data.log) setLog(data.log);
                if (data.modelMeta) setModelMeta(data.modelMeta);
            }
        } catch { /* ignore parse errors */ }
    }, []);

    // Save results to sessionStorage whenever they change
    useEffect(() => {
        if (rankings.length > 0) {
            sessionStorage.setItem('tournament_results', JSON.stringify({ rankings, h2h, log, modelMeta }));
        }
    }, [rankings, h2h, log, modelMeta]);

    // WebSocket listener for tournament events
    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/metrics');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'tournament_progress') {
                setProgress(data);
                if (data.standings) setLiveStandings(data.standings);
                setLog(prev => [...prev, `Game ${data.game}/${data.total}: ${data.red} (Red) vs ${data.white} (White) → ${data.result} (${data.moves} moves)`]);
            } else if (data.type === 'tournament_complete') {
                setRankings(data.rankings);
                setH2h(data.head_to_head);
                if (data.model_meta) setModelMeta(data.model_meta);
                setRunning(false);
                setLog(prev => [...prev, `🏆 Tournament complete!`]);
            }
        };
        return () => ws.close();
    }, []);

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [log]);

    const handleStart = async () => {
        setRunning(true);
        setRankings([]);
        setH2h({});
        setModelMeta({});
        setLiveStandings([]);
        setLog([]);
        setProgress(null);
        setError('');
        try {
            const res = await fetch('http://localhost:8000/api/tournament', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_games: numGames, depth, temperature, max_moves: maxMoves })
            });
            if (!res.ok) {
                const errData = await res.json();
                setError(errData.detail || 'Failed to start tournament');
                setRunning(false);
            }
        } catch (e: any) {
            setError(e.message || 'Connection error');
            setRunning(false);
        }
    };

    const handleStop = async () => {
        try {
            await fetch('http://localhost:8000/api/tournament/stop', { method: 'POST' });
        } catch (e) {
            console.error('Failed to stop tournament', e);
        }
    };

    const handleReset = () => {
        setRankings([]);
        setH2h({});
        setModelMeta({});
        setLiveStandings([]);
        setLog([]);
        setProgress(null);
        setError('');
        sessionStorage.removeItem('tournament_results');
    };

    // Build unique model list for H2H matrix
    const modelIds = [...new Set(rankings.map(r => r.model_id))];
    const modelNames: Record<string, string> = {};
    rankings.forEach(r => { modelNames[r.model_id] = r.name; });

    // Lookup H2H result for a given pair
    const getH2H = (redId: string, whiteId: string) => {
        const key = `${redId}_vs_${whiteId}`;
        return h2h[key] || null;
    };

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '1.5rem', alignItems: 'start' }}>
            {/* Left: Controls */}
            <div className="panel">
                <h2 style={{ fontSize: '1rem', marginBottom: '1rem' }}><Trophy size={20} /> AI Tournament</h2>

                <div className="input-group">
                    <label>Total Games <span>{numGames}</span></label>
                    <input type="range" min="5" max="200" step="5" value={numGames} onChange={(e) => setNumGames(Number(e.target.value))} />
                </div>
                <div className="input-group">
                    <label>Search Depth <span>{depth}</span></label>
                    <input type="range" min="1" max="6" value={depth} onChange={(e) => setDepth(Number(e.target.value))} />
                </div>
                <div className="input-group">
                    <label>Temperature τ <span>{temperature.toFixed(1)}</span></label>
                    <input type="range" min="0" max="5" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
                </div>
                <div className="input-group">
                    <label>Max Moves <span>{maxMoves}</span></label>
                    <input type="range" min="50" max="500" step="50" value={maxMoves} onChange={(e) => setMaxMoves(Number(e.target.value))} />
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
                    {!running ? (
                        <button
                            className="primary"
                            onClick={handleStart}
                            style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '0.5rem' }}
                        >
                            <Play size={18} /> Run Tournament
                        </button>
                    ) : (
                        <button
                            className="primary"
                            onClick={handleStop}
                            style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '0.5rem', backgroundColor: 'var(--accent-red)' }}
                        >
                            ■ Stop
                        </button>
                    )}
                    {(rankings.length > 0 || liveStandings.length > 0) && !running && (
                        <button
                            onClick={handleReset}
                            style={{ padding: '0.5rem 1rem', borderRadius: '0.5rem', border: '1px solid var(--glass-border)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.85rem' }}
                        >
                            Reset
                        </button>
                    )}
                </div>

                {error && (
                    <div style={{ marginTop: '0.75rem', padding: '0.5rem', borderRadius: '0.5rem', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid var(--accent-red)', color: 'var(--accent-red)', fontSize: '0.85rem', textAlign: 'center' }}>
                        {error}
                    </div>
                )}

                {/* Progress */}
                {progress && running && (
                    <div style={{ marginTop: '1rem' }}>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
                            Progress
                        </div>
                        <div style={{ height: '6px', borderRadius: '3px', backgroundColor: 'var(--bg-primary)', overflow: 'hidden' }}>
                            <div style={{ width: `${(progress.game / progress.total) * 100}%`, height: '100%', borderRadius: '3px', backgroundColor: 'var(--accent-blue)', transition: 'width 0.3s' }} />
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem', textAlign: 'center' }}>
                            {progress.game} / {progress.total}
                        </div>
                    </div>
                )}

                {/* Game log */}
                <div ref={logRef} style={{ marginTop: '1rem', maxHeight: '250px', overflowY: 'auto', fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'monospace', lineHeight: 1.6, backgroundColor: 'var(--bg-primary)', borderRadius: '0.5rem', padding: '0.5rem' }}>
                    {log.length === 0 ? 'Run a tournament to see results...' : log.map((l, i) => <div key={i}>{l}</div>)}
                </div>
            </div>

            {/* Right: Results */}
            <div>
                {/* Live standings during tournament */}
                {running && liveStandings.length > 0 && (
                    <div className="panel" style={{ marginBottom: '1.5rem' }}>
                        <h2 style={{ fontSize: '1rem', marginBottom: '0.75rem' }}>Live Standings</h2>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.05em' }}>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>#</th>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>Model</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>W</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>L</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>D</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>Games</th>
                                    <th style={{ textAlign: 'right', padding: '0.5rem' }}>Win Rate</th>
                                </tr>
                            </thead>
                            <tbody>
                                {liveStandings.map((r, idx) => (
                                    <tr key={r.model_id} style={{ borderBottom: '1px solid var(--glass-border)' }}>
                                        <td style={{ padding: '0.5rem', fontWeight: 600, color: 'var(--text-muted)' }}>{idx + 1}</td>
                                        <td style={{ padding: '0.5rem', fontWeight: 600 }}><ModelTooltip name={r.name} meta={modelMeta[r.model_id]} /></td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--accent-green)' }}>{r.wins}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--accent-red)' }}>{r.losses}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>{r.draws}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center' }}>{r.total}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'right' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.5rem' }}>
                                                <div style={{ width: '50px', height: '4px', borderRadius: '2px', backgroundColor: 'var(--bg-primary)', overflow: 'hidden' }}>
                                                    <div style={{ width: `${r.win_rate * 100}%`, height: '100%', borderRadius: '2px', backgroundColor: r.win_rate >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)', transition: 'width 0.3s' }} />
                                                </div>
                                                <span style={{ fontWeight: 700, fontSize: '0.8rem', color: r.win_rate >= 0.6 ? 'var(--accent-green)' : r.win_rate <= 0.4 ? 'var(--accent-red)' : 'var(--text-primary)', minWidth: '40px', textAlign: 'right' }}>
                                                    {(r.win_rate * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Final Rankings table */}
                {rankings.length > 0 && (
                    <div className="panel" style={{ marginBottom: '1.5rem' }}>
                        <h2 style={{ fontSize: '1rem', marginBottom: '0.75rem' }}><Trophy size={18} /> Rankings (Bradley-Terry)</h2>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.05em' }}>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>#</th>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>Model</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>W</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>L</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>D</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>Games</th>
                                    <th style={{ textAlign: 'right', padding: '0.5rem' }}>Rating</th>
                                    <th style={{ textAlign: 'right', padding: '0.5rem' }}>Win Rate</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rankings.map((r, idx) => (
                                    <tr key={r.model_id} style={{ borderBottom: '1px solid var(--glass-border)' }}>
                                        <td style={{ padding: '0.5rem', fontWeight: 700, color: idx === 0 ? '#f59e0b' : idx === 1 ? '#94a3b8' : idx === 2 ? '#cd7f32' : 'var(--text-muted)' }}>
                                            {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : idx + 1}
                                        </td>
                                        <td style={{ padding: '0.5rem', fontWeight: 600 }}><ModelTooltip name={r.name} meta={modelMeta[r.model_id]} /></td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--accent-green)' }}>{r.wins}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--accent-red)' }}>{r.losses}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>{r.draws}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'center' }}>{r.total}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'right', fontWeight: 700, fontFamily: 'monospace', fontSize: '0.9rem', color: (r.rating ?? 1500) >= 1550 ? 'var(--accent-green)' : (r.rating ?? 1500) <= 1450 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
                                            {r.rating ?? '—'}
                                        </td>
                                        <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                                            {(r.win_rate * 100).toFixed(1)}%
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Head-to-head matrix */}
                {rankings.length > 0 && modelIds.length > 1 && (
                    <div className="panel">
                        <h2 style={{ fontSize: '1rem', marginBottom: '0.75rem' }}>Head-to-Head Matrix</h2>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                            Each cell shows <span style={{ color: 'var(--piece-red)' }}>Red wins</span> / <span style={{ color: 'var(--text-muted)' }}>Draws</span> / <span style={{ color: 'var(--text-secondary)' }}>White wins</span> (row = Red, col = White)
                        </p>
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                                <thead>
                                    <tr>
                                        <th style={{ padding: '0.4rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>Red ↓ / White →</th>
                                        {modelIds.map(id => (
                                            <th key={id} style={{ padding: '0.4rem', fontSize: '0.7rem', color: 'var(--text-muted)', maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                {modelNames[id]}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {modelIds.map(redId => (
                                        <tr key={redId}>
                                            <td style={{ padding: '0.4rem', fontWeight: 600, fontSize: '0.75rem', whiteSpace: 'nowrap' }}>{modelNames[redId]}</td>
                                            {modelIds.map(whiteId => {
                                                if (redId === whiteId) {
                                                    return <td key={whiteId} style={{ padding: '0.4rem', textAlign: 'center', backgroundColor: 'var(--bg-primary)', borderRadius: '0.25rem' }}>—</td>;
                                                }
                                                const h = getH2H(redId, whiteId);
                                                if (!h) {
                                                    return <td key={whiteId} style={{ padding: '0.4rem', textAlign: 'center', color: 'var(--text-muted)' }}>·</td>;
                                                }
                                                const total = h.red_wins + h.white_wins + h.draws;
                                                const redPct = total > 0 ? h.red_wins / total : 0;
                                                return (
                                                    <td key={whiteId} style={{
                                                        padding: '0.4rem',
                                                        textAlign: 'center',
                                                        borderRadius: '0.25rem',
                                                        backgroundColor: redPct > 0.6 ? 'rgba(239,68,68,0.15)' : redPct < 0.4 ? 'rgba(59,130,246,0.15)' : 'var(--bg-primary)'
                                                    }}>
                                                        <span style={{ color: 'var(--piece-red)' }}>{h.red_wins}</span>
                                                        <span style={{ color: 'var(--text-muted)' }}> / {h.draws} / </span>
                                                        <span style={{ color: 'var(--text-secondary)' }}>{h.white_wins}</span>
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Empty state */}
                {rankings.length === 0 && !running && (
                    <div className="panel" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                        <Trophy size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                        <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>No tournament results yet</p>
                        <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>Train 2+ models, then run a tournament to see rankings</p>
                    </div>
                )}
            </div>
        </div>
    );
};
