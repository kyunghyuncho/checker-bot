import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { BarChart2 } from 'lucide-react';

interface MetricPoint {
    epoch: number;
    train_loss: number;
    val_loss?: number | null;
}

export const Metrics = () => {
    const [data, setData] = useState<MetricPoint[]>([]);
    const [status, setStatus] = useState<string>('Idle');

    useEffect(() => {
        // Connect to FastAPI WebSocket
        const ws = new WebSocket('ws://localhost:8000/ws/metrics');

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'metric') {
                setStatus('Training...');
                setData(prev => [...prev, {
                    epoch: msg.epoch,
                    train_loss: msg.train_loss,
                    val_loss: msg.val_loss ?? undefined
                }]);
            } else if (msg.type === 'status') {
                setStatus(msg.message);
            }
        };

        return () => {
            ws.close();
        };
    }, []);

    return (
        <div className="panel" style={{ flexGrow: 1 }}>
            <h2><BarChart2 size={20} /> Live Training Metrics</h2>
            <div style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>
                Status: <strong style={{ color: status === 'Training...' ? 'var(--accent-green)' : 'var(--text-primary)' }}>{status}</strong>
            </div>

            <div style={{ width: '100%', height: '200px' }}>
                {data.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" />
                            <XAxis dataKey="epoch" stroke="var(--text-muted)" />
                            <YAxis domain={['auto', 'auto']} stroke="var(--text-muted)" />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--glass-border)', borderRadius: '0.5rem' }}
                                itemStyle={{ color: 'var(--text-primary)' }}
                            />
                            <Legend wrapperStyle={{ color: 'var(--text-secondary)' }} />
                            <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="var(--accent-purple)" strokeWidth={3} dot={false} />
                            <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="var(--accent-blue)" strokeWidth={3} dot={false} strokeDasharray="5 5" />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                        Start training to see loss curves.
                    </div>
                )}
            </div>
        </div>
    );
};
