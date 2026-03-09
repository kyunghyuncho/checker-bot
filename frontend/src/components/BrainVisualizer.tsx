import React from 'react';
import { Activity } from 'lucide-react';

interface BrainProps {
    probabilities: {
        p_black: number;
        p_white: number;
    } | null;
}

export const BrainVisualizer: React.FC<BrainProps> = ({ probabilities }) => {
    return (
        <div className="panel" style={{ height: '100%' }}>
            <h2><Activity size={20} /> AI Brain</h2>

            <div className="edu-note" style={{ marginBottom: '2rem' }}>
                <strong>How it works:</strong> The Two-Headed CNN analyzes the spatial arrangement of the 8x8 board and outputs the probability of each side winning from the current state.
            </div>

            {probabilities ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: 600 }}>Black Win Probability</span>
                            <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>
                                {(probabilities.p_black * 100).toFixed(1)}%
                            </span>
                        </div>
                        {/* Custom Progress Bar */}
                        <div style={{ width: '100%', height: '12px', backgroundColor: 'var(--bg-primary)', borderRadius: '6px', overflow: 'hidden' }}>
                            <div style={{
                                width: `${probabilities.p_black * 100}%`,
                                height: '100%',
                                backgroundColor: 'var(--piece-black)',
                                borderRight: '2px solid #922b21',
                                transition: 'width 0.5s ease-out'
                            }} />
                        </div>
                    </div>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: 600 }}>White Win Probability</span>
                            <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>
                                {(probabilities.p_white * 100).toFixed(1)}%
                            </span>
                        </div>
                        {/* Custom Progress Bar */}
                        <div style={{ width: '100%', height: '12px', backgroundColor: 'var(--bg-primary)', borderRadius: '6px', overflow: 'hidden' }}>
                            <div style={{
                                width: `${probabilities.p_white * 100}%`,
                                height: '100%',
                                backgroundColor: 'var(--piece-white)',
                                borderRight: '2px solid #cbd5e1',
                                transition: 'width 0.5s ease-out'
                            }} />
                        </div>
                    </div>

                </div>
            ) : (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    No predictions available.<br />
                    <small>Train the model first to see AI evaluation.</small>
                </div>
            )}
        </div>
    );
};
