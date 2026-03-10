/**
 * BrainVisualizer.tsx — CNN Win Probability Display
 * ===================================================
 * Displays the Two-Headed CNN's output: P(Red wins) and P(White wins)
 * as animated horizontal progress bars for both assigned models.
 */
import React from 'react';
import { Activity } from 'lucide-react';

export interface DualProbabilities {
    red_eval: { p_black: number, p_white: number } | null;
    white_eval: { p_black: number, p_white: number } | null;
}

interface BrainProps {
    dualProbabilities: DualProbabilities | null;
    redAssigned: boolean;
    whiteAssigned: boolean;
}

const ProbabilityBar = ({ pBlack, pWhite, title, colorPrimary, colorSecondary, borderPrimary, borderSecondary }: any) => {
    return (
        <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>{title}</h3>

            <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                    <span style={{ fontWeight: 600 }}>Red Win Prob</span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>
                        {(pBlack * 100).toFixed(1)}%
                    </span>
                </div>
                <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--bg-primary)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                        width: `${pBlack * 100}%`, height: '100%', backgroundColor: colorPrimary,
                        borderRight: `2px solid ${borderPrimary}`, transition: 'width 0.5s ease-out'
                    }} />
                </div>
            </div>

            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                    <span style={{ fontWeight: 600 }}>White Win Prob</span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>
                        {(pWhite * 100).toFixed(1)}%
                    </span>
                </div>
                <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--bg-primary)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                        width: `${pWhite * 100}%`, height: '100%', backgroundColor: colorSecondary,
                        borderRight: `2px solid ${borderSecondary}`, transition: 'width 0.5s ease-out'
                    }} />
                </div>
            </div>
        </div>
    );
};

export const BrainVisualizer: React.FC<BrainProps> = ({ dualProbabilities, redAssigned, whiteAssigned }) => {
    if (!redAssigned && !whiteAssigned) {
        return (
            <div className="panel" style={{ height: '100%' }}>
                <h2><Activity size={20} /> AI Brain</h2>
                <div className="edu-note" style={{ marginBottom: '2rem' }}>
                    <strong>How it works:</strong> The Two-Headed CNN analyzes the spatial arrangement of the 8x8 board and outputs the probability of each side winning from the current state.
                </div>
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    No AI models currently active.<br />
                    <small>Assign a model to Red or White to see its thoughts.</small>
                </div>
            </div>
        );
    }

    return (
        <div className="panel" style={{ height: '100%' }}>
            <h2><Activity size={20} /> AI Brain</h2>

            <div className="edu-note" style={{ marginBottom: '2rem' }}>
                <strong>How it works:</strong> The Two-Headed CNN analyzes the spatial arrangement of the 8x8 board and outputs the probability of each side winning from the current state.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {redAssigned && (
                    dualProbabilities?.red_eval ? (
                        <ProbabilityBar
                            title="Red AI's Evaluation"
                            pBlack={dualProbabilities.red_eval.p_black}
                            pWhite={dualProbabilities.red_eval.p_white}
                            colorPrimary="var(--piece-red)" borderPrimary="#922b21"
                            colorSecondary="var(--piece-white)" borderSecondary="#cbd5e1"
                        />
                    ) : (
                        <div style={{ padding: '1rem', color: 'var(--text-muted)', textAlign: 'center' }}>Loading Red AI thoughts...</div>
                    )
                )}

                {whiteAssigned && (
                    dualProbabilities?.white_eval ? (
                        <ProbabilityBar
                            title="White AI's Evaluation"
                            pBlack={dualProbabilities.white_eval.p_black}
                            pWhite={dualProbabilities.white_eval.p_white}
                            colorPrimary="var(--piece-red)" borderPrimary="#922b21"
                            colorSecondary="var(--piece-white)" borderSecondary="#cbd5e1"
                        />
                    ) : (
                        <div style={{ padding: '1rem', color: 'var(--text-muted)', textAlign: 'center' }}>Loading White AI thoughts...</div>
                    )
                )}
            </div>
        </div>
    );
};
