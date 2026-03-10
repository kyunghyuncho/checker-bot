import React from 'react';

export interface ModelMeta {
    id?: string;
    hidden_dims?: number;
    num_conv_layers?: number;
    dropout_rate?: number;
    learning_rate?: number;
    discount_factor?: number;
    epochs_trained?: number;
    batch_size?: number;
    final_train_loss?: number;
    final_val_loss?: number;
}

interface ModelTooltipProps {
    name: string;
    meta?: ModelMeta;
}

export const ModelTooltip: React.FC<ModelTooltipProps> = ({ name, meta }) => {
    if (!meta) return <>{name}</>;

    // Special case for the built-in heuristic agent
    if (meta.id === 'heuristic_agent') {
        return (
            <span style={{ position: 'relative', cursor: 'help', borderBottom: '1px dotted var(--text-muted)' }} className="model-tooltip">
                {name}
                <span className="model-tooltip-content" style={{ whiteSpace: 'normal', width: '220px' }}>
                    <strong>{name}</strong>
                    <div style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
                        Traditional Minimax search engine using a handwritten material-advantage heuristic (Pieces + Kings).
                    </div>
                </span>
            </span>
        );
    }

    if (meta.id === 'random_agent') {
        return (
            <span style={{ position: 'relative', cursor: 'help', borderBottom: '1px dotted var(--text-muted)' }} className="model-tooltip">
                {name}
                <span className="model-tooltip-content" style={{ whiteSpace: 'normal', width: '220px' }}>
                    <strong>{name}</strong>
                    <div style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
                        Picks a completely random valid move. Serves as the absolute worst-case baseline.
                    </div>
                </span>
            </span>
        );
    }

    return (
        <span style={{ position: 'relative', cursor: 'help', borderBottom: '1px dotted var(--text-muted)' }} className="model-tooltip">
            {name}
            <span className="model-tooltip-content">
                <strong>{name}</strong>
                <br />Layers: {meta.num_conv_layers ?? '?'} × {meta.hidden_dims ?? '?'}ch
                <br />Dropout: {meta.dropout_rate ?? '?'}
                <br />LR: {meta.learning_rate ?? '?'}
                <br />Discount γ: {meta.discount_factor ?? '?'}
                <br />Epochs: {meta.epochs_trained ?? '?'}
                <br />Batch: {meta.batch_size ?? '?'}
                {meta.final_train_loss != null && <><br />Train loss: {meta.final_train_loss}</>}
                {meta.final_val_loss != null && <><br />Val loss: {meta.final_val_loss}</>}
            </span>
        </span>
    );
};
