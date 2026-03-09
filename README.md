# Checkers Deep Learning Platform

An interactive educational web app where students can configure, train, and play against a two-headed Convolutional Neural Network (CNN) that evaluates Checkers board states.

Built with **PyTorch Lightning** and **FastAPI** on the backend, and **Vite**, **React**, and **Vanilla CSS** on the frontend.

## Features

- **Interactive Checkers Board** — Drag-and-drop gameplay powered by `@dnd-kit`, with full rule enforcement via the Python engine (mandatory jumps, kinging, multi-jumps).
- **Player Color Selection** — Choose to play as Black or White. White always moves first.
- **AI Opponent** — Minimax search with alpha-beta pruning. Optionally enhanced by a trained CNN evaluation function.
- **Game Over Detection** — A blurred overlay announces the winner when one side has no legal moves remaining.
- **CNN Brain Visualizer** — After training, the AI Brain panel displays the network's real-time win probability estimates for both sides.
- **Data Generation** — Self-play engine generates labeled training datasets (board state → game outcome).
- **Model Training** — Train a two-headed CNN via PyTorch Lightning, with live loss charts streamed over WebSocket.
- **Data Inspector** — Browse generated game data move-by-move with an interactive board viewer.

## Architecture

```
checker-bot/
├── backend/
│   ├── api/main.py          # FastAPI endpoints (generate, train, infer, validate_move)
│   ├── engine/board.py      # Core Checkers engine (rules, move generation, evaluation)
│   ├── engine/minimax.py    # Minimax + alpha-beta search
│   ├── model/cnn.py         # Two-headed CNN architecture
│   ├── model/lightning_module.py  # PyTorch Lightning training module
│   └── data/generator.py    # Self-play data generation
├── frontend/
│   ├── src/pages/Game.tsx    # Main dashboard layout
│   ├── src/components/
│   │   ├── Board.tsx         # Interactive drag-and-drop board
│   │   ├── BrainVisualizer.tsx  # CNN win probability display
│   │   ├── ConfigPanel.tsx   # Data generation & training controls
│   │   └── Metrics.tsx       # Live training loss chart
│   └── src/index.css         # Design system & global styles
├── PLAN.md                   # Original architecture & requirements
└── README.md
```

## Quickstart

### 1. Python Backend

The Python environment uses `uv` for dependency management.

```bash
# Create and activate the virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch pytorch-lightning fastapi uvicorn

# Run the API server
uvicorn backend.api.main:app --reload --port 8000
```

### 2. React Frontend

```bash
cd frontend
npm install
npm run dev
```

Navigate to the localhost port shown by `npm run dev` to interact with the application.

## Game Rules (Checkers)

- **Turn Order**: White always moves first.
- **Direction**: Black pieces start at the top (rows 0–2) and move down. White pieces start at the bottom (rows 5–7) and move up.
- **Movement**: Pieces move diagonally forward one square at a time.
- **Mandatory Jumps**: If a jump is available, it *must* be taken. This rule has been standard since 1535 (*Le Jeu Forcé*).
- **Multi-Jumps**: If additional jumps are available after landing, the same piece must continue jumping.
- **Kinging**: Reaching the far edge crowns a piece as a King. Kings can move diagonally in both directions. Kinging ends the turn.
- **Winning**: Capture all opponent pieces, or leave them with no legal moves.

## API Endpoints

| Method | Endpoint              | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| POST   | `/api/generate`       | Generate self-play training data                 |
| POST   | `/api/train`          | Start CNN training with PyTorch Lightning         |
| GET    | `/api/dataset`        | Retrieve generated dataset for the Data Inspector |
| POST   | `/api/validate_move`  | Validate a human move against the engine rules    |
| POST   | `/api/infer`          | Get AI's best move + CNN probabilities            |
| WS     | `/ws/metrics`         | Stream training loss metrics in real-time         |

## Documentation

- `PLAN.md`: Initial architecture and project requirements.