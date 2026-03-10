# Checkers Deep Learning Platform

An interactive educational web app where students can configure, train, and play against a two-headed Convolutional Neural Network (CNN) that evaluates Checkers board states.

Built with **PyTorch Lightning** and **FastAPI** on the backend, and **Vite**, **React**, and **Vanilla CSS** on the frontend.

## Features

- **Interactive Checkers Board** — Drag-and-drop gameplay powered by `@dnd-kit`, with full rule enforcement via the Python engine (mandatory jumps, kinging, multi-jumps).
- **AI Arena Mode** — Assign trained models to play as Red, White, or both. Supports Human vs Human, Human vs AI, and AI vs AI matchups.
- **AI Opponent** — Minimax search with alpha-beta pruning. Optionally enhanced by a trained CNN evaluation function.
- **Epsilon Slider** — Control AI move randomness (ε 0.00–0.30) for varied AI vs AI games.
- **Model Registry** — Train, save, and manage multiple CNN models. Assign any model to either side via the arena panel.
- **Game Over Detection** — A blurred overlay announces the winner when one side has no legal moves remaining.
- **CNN Brain Visualizer** — After training, the AI Brain panel displays the network's real-time win probability estimates for both sides.
- **Data Generation** — Self-play engine generates labeled training datasets (board state → game outcome).
- **Model Training** — Train a two-headed CNN via PyTorch Lightning, with live loss charts (train + validation) streamed over WebSocket.
- **Early Stopping** — Configurable validation split and patience for early stopping during training.
- **Data Inspector** — Browse generated game data move-by-move with an interactive board viewer.
- **Collapsible Panels** — Data generation and training controls collapse to save space.

## Architecture

```
checker-bot/
├── backend/
│   ├── api/main.py                # FastAPI server + model registry + arena inference
│   ├── engine/board.py            # Core Checkers engine (rules, move generation, evaluation)
│   ├── engine/minimax.py          # Minimax + alpha-beta pruning + epsilon-greedy
│   ├── model/cnn.py               # Two-headed CNN architecture (5-channel input)
│   ├── model/lightning_module.py  # PyTorch Lightning training module + dataset
│   ├── data/generator.py          # Self-play data generation with outcome labeling
│   └── models/                    # Saved model checkpoints + metadata (auto-created)
├── frontend/
│   ├── src/pages/
│   │   ├── Game.tsx               # Main dashboard (board + controls + brain)
│   │   └── DataInspector.tsx      # Dataset viewer page
│   ├── src/components/
│   │   ├── Board.tsx              # Interactive drag-and-drop board (DnD Kit)
│   │   ├── BrainVisualizer.tsx    # CNN win probability display
│   │   ├── ConfigPanel.tsx        # Collapsible data gen & training controls
│   │   ├── Metrics.tsx            # Live training loss chart (Recharts)
│   │   └── ModelRegistry.tsx      # Model arena panel with Red/White assignment
│   └── src/index.css              # Design system & global styles
├── PLAN.md                        # Original architecture & requirements
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
- **Direction**: Red pieces start at the top (rows 0–2) and move down. White pieces start at the bottom (rows 5–7) and move up.
- **Movement**: Pieces move diagonally forward one square at a time.
- **Mandatory Jumps**: If a jump is available, it *must* be taken. This rule has been standard since 1535 (*Le Jeu Forcé*).
- **Multi-Jumps**: If additional jumps are available after landing, the same piece must continue jumping.
- **Kinging**: Reaching the far edge crowns a piece as a King. Kings can move diagonally in both directions. Kinging ends the turn.
- **Winning**: Capture all opponent pieces, or leave them with no legal moves.

## Minimax AI Engine

The project powers its AI bots using a pure **Minimax Search Algorithm** enhanced by **Alpha-Beta Pruning**, optionally guided by the CNN.

### 1. The Game Tree Search
Minimax is a decision-making algorithm that looks ahead into the future of the game by building a "game tree." 
- It simulates every possible legal move the current player can make.
- Then, for each of those moves, it simulates every possible response the opponent can make.
- It repeats this alternating process up to a strict **Search Depth** limit (e.g., Depth 4 means looking 4 full turns into the future).

The algorithm assumes both players are playing perfectly. The AI's goal is to **maximize** its own score, while assuming the opponent will always make the move that **minimizes** the AI's score. It works backward from the deepest nodes of the tree to pick the branch that guarantees the best possible outcome.

### 2. Alpha-Beta Pruning
Because the number of possible board states explodes exponentially with every step forward (the "branching factor"), looking even a few turns ahead can require millions of simulations.
**Alpha-Beta Pruning** drastically speeds up Minimax by abandoning branches of the tree that are mathematically proven to be worse than a move previously examined. If the AI finds a move that gives it +5 points, and while checking a different branch it sees the opponent can force a move that leaves the AI with -2 points, it instantly stops calculating that entire branch.

### 3. Board Evaluation (The Leaf Nodes)
When Minimax reaches its search depth limit (e.g., exactly 4 turns into the future), it can't simulate anymore. It must look at the board state and guess who is winning. It assigns a number to the board (positive = Red winning, negative = White winning). The AI supports two ways of scoring these "leaf nodes":
* **Hardcoded Heuristic (Fallback):** If no AI model is loaded, the engine simply counts pieces on the board. A regular piece is worth 1.0, a King is worth 1.5, and controlling the center 4 squares is worth an extra 0.5.
* **The "Brain" (CNN Model):** If a trained neural network is assigned, the board is converted into a PyTorch tensor and fed into the CNN. The CNN outputs two probabilities: `P(Red)` and `P(White)`. The board is then scored mathematically using the difference between these two probabilities, essentially letting the neural network "feel" which side is winning based on its trained experience, rather than just blindly counting pieces.

### 4. Epsilon-Greedy Randomness
To prevent the exact same game from playing out repeatedly, the AI employs an **Epsilon (ε)** parameter. 
Before executing an optimal Minimax search, the engine rolls a multi-sided die. If the roll is less than ε (for example, a 5% chance if ε=0.05), the AI completely ignores the search tree and simply picks a legal move at random. This allows for vast variety in AI vs AI matchups and generates diverse, non-repetitive data during self-play data generation.

## API Endpoints

| Method | Endpoint              | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| POST   | `/api/generate`       | Generate self-play training data                 |
| POST   | `/api/train`          | Start CNN training with PyTorch Lightning         |
| GET    | `/api/dataset`        | Retrieve generated dataset for the Data Inspector |
| POST   | `/api/validate_move`  | Validate a human move against the engine rules    |
| POST   | `/api/infer`          | Get AI's best move + CNN probabilities            |
| GET    | `/api/models`         | List all saved models with metadata               |
| DELETE | `/api/models/{id}`    | Delete a saved model                             |
| WS     | `/ws/metrics`         | Stream training loss metrics in real-time         |

## Documentation

- `PLAN.md`: Initial architecture and project requirements.