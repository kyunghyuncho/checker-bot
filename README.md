# Checkers Deep Learning Platform

An interactive educational web app where students can configure, train, and play against a two-headed Convolutional Neural Network (CNN) that evaluates Checkers board states.

Built with **PyTorch Lightning** and **FastAPI** on the backend, and **Vite**, **React**, and **Vanilla CSS** on the frontend.

## Features

- **Interactive Checkers Board** — Drag-and-drop gameplay powered by `@dnd-kit`, with full rule enforcement via the Python engine (mandatory jumps, kinging, multi-jumps).
- **AI Arena Mode** — Assign trained models to play as Red, White, or both. Supports Human vs Human, Human vs AI, and AI vs AI matchups.
- **AI Opponent** — Minimax search with alpha-beta pruning, enhanced by a trained residual CNN evaluation function.
- **Softmax Move Sampling** — Control AI exploration via temperature τ. At τ=0 the AI plays greedily; higher τ favors diverse, stochastic play.
- **Model Registry** — Train, save, and manage multiple CNN models. Assign any model to either side via the arena panel.
- **Game Over Detection** — A blurred overlay announces the winner when one side has no legal moves remaining.
- **CNN Brain Visualizer** — After training, the AI Brain panel displays each assigned model's real-time win probability estimates independently (Red AI's thoughts vs White AI's thoughts).
- **Data Generation** — Self-play engine generates labeled training datasets (board state → game outcome).
- **Exponential Label Discounting** — Configurable discount factor (γ) that smooths training labels: early-game positions are labeled closer to 0.5 (uncertain), while late-game positions retain the true outcome. This dramatically improves CNN training quality.
- **Model Training** — Train a two-headed CNN via PyTorch Lightning, with live loss charts (train + validation) streamed over WebSocket.
- **Stop Training** — Gracefully interrupt an ongoing training run at any time. The model saves its current progress and the UI returns to idle.
- **Early Stopping** — Configurable validation split and patience for early stopping during training.
- **Game Monitor** — Real-time statistics panel below the board showing piece counts, material advantage, legal move count, and two sparkline charts: material balance and CNN win probability over time.
- **Configurable Search Depth** — Adjust the minimax search depth (1–8) for interactive play.
- **Data Inspector** — Browse generated game data move-by-move with an interactive board viewer.
- **AI Tournament** — Pit N trained models against each other in M random games. View rankings with win rates and a color-coded head-to-head matrix.
- **Collapsible Panels** — Data generation and training controls collapse to save space.

## Architecture

```
checker-bot/
├── backend/
│   ├── api/main.py                # FastAPI server + model registry + arena inference
│   ├── engine/board.py            # Core Checkers engine (rules, move generation, evaluation)
│   ├── engine/minimax.py          # Minimax + alpha-beta pruning + softmax sampling
│   ├── model/cnn.py               # Two-headed residual CNN architecture (5-channel input)
│   ├── model/lightning_module.py  # PyTorch Lightning training module + dataset
│   ├── data/generator.py          # Self-play data generation with outcome labeling
│   └── models/                    # Saved model checkpoints + metadata (auto-created)
├── frontend/
│   ├── src/pages/
│   │   ├── Game.tsx               # Main dashboard (board + controls + brain)
│   │   ├── DataInspector.tsx      # Dataset viewer page
│   │   └── Tournament.tsx        # AI tournament with rankings & H2H matrix
│   ├── src/components/
│   │   ├── Board.tsx              # Interactive drag-and-drop board (DnD Kit)
│   │   ├── BrainVisualizer.tsx    # Dual CNN win probability display
│   │   ├── ConfigPanel.tsx        # Collapsible data gen & training controls
│   │   ├── GameMonitor.tsx        # Real-time game stats & sparkline charts
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

### 4. Softmax (Boltzmann) Sampling
To produce diverse games and prevent repetitive play, the AI uses **softmax sampling** over its minimax-evaluated moves. Instead of always picking the single best move, each legal move's minimax score is converted into a probability via the softmax function, controlled by a **temperature (τ)** parameter:
- **τ = 0**: Purely greedy — always the highest-scoring move.
- **τ = 1**: Balanced exploration — good moves are likely but weaker alternatives still occur.
- **τ → ∞**: Uniform random — all legal moves become equally likely.

This is a major improvement over the previous epsilon-greedy strategy, which randomly selected a *completely random* move with probability ε. Softmax sampling ensures that even when exploring, the AI favors *reasonable* alternatives rather than catastrophic blunders.

### 5. AI Tournament & Elo Ratings (Bradley-Terry Model)
When running an **AI Tournament**, the system pits models against each other in random pairings. To determine who is truly the best, we calculate Elo ratings for every participant based on the tournament results.

Instead of traditional sequential Elo updates (which are dependent on match order), the server calculates ratings by iteratively solving the **Bradley-Terry (BT) model**.
The BT model assumes the probability of model $i$ beating model $j$ is defined by their innate "skills" $s_i$ and $s_j$:
$$P(i \text{ beats } j) = \frac{s_i}{s_i + s_j}$$

To solve for these skills given a matrix of win/loss outcomes, we use Minorize-Maximization (MM) or iterative proportional fitting:
1. Initialize all skills $s_i = 1.0$.
2. For each model $i$, iterate:
   $$s_i^{(new)} = \frac{W_i}{\sum_{j \neq i} \frac{N_{ij}}{s_i + s_j}}$$
   *(Where $W_i$ is total wins for model $i$, and $N_{ij}$ is total games between $i$ and $j$. Draws count as 0.5 wins for both.)*
3. Repeat step 2 until the skills converge.
4. Convert those abstract skills into standard Elo ratings: $\text{Elo}_i = 400 \log_{10}(s_i) + 1200$.

This mathematically robust approach guarantees that the final leaderboard accurately reflects the true relative strengths of the models globally.

## API Endpoints

| Method | Endpoint              | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| POST   | `/api/generate`       | Generate self-play training data                 |
| POST   | `/api/train`          | Start CNN training with PyTorch Lightning         |
| POST   | `/api/train/stop`     | Gracefully stop an ongoing training run           |
| GET    | `/api/dataset`        | Retrieve generated dataset for the Data Inspector |
| POST   | `/api/validate_move`  | Validate a human move against the engine rules    |
| POST   | `/api/infer`          | Get AI's best move for the current board          |
| POST   | `/api/evaluate`       | Get both models' win probabilities for a board    |
| GET    | `/api/models`         | List all saved models with metadata               |
| DELETE | `/api/models/{id}`    | Delete a saved model                             |
| WS     | `/ws/metrics`         | Stream training loss & status in real-time        |

## Documentation

- `PLAN.md`: Initial architecture and project requirements.