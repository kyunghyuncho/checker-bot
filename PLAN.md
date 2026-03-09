# PLAN.md: Interactive Checkers Deep Learning Platform

## 1. Project Overview & Architecture
* **Goal:** Build an interactive educational web app where students can configure, train, and play against a two-headed Convolutional Neural Network (CNN) that evaluates Checkers board states.
* **Target Environment:** Local execution on standard student laptops (CPU or entry-level GPU).
* **Tech Stack:**
    * **Backend & ML Engine:** Python, PyTorch, PyTorch Lightning.
    * **API Layer:** FastAPI (handles training streams, game state validation, and inference).
    * **Frontend:** React (Next.js or Vite) with Recharts (for live loss curves) and `dnd-kit` (for drag-and-drop board interactions).

---

## 2. Data Strategy: Sourcing, Generation & Inspection
To ensure students have enough data without overwhelming their local machines, the platform will support both synthetic generation and real-world data parsing.

### A. Synthetic Data Generation (Self-Play)
Students will use a classical search algorithm to generate custom datasets.
* **Algorithm:** Minimax with Alpha-Beta pruning. 

* **Heuristic Evaluation:** A hand-crafted function to evaluate leaf nodes:
  $$V(s) = w_1(P_{black} - P_{white}) + w_2(K_{black} - K_{white}) + w_3(C_{black} - C_{white}) + w_4(T_{white} - T_{black})$$
  *(Where $P$ = pieces, $K$ = kings, $C$ = center control, $T$ = threatened pieces)*
* **Ensuring Diversity:** To prevent the engine from playing the exact same game repeatedly:
    * **Randomized Openings:** Force the first 2-4 plies to be chosen randomly from legal moves.
    * **$\epsilon$-Greedy Play:** Introduce a small probability $\epsilon$ of picking a random move during self-play to force the network to learn from suboptimal and messy board states.
* **Labeling:** Backpropagate the terminal state (Win/Loss) to all board state tensors in that trajectory.

### B. Real-World Data Parsing
* **Sourcing:** Provide scripts to parse Portable Draughts Notation (PDN) files from open databases like Lidraughts.org or the ACF archives.
* **Handling Draws:** Expert games have a high draw rate. Configuration options will allow students to either:
    1. Filter out draws completely (to use standard Binary Cross Entropy).
    2. Map draws to a target label of $0.5$ and use Mean Squared Error (MSE) for those batches.

### C. Data Inspector (UI)
* A dedicated tab in the frontend allowing students to page through the generated tensors, viewing the 2D board state alongside its ground-truth labels (e.g., `Black Win: 1, White Win: 0`).

---

## 3. Neural Network Architecture (The "Two-Headed" CNN)
The network must be shallow enough to train on a laptop but deep enough to demonstrate feature extraction.



* **Input Representation:** An $8 \times 8 \times C$ tensor. Channels ($C$) represent: Black pieces, White pieces, Black Kings, White Kings, and Turn Indicator.
* **Shared Backbone:** Configurable 2-4 convolutional layers (ReLU activations, Batch Normalization) to learn spatial configurations.
* **Prediction Heads:**
    * **Head 1 (Black Win):** Fully connected layers $\rightarrow$ Sigmoid output ($P_{black}$).
    * **Head 2 (White Win):** Fully connected layers $\rightarrow$ Sigmoid output ($P_{white}$).
* **Loss Function:** Binary Cross Entropy (BCE) for both heads.

---

## 4. AI Agent & Search Implementation
The trained network acts as the heuristic evaluator. 

* **Greedy Search (1-Step):** Generate all legal next moves, evaluate them through the CNN, and pick the move that maximizes $P_{black} - P_{white}$. Fast, but susceptible to multi-jump traps.
* **Shallow Search (2-Step Lookahead):** Build a depth-2 tree and evaluate leaf nodes.
* **Stochastic Temperature:** Apply a softmax function with a temperature parameter ($\tau$) to the network's output probabilities. A higher $\tau$ forces the AI to occasionally select the 2nd or 3rd best move, creating a more dynamic opponent.

---

## 5. System Interfaces

### Backend Endpoints (FastAPI)
| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/generate` | POST | Triggers synthetic data generation with specific params ($\epsilon$, depth). |
| `/api/train` | POST | Accepts hyperparams and starts PyTorch Lightning `Trainer`. |
| `/ws/metrics` | WS | Websocket streaming live training/val loss and epoch progress. |
| `/api/infer` | POST | Accepts current board state, returns AI's selected move + confidence scores. |

### Frontend Components (React)
* **Configuration Panel:** Sliders/inputs for CNN layers, filters, learning rate, and search depth.
* **Live Metrics:** Real-time line charts plotting training vs. validation loss.
* **Interactive Game Board:** Professionally styled 8x8 grid enforcing Checkers rules (forced jumps, kinging).
* **AI "Brain" Visualizer:** A side-panel active during the AI's turn displaying the top 3 considered moves and the network's probability scores, demystifying the model's decisions.

---

## 6. Execution Phases
1. **Phase 1: Game Engine.** Implement Python Checkers logic (board representation, move validation, win conditions).
2. **Phase 2: Data Pipeline.** Build the Minimax self-play script and the PDN parser.
3. **Phase 3: The Model.** Author the PyTorch Lightning module with dynamic hyperparameter injection.
4. **Phase 4: API Layer.** Wrap the engine and model in FastAPI; implement Websockets for metrics.
5. **Phase 5: Frontend UI.** Build the React dashboard, data inspector, and interactive board.