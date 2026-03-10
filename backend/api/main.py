"""
Checkers Deep Learning API Server
===================================
FastAPI backend that provides:
    1. Self-play data generation     (POST /api/generate)
    2. CNN model training            (POST /api/train)
    3. AI inference / move selection  (POST /api/infer)
    4. Move validation               (POST /api/validate_move)
    5. Dataset inspection            (GET  /api/dataset)
    6. Model registry CRUD           (GET  /api/models, DELETE /api/models/{id})
    7. Real-time metrics streaming    (WS   /ws/metrics)

The server is stateless per-request except for:
    - model_cache:       In-memory cache of loaded model checkpoints
    - active_websockets: Connected WebSocket clients for live metric streaming
    - is_training:       Lock to prevent concurrent training runs

All training and data generation run in background tasks so the API remains
responsive during long-running operations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import os
import torch
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping

# Local imports
from backend.data.generator import generate_dataset
from backend.model.lightning_module import CheckersDataset, CheckersLightningModule
from backend.model.cnn import board_to_tensor
from backend.engine.board import CheckersBoard, BLACK, WHITE
from backend.engine.minimax import get_best_move

# ── App Initialization ──────────────────────────────────────────────

app = FastAPI(title="Checkers Deep Learning API")

# Allow requests from the Vite frontend (development: any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ─────────────────────────────────────────────────────

active_websockets: list = []     # Connected WebSocket clients for metric streaming
is_training: bool = False        # Lock: only one training job can run at a time
stop_training: bool = False      # Flag to signal training to stop gracefully
stop_tournament: bool = False    # Flag to signal tournament to stop
model_cache: dict = {}           # In-memory cache: model_id → CheckersLightningModule

# Directory where .ckpt and .meta.json files are persisted
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "backend", "models"
)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helper Functions ─────────────────────────────────────────────────

def _get_model_list():
    """
    Scan the models directory for .meta.json files and return their contents.
    Each meta file contains the model's ID, creation date, hyperparameters,
    and final loss values.
    """
    models = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith(".meta.json"):
            with open(os.path.join(MODELS_DIR, f), "r") as fh:
                models.append(json.load(fh))
    return models


def _load_model_by_id(model_id: str):
    """
    Load a model checkpoint by ID, using the in-memory cache when possible.

    Returns the CheckersLightningModule in eval mode, or None if not found.
    """
    # Check cache first to avoid redundant disk I/O
    if model_id in model_cache:
        return model_cache[model_id]

    ckpt_path = os.path.join(MODELS_DIR, f"{model_id}.ckpt")
    meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        return None

    # Load from disk and cache for future requests
    model = CheckersLightningModule.load_from_checkpoint(ckpt_path)
    model.eval()  # Set to inference mode (disables dropout, batch norm uses running stats)
    model_cache[model_id] = model
    return model


# ── Request / Response Models ────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Parameters for self-play data generation."""
    num_games: int = 50                          # Number of games to simulate
    depth: int = 2                               # Minimax search depth
    temperature: float = 1.0                     # Softmax temperature for move sampling
    output_file: str = "backend/data/dataset.json"  # Output JSON path


class TrainRequest(BaseModel):
    """Parameters for model training."""
    dataset_file: str = "backend/data/dataset.json"  # Input dataset path
    epochs: int = 20                              # Maximum training epochs
    learning_rate: float = 0.001                  # Adam optimizer learning rate
    hidden_dims: int = 64                         # CNN filter count / hidden dims
    num_conv_layers: int = 3                      # Number of conv blocks
    dropout_rate: float = 0.1                     # Dropout probability in FC heads
    batch_size: int = 32                          # Training batch size
    val_split: float = 0.1                        # Fraction of data for validation
    patience: int = 5                             # Early stopping patience (0 = disabled)
    discount_factor: float = 0.05                  # Label discounting γ: 0=none, 1=max


class InferRequest(BaseModel):
    """Parameters for AI move inference."""
    board_state: list[list[int]]                  # Complete 8x8 grid of integers
    current_turn: int                             # 1 (Black/Red) or 2 (White)
    depth: int = 1                                # Minimax search depth
    model_id: str | None = None                   # ID of the model playing this turn
    temperature: float = 0.0                      # Softmax temperature (0 = greedy)


class EvaluateRequest(BaseModel):
    """Parameters for dual-model board evaluation."""
    board_state: list[list[int]]                  # Complete 8x8 grid of integers
    current_turn: int                             # 1 (Black/Red) or 2 (White)
    red_model_id: str | None = None               # ID of the model assigned to Red
    white_model_id: str | None = None             # ID of the model assigned to White


class MoveValidationRequest(BaseModel):
    """Parameters for validating a human player's attempted move."""
    board_state: list[list[int]]         # Current board grid
    current_turn: int                    # Whose turn it is
    start_r: int                         # Piece starting row
    start_c: int                         # Piece starting column
    end_r: int                           # Target landing row
    end_c: int                           # Target landing column


# ── WebSocket Metrics Callback ───────────────────────────────────────

class WebSocketMetricsCallback(Callback):
    """
    PyTorch Lightning Callback that broadcasts training metrics to all
    connected WebSocket clients at the end of each epoch.

    This bridges the synchronous Lightning training loop (running in a
    background thread) with the async FastAPI WebSocket connections.
    """

    def __init__(self, main_loop):
        """
        Args:
            main_loop: The asyncio event loop of the main FastAPI thread.
                       Used to schedule WebSocket sends from the training thread.
        """
        self.main_loop = main_loop

    def on_validation_epoch_end(self, trainer, pl_module):
        """Send epoch metrics to all connected WebSocket clients."""
        metrics = trainer.callback_metrics

        # Get training loss (Lightning uses different keys depending on version)
        train_loss_val = metrics.get("train_loss_epoch")
        if train_loss_val is None:
            train_loss_val = metrics.get("train_loss")

        # Skip the sanity-check validation that Lightning runs before training starts
        # (during sanity check, train_loss hasn't been computed yet)
        if train_loss_val is None:
            return

        val_loss_val = metrics.get("val_loss")

        payload = {
            "epoch": trainer.current_epoch,
            "train_loss": round(train_loss_val.item(), 4),
            "val_loss": round(val_loss_val.item(), 4) if val_loss_val is not None else None,
            "type": "metric"
        }

        # Broadcast to all connected clients
        # run_coroutine_threadsafe bridges the training thread → async event loop
        for ws in active_websockets:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps(payload)),
                self.main_loop
            )


# ═══════════════════════════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════════════════════════

# ── Data Generation ──────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Trigger self-play data generation in a background thread.
    
    Streams progress updates to WebSocket clients during generation.
    The resulting dataset is saved to the specified output file.
    """
    main_loop = asyncio.get_running_loop()

    def _run_gen():
        def progress(current, total):
            """Callback invoked after each batch of games completes."""
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({
                        "type": "status",
                        "message": f"Generating data: {current}/{total} games..."
                    })),
                    main_loop
                )

        try:
            generate_dataset(req.num_games, req.output_file, req.depth, req.temperature, progress)
            print("Dataset generation complete.")
            # Notify frontend of completion
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({
                        "type": "status",
                        "message": "Data Generation Complete!"
                    })),
                    main_loop
                )
        except Exception as e:
            print(f"Error generating dataset: {e}")

    background_tasks.add_task(_run_gen)
    return {"message": f"Started generating {req.num_games} games locally."}


# ── Model Training ───────────────────────────────────────────────────

@app.post("/api/train")
async def api_train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start model training in a background thread.

    Only one training run is allowed at a time (guarded by is_training flag).
    Streams epoch-level loss metrics to WebSocket clients during training.
    On completion, saves the model checkpoint + metadata and notifies the frontend.
    """
    global is_training, stop_training
    if is_training:
        raise HTTPException(status_code=400, detail="Training is already in progress.")

    main_loop = asyncio.get_running_loop()

    def _run_train():
        global is_training, stop_training
        is_training = True
        stop_training = False
        
        # Notify the frontend immediately so it can reset the metrics chart
        for ws in active_websockets:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps({"type": "training_started", "message": "Initializing training..."})),
                main_loop
            )

        try:
            # ── 1. Load and split dataset ────────────────────────────
            dataset = CheckersDataset(req.dataset_file, discount_factor=req.discount_factor)
            from torch.utils.data import DataLoader, random_split

            total = len(dataset)
            val_size = max(1, int(total * req.val_split))  # At least 1 validation sample
            train_size = total - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=req.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=req.batch_size, shuffle=False)

            # ── 2. Initialize a fresh model ──────────────────────────
            model = CheckersLightningModule(
                learning_rate=req.learning_rate,
                hidden_dims=req.hidden_dims,
                num_conv_layers=req.num_conv_layers,
                dropout_rate=req.dropout_rate
            )

            # ── 3. Configure trainer ─────────────────────────────────
            class CancelTrainingCallback(Callback):
                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    global stop_training
                    if stop_training:
                        print("Stop training requested! Halting epoch...")
                        trainer.should_stop = True

            callbacks = [WebSocketMetricsCallback(main_loop), CancelTrainingCallback()]
            if req.patience > 0:
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=req.patience,
                    mode='min',
                    verbose=True
                ))

            trainer = Trainer(
                max_epochs=req.epochs,
                callbacks=callbacks,
                enable_checkpointing=False,  # We handle saving manually below
                logger=False                 # Disable default TensorBoard logger
            )

            # ── 4. Train ─────────────────────────────────────────────
            print("Starting training loop...")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # ── 5. Save checkpoint and metadata to disk ──────────────
            model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(MODELS_DIR, f"{model_id}.ckpt")
            trainer.save_checkpoint(ckpt_path)

            # Cache the trained model for immediate inference
            model.eval()
            model_cache[model_id] = model

            # Build metadata record
            meta = {
                "id": model_id,
                "created_at": datetime.now().isoformat(),
                "epochs_trained": trainer.current_epoch + 1,
                "hidden_dims": req.hidden_dims,
                "num_conv_layers": req.num_conv_layers,
                "dropout_rate": req.dropout_rate,
                "learning_rate": req.learning_rate,
                "batch_size": req.batch_size,
            }

            # Include final loss values if available
            final_metrics = trainer.callback_metrics
            tl = final_metrics.get("train_loss")
            vl = final_metrics.get("val_loss")
            if tl is not None:
                meta["final_train_loss"] = round(tl.item(), 4)
            if vl is not None:
                meta["final_val_loss"] = round(vl.item(), 4)

            with open(os.path.join(MODELS_DIR, f"{model_id}.meta.json"), "w") as mf:
                json.dump(meta, mf, indent=2)

            print(f"Training complete! Model saved as {model_id}")

            # ── 6. Notify frontend ───────────────────────────────────
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({"type": "status", "message": "Training Complete!"})),
                    main_loop
                )
            # Signal that the model list has changed
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({"type": "models_updated"})),
                    main_loop
                )

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            is_training = False  # Release the training lock

    background_tasks.add_task(_run_train)
    return {"message": "Training started."}


@app.post("/api/train/stop")
async def api_train_stop():
    """
    Requested by the frontend to gracefully stop an ongoing training run.
    """
    global stop_training, is_training
    if not is_training:
        return {"message": "No training is currently running."}
    
    stop_training = True
    return {"message": "Stop signal sent to trainer."}


# ── Dataset Inspection ───────────────────────────────────────────────

@app.get("/api/dataset")
async def api_dataset(file_path: str = "backend/data/dataset.json"):
    """
    Return the generated dataset JSON for the frontend Data Inspector page.

    Returns an empty list if no dataset has been generated yet.
    """
    if not os.path.exists(file_path):
        return {"games": []}
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return {"games": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Move Validation ──────────────────────────────────────────────────

@app.post("/api/validate_move")
async def api_validate_move(req: MoveValidationRequest):
    """
    Validate a human player's attempted move against the game engine rules.

    Reconstructs the board state from the request, generates all legal moves,
    and checks if the requested (start, end) matches any of them. This enforces
    the mandatory jump rule — even if a simple move looks valid, it will be
    rejected if a jump is available for any piece.

    Returns:
        is_valid: True if the move is legal
        jumped_pieces: List of [row, col] positions of captured pieces (if any)
    """
    board = CheckersBoard()
    board.grid = req.board_state
    board.current_turn = req.current_turn

    # Generate all legal moves (mandatory jumps are enforced by the engine)
    legal_moves = board.get_valid_moves(req.current_turn)

    for (start, end, jumped) in legal_moves:
        if start == (req.start_r, req.start_c) and end == (req.end_r, req.end_c):
            # Exact match with a legal move — return the captured pieces
            return {
                "is_valid": True,
                "jumped_pieces": [list(rc) for rc in jumped]
            }

    # No matching legal move found
    return {
        "is_valid": False,
        "jumped_pieces": []
    }


# ── AI Inference ─────────────────────────────────────────────────────

@app.post("/api/infer")
async def api_infer(req: InferRequest):
    """
    Select the AI's best move and optionally return CNN win probabilities.

    Pipeline:
        1. Check if the game is already over
        2. If a model_id is provided, load the CNN and compute P(black), P(white)
        3. Run minimax with alpha-beta pruning to find the best move
        4. Apply the move and check if it results in game over
        5. Return the move, CNN probabilities, and game-over status

    The CNN probabilities are for visualization only — move selection
    always uses the minimax search algorithm.
    """
    board = CheckersBoard()
    board.grid = req.board_state
    board.current_turn = req.current_turn

    # Step 1: Check if the game is already over before computing anything
    game_over = board.check_game_over()
    if game_over is not None:
        return {
            "move": None,
            "cnn_probabilities": None,
            "game_over": game_over
        }

    # Step 2: Load model if provided for heuristic evaluation during minimax
    model = None
    if req.model_id:
        model = _load_model_by_id(req.model_id)

    # Step 3: Select the best move via minimax (softmax sampling)
    best_move = get_best_move(board, depth=req.depth, temperature=req.temperature, model=model)

    # Step 4: Apply the move and check for game over
    move_payload = None
    if best_move:
        start_rc, end_rc, captured = best_move
        move_payload = {
            "start": list(start_rc),
            "end": list(end_rc),
            "jumped_pieces": [list(rc) for rc in captured]
        }
        # Apply the move to detect if it ends the game
        board.make_move(best_move)
        game_over = board.check_game_over()
    else:
        # No valid move → current player loses (opponent wins)
        game_over = WHITE if req.current_turn == BLACK else BLACK

    return {
        "move": move_payload,
        "game_over": game_over
    }


@app.post("/api/evaluate")
async def api_evaluate(req: EvaluateRequest):
    """
    Evaluates the board state against the provided models and returns 
    their respective win probabilities. Used by the frontend AI Brain Visualizer
    to show both sides' thoughts simultaneously.
    """
    result = {"red_eval": None, "white_eval": None}
    
    # Evaluate RED model
    if req.red_model_id:
        red_model = _load_model_by_id(req.red_model_id)
        if red_model is not None:
            tensor = board_to_tensor(req.board_state, req.current_turn)
            tensor = tensor.unsqueeze(0)
            red_model.eval()
            with torch.no_grad():
                p_black, p_white = red_model(tensor)
                result["red_eval"] = {
                    "p_black": round(p_black.item(), 4),
                    "p_white": round(p_white.item(), 4)
                }
                
    # Evaluate WHITE model
    if req.white_model_id:
        white_model = _load_model_by_id(req.white_model_id)
        if white_model is not None:
            tensor = board_to_tensor(req.board_state, req.current_turn)
            tensor = tensor.unsqueeze(0)
            white_model.eval()
            with torch.no_grad():
                p_black, p_white = white_model(tensor)
                result["white_eval"] = {
                    "p_black": round(p_black.item(), 4),
                    "p_white": round(p_white.item(), 4)
                }

    return result

# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming real-time training metrics to the frontend.

    The connection stays open and receives metric payloads whenever the
    WebSocketMetricsCallback fires during training. The client (Metrics chart)
    listens for these events to update the loss curves live.
    """
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            # Keep the connection alive by waiting for any client messages
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


# ── AI Tournament ────────────────────────────────────────────────────

class TournamentRequest(BaseModel):
    """Parameters for running an AI tournament."""
    num_games: int = 20                   # Total number of games to play
    depth: int = 2                        # Minimax search depth per move
    temperature: float = 1.0              # Softmax temperature for move sampling
    max_moves: int = 200                  # Max moves per game (draw if exceeded)


@app.post("/api/tournament")
async def api_tournament(req: TournamentRequest, background_tasks: BackgroundTasks):
    """
    Run a round-robin AI tournament in the background.
    Randomly pairs models for each game, collects win/loss/draw stats,
    and streams progress + final results via WebSocket.
    """
    models = _get_model_list()
    if len(models) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models for a tournament.")

    model_ids = [m["id"] for m in models]
    model_names = {m["id"]: m.get("name", m["id"]) for m in models}

    # Capture the event loop for cross-thread WebSocket broadcasting
    main_loop = asyncio.get_running_loop()

    def _run_tournament():
        import random as rng
        from backend.engine.board import CheckersBoard
        from backend.engine.minimax import get_best_move

        global stop_tournament
        stop_tournament = False

        # Initialize per-pair stats: {(red_id, white_id): {wins, losses, draws}}
        pair_stats: dict = {}
        model_totals: dict = {mid: {"wins": 0, "losses": 0, "draws": 0} for mid in model_ids}

        for game_idx in range(req.num_games):
            # Check stop signal
            if stop_tournament:
                break

            # Pick two distinct random models
            red_id, white_id = rng.sample(model_ids, 2)
            red_model = _load_model_by_id(red_id)
            white_model = _load_model_by_id(white_id)

            pair_key = f"{red_id}_vs_{white_id}"
            if pair_key not in pair_stats:
                pair_stats[pair_key] = {"red_id": red_id, "white_id": white_id, "red_wins": 0, "white_wins": 0, "draws": 0}

            # Play a game
            board = CheckersBoard()
            move_count = 0
            winner = None

            while True:
                winner = board.check_game_over()
                if winner is not None:
                    break
                if move_count > req.max_moves:
                    winner = 0
                    break

                # Pick the right model for the current player
                current_model = red_model if board.current_turn == 1 else white_model
                move = get_best_move(board, req.depth, req.temperature, model=current_model)

                if move:
                    board.make_move(move)
                    move_count += 1
                else:
                    winner = 2 if board.current_turn == 1 else 1
                    break

            # Record result
            if winner == 1:  # Red won
                pair_stats[pair_key]["red_wins"] += 1
                model_totals[red_id]["wins"] += 1
                model_totals[white_id]["losses"] += 1
            elif winner == 2:  # White won
                pair_stats[pair_key]["white_wins"] += 1
                model_totals[white_id]["wins"] += 1
                model_totals[red_id]["losses"] += 1
            else:  # Draw
                pair_stats[pair_key]["draws"] += 1
                model_totals[red_id]["draws"] += 1
                model_totals[white_id]["draws"] += 1

            # Stream progress with cumulative standings
            result_label = "Red" if winner == 1 else "White" if winner == 2 else "Draw"

            # Build live standings snapshot
            standings = []
            for mid in model_ids:
                t = model_totals[mid]
                total_played = t["wins"] + t["losses"] + t["draws"]
                wr = t["wins"] / total_played if total_played > 0 else 0.0
                standings.append({
                    "model_id": mid,
                    "name": model_names[mid],
                    "wins": t["wins"],
                    "losses": t["losses"],
                    "draws": t["draws"],
                    "total": total_played,
                    "win_rate": round(wr, 3)
                })
            standings.sort(key=lambda s: s["win_rate"], reverse=True)

            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({
                        "type": "tournament_progress",
                        "game": game_idx + 1,
                        "total": req.num_games,
                        "red": model_names[red_id],
                        "white": model_names[white_id],
                        "result": result_label,
                        "moves": move_count,
                        "standings": standings
                    })),
                    main_loop
                )

        # ── Fit Bradley-Terry ranking model ──────────────────────────
        # MLE iterative algorithm: r_i = W_i / Σ_j (n_ij / (r_i + r_j))
        # where W_i = total wins for model i, n_ij = games between i and j
        # Draws count as 0.5 win for each side.

        # Build win matrix from pair_stats
        idx_map = {mid: i for i, mid in enumerate(model_ids)}
        n = len(model_ids)
        win_matrix = [[0.0] * n for _ in range(n)]

        for key, ps in pair_stats.items():
            ri = idx_map[ps["red_id"]]
            wi = idx_map[ps["white_id"]]
            # Red wins count as wins for red model
            win_matrix[ri][wi] += ps["red_wins"] + 0.5 * ps["draws"]
            win_matrix[wi][ri] += ps["white_wins"] + 0.5 * ps["draws"]

        # Iterative BT fitting (50 iterations is more than enough)
        ratings = [1.0] * n
        for _ in range(50):
            new_ratings = [0.0] * n
            for i in range(n):
                w_i = sum(win_matrix[i])
                if w_i == 0:
                    new_ratings[i] = ratings[i]
                    continue
                denom = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    n_ij = win_matrix[i][j] + win_matrix[j][i]
                    if n_ij > 0:
                        denom += n_ij / (ratings[i] + ratings[j])
                new_ratings[i] = w_i / denom if denom > 0 else ratings[i]
            # Normalize so geometric mean = 1
            import math
            geo_mean = math.exp(sum(math.log(max(r, 1e-10)) for r in new_ratings) / n)
            ratings = [r / geo_mean for r in new_ratings]

        # Convert to Elo scale: Elo = 1500 + 400 * log10(rating)
        import math
        elo_ratings = {}
        for i, mid in enumerate(model_ids):
            elo_ratings[mid] = round(1500 + 400 * math.log10(max(ratings[i], 1e-10)))

        # Build rankings sorted by BT rating
        rankings = []
        for mid in model_ids:
            t = model_totals[mid]
            total_games = t["wins"] + t["losses"] + t["draws"]
            win_rate = t["wins"] / total_games if total_games > 0 else 0.0
            rankings.append({
                "model_id": mid,
                "name": model_names[mid],
                "wins": t["wins"],
                "losses": t["losses"],
                "draws": t["draws"],
                "total": total_games,
                "win_rate": round(win_rate, 3),
                "rating": elo_ratings[mid]
            })
        rankings.sort(key=lambda r: r["rating"], reverse=True)

        # Send final results
        for ws in active_websockets:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps({
                    "type": "tournament_complete",
                    "rankings": rankings,
                    "head_to_head": pair_stats
                })),
                main_loop
            )

    background_tasks.add_task(_run_tournament)
    return {"message": f"Tournament started: {req.num_games} games among {len(model_ids)} models."}


@app.post("/api/tournament/stop")
async def api_tournament_stop():
    """Signal the running tournament to stop after the current game."""
    global stop_tournament
    stop_tournament = True
    return {"message": "Tournament stop signal sent."}


# ── Model Registry ───────────────────────────────────────────────────

@app.get("/api/models")
async def api_list_models():
    """Return metadata for all saved models in the registry."""
    return {"models": _get_model_list()}


@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str):
    """
    Delete a model from disk and remove it from the in-memory cache.

    Removes both the .ckpt checkpoint file and the .meta.json metadata file.
    """
    ckpt_path = os.path.join(MODELS_DIR, f"{model_id}.ckpt")
    meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta.json")

    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    os.remove(meta_path)

    # Evict from cache if present
    model_cache.pop(model_id, None)

    return {"message": f"Model '{model_id}' deleted."}
