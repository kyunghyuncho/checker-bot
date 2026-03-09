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

# Local Imports
from backend.data.generator import generate_dataset
from backend.model.lightning_module import CheckersDataset, CheckersLightningModule
from backend.model.cnn import board_to_tensor
from backend.engine.board import CheckersBoard, BLACK, WHITE
from backend.engine.minimax import get_best_move

app = FastAPI(title="Checkers Deep Learning API")

# Allow requests from the Vite frontend regardless of port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
active_websockets = []
is_training = False
model_cache: dict = {}  # Cache of loaded models: model_id -> LightningModule

# Model storage directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def _get_model_list():
    """Scan the models directory and return metadata for all saved models."""
    models = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith(".meta.json"):
            with open(os.path.join(MODELS_DIR, f), "r") as fh:
                meta = json.load(fh)
                models.append(meta)
    return models

def _load_model_by_id(model_id: str):
    """Load a saved model checkpoint by its ID, using cache."""
    if model_id in model_cache:
        return model_cache[model_id]
    ckpt_path = os.path.join(MODELS_DIR, f"{model_id}.ckpt")
    meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta.json")
    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        return None
    model = CheckersLightningModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model_cache[model_id] = model
    return model

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    num_games: int = 10
    depth: int = 4
    epsilon: float = 0.1
    output_file: str = "backend/data/dataset.json"

class TrainRequest(BaseModel):
    dataset_file: str = "backend/data/dataset.json"
    epochs: int = 20
    learning_rate: float = 0.001
    hidden_dims: int = 64
    num_conv_layers: int = 2
    dropout_rate: float = 0.2
    batch_size: int = 32
    val_split: float = 0.2
    patience: int = 5

class InferRequest(BaseModel):
    # The board grid (8x8) and current turn (1 for Black, 2 for White)
    board_state: list[list[int]]
    current_turn: int
    depth: int = 1
    model_id: str | None = None  # Specific model to use for this inference
    epsilon: float = 0.0  # Randomness for move selection (0 = deterministic)

class MoveValidationRequest(BaseModel):
    board_state: list[list[int]]
    current_turn: int
    start_r: int
    start_c: int
    end_r: int
    end_c: int

# --- WebSocket Callback for Lightning ---
class WebSocketMetricsCallback(Callback):
    """
    A PyTorch Lightning Callback that streams loss metrics
    to all active WebSockets at the end of every epoch.
    """
    def __init__(self, main_loop):
        self.main_loop = main_loop

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss_val = metrics.get("train_loss_epoch")
        if train_loss_val is None:
             train_loss_val = metrics.get("train_loss")

        # Skip the sanity-check validation that PL runs before training starts
        if train_loss_val is None:
            return

        val_loss_val = metrics.get("val_loss")

        out = {
            "epoch": trainer.current_epoch,
            "train_loss": round(train_loss_val.item(), 4),
            "val_loss": round(val_loss_val.item(), 4) if val_loss_val is not None else None,
            "type": "metric"
        }
        
        # Broadcast to all connected clients securely across thread boundary
        for ws in active_websockets:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps(out)), 
                self.main_loop
            )


# --- Endpoints ---

@app.post("/api/generate")
async def api_generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Triggers dataset generation in the background."""
    main_loop = asyncio.get_running_loop()
    
    def _run_gen():
        def progress(current, total):
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({
                        "type": "status", 
                        "message": f"Generating data: {current}/{total} games..."
                    })),
                    main_loop
                )
                
        try:
            generate_dataset(req.num_games, req.output_file, req.depth, req.epsilon, progress)
            print("Dataset generation complete.")
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

@app.post("/api/train")
async def api_train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Starts model training using PyTorch Lightning."""
    global is_training, current_model
    if is_training:
        raise HTTPException(status_code=400, detail="Training is already in progress.")

    main_loop = asyncio.get_running_loop()

    def _run_train():
        global is_training
        is_training = True
        try:
            # 1. Load Data
            dataset = CheckersDataset(req.dataset_file)
            from torch.utils.data import DataLoader, random_split
            
            # Split into train/val
            total = len(dataset)
            val_size = max(1, int(total * req.val_split))
            train_size = total - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=req.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=req.batch_size, shuffle=False)
            
            # 2. Initialize Model
            model = CheckersLightningModule(
                learning_rate=req.learning_rate, 
                hidden_dims=req.hidden_dims,
                num_conv_layers=req.num_conv_layers,
                dropout_rate=req.dropout_rate
            )
            
            # 3. Configure Trainer with WebSocket callback + Early Stopping
            callbacks = [WebSocketMetricsCallback(main_loop)]
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
                enable_checkpointing=False, # Simplify for educational app
                logger=False
            )
            
            # 4. Train
            print("Starting training loop...")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Save the model into cache for immediate inference
            # 5. Persist model to disk
            model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(MODELS_DIR, f"{model_id}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            
            # Cache the trained model
            model.eval()
            model_cache[model_id] = model
            
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
            
            # Notify frontend
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({"type": "status", "message": "Training Complete!"})),
                    main_loop
                )
            # Also notify with model list update
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
            is_training = False

    background_tasks.add_task(_run_train)
    return {"message": "Training started."}


@app.get("/api/dataset")
async def api_dataset(file_path: str = "backend/data/dataset.json"):
    """Returns the generated dataset JSON for the frontend inspector."""
    if not os.path.exists(file_path):
        return {"games": []}
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return {"games": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate_move")
async def api_validate_move(req: MoveValidationRequest):
    """
    Checks if a requested move is valid according to the Python engine rules.
    If valid, returns the list of pieces that were jumped (so the UI can remove them)
    and any piece promotions.
    """
    board = CheckersBoard()
    board.grid = req.board_state
    board.current_turn = req.current_turn
    
    # Get all legal moves (this enforces mandatory jumps)
    legal_moves = board.get_valid_moves(req.current_turn)
    
    for (start, end, jumped) in legal_moves:
        if start == (req.start_r, req.start_c) and end == (req.end_r, req.end_c):
            # The move exactly matches a legal move generated by the engine!
            return {
                "is_valid": True,
                "jumped_pieces": [list(rc) for rc in jumped]
            }
            
    # Move is not in the legal list (either illegal direction, or missed a mandatory jump)
    return {
        "is_valid": False,
        "jumped_pieces": []
    }

@app.post("/api/infer")
async def api_infer(req: InferRequest):
    """
    Returns the AI's best move given the current board state.
    Uses the model specified by model_id if provided.
    """
    board = CheckersBoard()
    board.grid = req.board_state
    board.current_turn = req.current_turn

    # 1. Check if game is already over
    game_over = board.check_game_over()
    if game_over is not None:
        return {
            "move": None,
            "cnn_probabilities": None,
            "game_over": game_over
        }

    # 2. Get CNN Probabilities if a model is specified
    cnn_probs = None
    model = None
    if req.model_id:
        model = _load_model_by_id(req.model_id)
    
    if model is not None:
        tensor = board_to_tensor(req.board_state, req.current_turn)
        tensor = tensor.unsqueeze(0) 
        
        model.eval()
        with torch.no_grad():
            p_black, p_white = model(tensor)
            cnn_probs = {
                "p_black": round(p_black.item(), 4),
                "p_white": round(p_white.item(), 4)
            }

    # 3. Get the best move from the engine
    best_move = get_best_move(board, depth=req.depth, epsilon=req.epsilon)

    move_payload = None
    if best_move:
        start_rc, end_rc, captured = best_move
        move_payload = {
            "start": list(start_rc),
            "end": list(end_rc),
            "jumped_pieces": [list(rc) for rc in captured]
        }
        
        # 4. Apply the move to see if the AI's move won the game
        board.make_move(best_move)
        game_over = board.check_game_over()
    else:
        # No valid move found — current player loses
        game_over = WHITE if req.current_turn == BLACK else BLACK

    return {
        "move": move_payload,
        "cnn_probabilities": cnn_probs,
        "game_over": game_over
    }


@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for streaming real-time loss charts."""
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


# --- Model Registry Endpoints ---

@app.get("/api/models")
async def api_list_models():
    """Returns the list of all saved models with metadata."""
    return {"models": _get_model_list()}

@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str):
    """Delete a saved model from disk."""
    ckpt_path = os.path.join(MODELS_DIR, f"{model_id}.ckpt")
    meta_path = os.path.join(MODELS_DIR, f"{model_id}.meta.json")
    
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    os.remove(meta_path)
    
    # Remove from cache if present
    model_cache.pop(model_id, None)
    
    return {"message": f"Model '{model_id}' deleted."}
