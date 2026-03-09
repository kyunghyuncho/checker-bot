from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

# Local Imports
from backend.data.generator import generate_dataset
from backend.model.lightning_module import CheckersDataset, CheckersLightningModule
from backend.model.cnn import board_to_tensor
from backend.engine.board import CheckersBoard
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
# In a production app, we would use a database and Redis.
# For this local educational app, global state is sufficient.
active_websockets = []
is_training = False
current_model = None # Will hold the trained LightningModule

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    num_games: int = 10
    depth: int = 4
    epsilon: float = 0.1
    output_file: str = "backend/data/dataset.json"

class TrainRequest(BaseModel):
    dataset_file: str = "backend/data/dataset.json"
    epochs: int = 5
    learning_rate: float = 0.001
    hidden_dims: int = 64

class InferRequest(BaseModel):
    # The board grid (8x8) and current turn (1 for Black, 2 for White)
    board_state: list[list[int]]
    current_turn: int
    depth: int = 1 # Search depth for inference

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

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # Extract scalar values from tensors
        train_loss_val = metrics.get("train_loss_epoch")  # PL tracks epoch level loss with _epoch suffix
        if train_loss_val is None:
             train_loss_val = metrics.get("train_loss")

        out = {
            "epoch": trainer.current_epoch,
            "train_loss": train_loss_val.item() if train_loss_val is not None else 0,
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
        global is_training, current_model
        is_training = True
        try:
            # 1. Load Data
            dataset = CheckersDataset(req.dataset_file)
            from torch.utils.data import DataLoader
            # Small batch size for laptops
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # 2. Initialize Model
            model = CheckersLightningModule(
                learning_rate=req.learning_rate, 
                hidden_dims=req.hidden_dims
            )
            
            # 3. Configure Trainer with WebSocket callback
            trainer = Trainer(
                max_epochs=req.epochs,
                callbacks=[WebSocketMetricsCallback(main_loop)],
                enable_checkpointing=False, # Simplify for educational app
                logger=False
            )
            
            # 4. Train
            print("Starting training loop...")
            trainer.fit(model, train_dataloaders=dataloader)
            
            # Save the model into global state for inference
            current_model = model
            print("Training complete!")
            
            # Notify frontend
            for ws in active_websockets:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps({"type": "status", "message": "Training Complete!"})),
                    main_loop
                )

        except Exception as e:
            print(f"Training failed: {e}")
        finally:
            is_training = False

    background_tasks.add_task(_run_train)
    return {"message": "Training started."}

import os

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
    Utilizes the trained CNN if available to evaluate the heuristic,
    otherwise falls back to the hardcoded material heuristic.
    """
    board = CheckersBoard()
    board.grid = req.board_state
    board.current_turn = req.current_turn

    # 1. Check if game is already over (meaning the human's last move won the game)
    game_over = board.check_game_over()
    if game_over is not None:
        return {
            "move": None,
            "cnn_probabilities": None,
            "game_over": game_over
        }

    # 2. Get CNN Probabilities if model is loaded
    cnn_probs = None
    if current_model is not None:
        tensor = board_to_tensor(req.board_state, req.current_turn)
        tensor = tensor.unsqueeze(0) 
        
        current_model.eval()
        with torch.no_grad():
            p_black, p_white = current_model(tensor)
            cnn_probs = {
                "p_black": round(p_black.item(), 4),
                "p_white": round(p_white.item(), 4)
            }

    # 3. Get the best move from the engine
    best_move = get_best_move(board, depth=req.depth, epsilon=0)

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
            # Keep connection alive
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
