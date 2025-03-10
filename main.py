from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Game Models
class CreateGameRequest(BaseModel):
    time_limit_minutes: int

class GameInfo(BaseModel):
    id: str
    time_limit_minutes: int
    creator_id: Optional[str] = None
    opponent_id: Optional[str] = None
    status: str  # waiting, active, finished
    created_at: str
    white_player: Optional[str] = None
    black_player: Optional[str] = None
    current_turn: str = "white"
    board_state: List[List[str]]
    white_time_left_seconds: int
    black_time_left_seconds: int
    winner: Optional[str] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.player_to_game: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, game_id: str, player_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = {}
        self.active_connections[game_id][player_id] = websocket
        self.player_to_game[player_id] = game_id

    def disconnect(self, game_id: str, player_id: str):
        if game_id in self.active_connections and player_id in self.active_connections[game_id]:
            del self.active_connections[game_id][player_id]
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
        if player_id in self.player_to_game:
            del self.player_to_game[player_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, game_id: str, exclude: Optional[str] = None):
        if game_id in self.active_connections:
            for player_id, connection in self.active_connections[game_id].items():
                if exclude is None or player_id != exclude:
                    await connection.send_text(message)

    async def broadcast_game_update(self, game_id: str):
        if game_id in games:
            game_state = games[game_id]
            await self.broadcast(
                json.dumps({"type": "game_update", "data": game_state}),
                game_id
            )

# Game state storage
games: Dict[str, dict] = {}
manager = ConnectionManager()

# Initialize a chess board with starting positions
def initialize_board():
    # Empty board
    board = [['' for _ in range(8)] for _ in range(8)]

    # Set up pawns
    for col in range(8):
        board[1][col] = 'bp'  # black pawns
        board[6][col] = 'wp'  # white pawns

    # Set up other pieces
    pieces = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
    for col, piece in enumerate(pieces):
        board[0][col] = 'b' + piece  # black pieces
        board[7][col] = 'w' + piece  # white pieces

    return board

def is_path_clear(game_state, start_pos, end_pos):
    start_row, start_col = start_pos
    end_row, end_col = end_pos

    # Knights can jump over pieces
    piece = game_state["board_state"][start_row][start_col]
    if piece.endswith('n'):
        return True

    # For other pieces, check if the path is clear
    row_step = 0 if start_row == end_row else (1 if end_row > start_row else -1)
    col_step = 0 if start_col == end_col else (1 if end_col > start_col else -1)

    current_row, current_col = start_row + row_step, start_col + col_step

    while (current_row, current_col) != (end_row, end_col):
        if game_state["board_state"][current_row][current_col]:
            return False  # Path is blocked
        current_row += row_step
        current_col += col_step

    return True

def can_castle(game_state, player_color, side):
    rights = game_state["castling_rights"][player_color]
    row = 7 if player_color == 'white' else 0
    if rights["king_moved"]:
        return False

    if side == "king":
        if rights["rook_h_moved"]:
            return False
        if game_state["board_state"][row][5] or game_state["board_state"][row][6]:
            return False
        if is_square_under_attack(game_state["board_state"], (row, 4), player_color) or \
           is_square_under_attack(game_state["board_state"], (row, 5), player_color) or \
           is_square_under_attack(game_state["board_state"], (row, 6), player_color):
            return False
    else:  # queen side
        if rights["rook_a_moved"]:
            return False
        if game_state["board_state"][row][1] or game_state["board_state"][row][2] or game_state["board_state"][row][3]:
            return False
        if is_square_under_attack(game_state["board_state"], (row, 4), player_color) or \
           is_square_under_attack(game_state["board_state"], (row, 3), player_color) or \
           is_square_under_attack(game_state["board_state"], (row, 2), player_color):
            return False
    return True

def is_valid_move(game_state, start_pos, end_pos, player_color):
    # Previous validations remain the same
    start_row, start_col = start_pos
    end_row, end_col = end_pos

    # Check if positions are within board bounds
    if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
        return False

    # Check if there's a piece at the start position
    if not game_state["board_state"][start_row][start_col]:
        return False

    # Check if the piece belongs to the current player
    piece = game_state["board_state"][start_row][start_col]
    if not piece.startswith(player_color[0]):
        return False

    # Check if target is not player's own piece
    target = game_state["board_state"][end_row][end_col]
    if target and target.startswith(player_color[0]):
        return False

    # Validate piece-specific movements
    piece_type = piece[1]
    valid_movement = False

    if piece_type == 'p':  # Pawn
        direction = -1 if player_color == 'white' else 1
        start_rank = 6 if player_color == 'white' else 1
		# Normal moves
        if start_col == end_col:
            if end_row == start_row + direction and not target:
                valid_movement = True
            elif start_row == start_rank and end_row == start_row + 2 * direction and \
				 not game_state["board_state"][start_row + direction][start_col] and not target:
                valid_movement = True
		# Capture moves (including en passant)
        elif abs(start_col - end_col) == 1 and end_row == start_row + direction:
            if target:
                valid_movement = True
            elif game_state["en_passant_target"] == (end_row, end_col):
                valid_movement = True

    elif piece_type == 'r':  # Rook
        if start_row == end_row or start_col == end_col:
            valid_movement = True

    elif piece_type == 'n':  # Knight
        if (abs(start_row - end_row) == 2 and abs(start_col - end_col) == 1) or \
           (abs(start_row - end_row) == 1 and abs(start_col - end_col) == 2):
            valid_movement = True

    elif piece_type == 'b':  # Bishop
        if abs(start_row - end_row) == abs(start_col - end_col):
            valid_movement = True

    elif piece_type == 'q':  # Queen
        if start_row == end_row or start_col == end_col or \
           abs(start_row - end_row) == abs(start_col - end_col):
            valid_movement = True

    elif piece_type == 'k':  # King
        if abs(start_row - end_row) <= 1 and abs(start_col - end_col) <= 1:
            valid_movement = True

    # Check if path is clear (except for knights)
    if valid_movement and piece_type != 'n':
        valid_movement = is_path_clear(game_state, start_pos, end_pos)

    # Check if move would leave the king in check
    if valid_movement:
        # Create a temporary board to test the move
        temp_board = [row[:] for row in game_state["board_state"]]
        temp_board[end_row][end_col] = temp_board[start_row][start_col]
        temp_board[start_row][start_col] = ""

        # Check if the king would be in check after this move
        king_pos = find_king(temp_board, player_color)
        if king_pos and is_square_under_attack(temp_board, king_pos, player_color):
            valid_movement = False
	
    elif piece_type == 'k':  # King
        #Normal move
        if abs(start_row - end_row) <= 1 and abs(start_col - end_col) <= 1:
            valid_movement = True
        # Castling
        elif start_row == end_row and abs(start_col - end_col) == 2:
            side = 'king' if end_col > start_col else 'queen'
            valid_movement = can_castle(game_state, player_color, side)
	
    return valid_movement

# Helper functions to find the king and determine if a square is under attack
def find_king(board, player_color):
    king_piece = player_color[0] + 'k'
    for row in range(8):
        for col in range(8):
            if board[row][col] == king_piece:
                return (row, col)
    return None

def is_square_under_attack(board, square, player_color):
    row, col = square
    opponent_color = 'black' if player_color == 'white' else 'white'

    # Check for attacks from all directions

    # Check for pawns
    pawn_directions = [(-1, -1), (-1, 1)] if player_color == 'white' else [(1, -1), (1, 1)]
    for dr, dc in pawn_directions:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent_color[0] + 'p':
            return True

    # Check for knights
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    for dr, dc in knight_moves:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent_color[0] + 'n':
            return True

    # Check for rooks/queens (horizontal and vertical)
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r][c]:
                if board[r][c] in [opponent_color[0] + 'r', opponent_color[0] + 'q']:
                    return True
                break
            r, c = r + dr, c + dc

    # Check for bishops/queens (diagonal)
    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r][c]:
                if board[r][c] in [opponent_color[0] + 'b', opponent_color[0] + 'q']:
                    return True
                break
            r, c = r + dr, c + dc

    # Check for king (adjacent squares)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent_color[0] + 'k':
                return True

    return False

# Function to check if a player is in checkmate
def is_checkmate(game_state, player_color):
    # Find the king's position
    king_pos = find_king(game_state["board_state"], player_color)
    if not king_pos:
        return False  # Safety check: king should always exist

    # If king is not in check, can't be checkmate
    if not is_square_under_attack(game_state["board_state"], king_pos, player_color):
        return False

    # Try all possible moves for player's pieces to escape check
    for start_row in range(8):
        for start_col in range(8):
            piece = game_state["board_state"][start_row][start_col]
            if piece and piece.startswith(player_color[0]):
                for end_row in range(8):
                    for end_col in range(8):
                        if is_valid_move(game_state, (start_row, start_col), (end_row, end_col), player_color):
                            # Temporarily make the move
                            original_piece = game_state["board_state"][end_row][end_col]
                            game_state["board_state"][end_row][end_col] = piece
                            game_state["board_state"][start_row][start_col] = ""

                            new_king_pos = (end_row, end_col) if piece[1] == "K" else king_pos

                            # Check if move gets king out of check
                            if not is_square_under_attack(game_state["board_state"], new_king_pos, player_color):
                                # Undo move
                                game_state["board_state"][start_row][start_col] = piece
                                game_state["board_state"][end_row][end_col] = ""
                                return False

                            # Undo move
                            game_state["board_state"][start_row][start_col] = piece
                            game_state["board_state"][end_row][end_col] = ""

    # If no valid move is found to escape check, it's checkmate
    return True


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/games")
async def create_game(game_request: CreateGameRequest):
    game_id = str(uuid.uuid4())
    time_limit_seconds = game_request.time_limit_minutes * 60

    games[game_id] = {
        "id": game_id,
        "time_limit_minutes": game_request.time_limit_minutes,
        "status": "waiting",
        "created_at": datetime.now().isoformat(),
        "board_state": initialize_board(),
        "white_time_left_seconds": time_limit_seconds,
        "black_time_left_seconds": time_limit_seconds,
        "current_turn": "white",
        "moves": [],
        "chat_messages": [],
        "castling_rights": {
        "white": {"king_moved": False, "rook_h_moved": False, "rook_a_moved": False},
        "black": {"king_moved": False, "rook_h_moved": False, "rook_a_moved": False}
			},
		"en_passant_target": None  # Coordinates of the pawn that just moved two squares
    }

    return {"game_id": game_id}

@app.get("/game/{game_id}", response_class=HTMLResponse)
async def get_game(request: Request, game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    return templates.TemplateResponse("game.html", {
        "request": request,
        "game_id": game_id,
        "game": games[game_id]
    })

@app.websocket("/ws/game/{game_id}/{player_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, player_id: str):
    if game_id not in games:
        await websocket.close(code=1000, reason="Game not found")
        return

    await manager.connect(websocket, game_id, player_id)

    game_state = games[game_id]

    # Assign player to white or black if not already assigned
    if game_state["status"] == "waiting":
        if "white_player" not in game_state or game_state["white_player"] is None:
            game_state["white_player"] = player_id
            await manager.send_personal_message(
                json.dumps({"type": "player_assignment", "color": "white"}),
                websocket
            )
        elif "black_player" not in game_state or game_state["black_player"] is None:
            game_state["black_player"] = player_id
            game_state["status"] = "active"
            await manager.send_personal_message(
                json.dumps({"type": "player_assignment", "color": "black"}),
                websocket
            )
            # Start the game
            await manager.broadcast(
                json.dumps({"type": "game_started"}),
                game_id
            )
    else:
        # For reconnecting players
        if game_state["white_player"] == player_id:
            await manager.send_personal_message(
                json.dumps({"type": "player_assignment", "color": "white"}),
                websocket
            )
        elif game_state["black_player"] == player_id:
            await manager.send_personal_message(
                json.dumps({"type": "player_assignment", "color": "black"}),
                websocket
            )
        else:
            await manager.send_personal_message(
                json.dumps({"type": "player_assignment", "color": "spectator"}),
                websocket
            )

    # Send current game state
    await manager.send_personal_message(
        json.dumps({"type": "game_update", "data": game_state}),
        websocket
    )

    # Get existing chat history
    if "chat_messages" in game_state:
        for msg in game_state["chat_messages"]:
            await manager.send_personal_message(
                json.dumps({"type": "chat_message", "data": msg}),
                websocket
            )

    # Start game clock if applicable
    if game_state["status"] == "active" and len(manager.active_connections.get(game_id, {})) >= 2:
        asyncio.create_task(run_game_clock(game_id))

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "chat":
                chat_message = {
                    "sender": player_id,
                    "sender_color": "white" if game_state["white_player"] == player_id else "black",
                    "content": message["content"],
                    "timestamp": datetime.now().isoformat()
                }
                game_state["chat_messages"].append(chat_message)
                await manager.broadcast(
                    json.dumps({"type": "chat_message", "data": chat_message}),
                    game_id
                )

            elif message["type"] == "move":
                # Validate and process the move
                if game_state["status"] != "active":
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Game is not active"}),
                        websocket
                    )
                    continue

                player_color = "white" if game_state["white_player"] == player_id else "black"
                if player_color != game_state["current_turn"]:
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Not your turn"}),
                        websocket
                    )
                    continue

                start_pos = (message["start_row"], message["start_col"])
                end_pos = (message["end_row"], message["end_col"])

                if is_valid_move(game_state, start_pos, end_pos, player_color):
                    # Record the move
                    move = {
                        "player": player_id,
                        "color": player_color,
                        "from": start_pos,
                        "to": end_pos,
                        "piece": game_state["board_state"][start_pos[0]][start_pos[1]],
                        "captured": game_state["board_state"][end_pos[0]][end_pos[1]],
                        "timestamp": datetime.now().isoformat()
                    }
                    game_state["moves"].append(move)

                    start_row, start_col = start_pos
                    end_row, end_col = end_pos
                    piece = game_state["board_state"][start_row][start_col]

                    # Update the board
                    game_state["board_state"][end_pos[0]][end_pos[1]] = game_state["board_state"][start_pos[0]][start_pos[1]]
                    game_state["board_state"][start_pos[0]][start_pos[1]] = ""

                    # En Passant handling
                    game_state["en_passant_target"] = None

                    if piece.endswith('p') and abs(start_row - end_row) == 2:
                        ep_row = (start_row + end_row) // 2
                        game_state["en_passant_target"] = (ep_row, start_col)

                    if piece.endswith('p') and abs(start_col - end_col) == 1 and not captured_piece:
                        captured_row = start_row
                        game_state["board_state"][captured_row][end_col] = ""
                        move["captured"] = f"{'b' if player_color == 'white' else 'w'}p"
					# END OF En Passant handling

					# Castling rights & moves
                    if piece.endswith('k'):
                        game_state["castling_rights"][player_color]["king_moved"] = True
                    if piece.endswith('r'):
                        if start_col == 0:
                        	game_state["castling_rights"][player_color]["rook_a_moved"] = True
                        elif start_col == 7:
                        	game_state["castling_rights"][player_color]["rook_h_moved"] = True

                    if piece.endswith('k') and abs(start_col - end_col) == 2:
                        row = 7 if player_color == 'white' else 0
                        if end_col == 6:  # King-side castling
                        	game_state["board_state"][row][5] = game_state["board_state"][row][7]
                        	game_state["board_state"][row][7] = ""
                        elif end_col == 2:  # Queen-side castling
                        	game_state["board_state"][row][3] = game_state["board_state"][row][0]
                        	game_state["board_state"][row][0] = ""
					# END OF Castling handling

                    # Check for king capture (simplified win condition)
                    captured_piece = move["captured"]
                    if captured_piece and captured_piece.endswith('k'):
                        game_state["status"] = "finished"
                        game_state["winner"] = player_color

                    # Switch turns
                    game_state["current_turn"] = "black" if player_color == "white" else "white"

                    # Broadcast the updated game state
                    await manager.broadcast_game_update(game_id)
                else:
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Invalid move"}),
                        websocket
                    )

            elif message["type"] == "resign":
                game_state["status"] = "finished"
                game_state["winner"] = "black" if player_color == "white" else "white"
                await manager.broadcast_game_update(game_id)

    except WebSocketDisconnect:
        manager.disconnect(game_id, player_id)
        await manager.broadcast(
            json.dumps({
                "type": "player_disconnected",
                "player_id": player_id,
                "is_white": game_state["white_player"] == player_id
            }),
            game_id
        )

async def run_game_clock(game_id):
    """Run the game clock for both players"""
    if game_id not in games:
        return

    game_state = games[game_id]
    if game_state["status"] != "active":
        return

    last_update = datetime.now()

    while game_state["status"] == "active":
        await asyncio.sleep(1)  # Update every second

        current_time = datetime.now()
        elapsed_seconds = (current_time - last_update).total_seconds()
        last_update = current_time

        # Decrement the time for the current player
        if game_state["current_turn"] == "white":
            game_state["white_time_left_seconds"] -= elapsed_seconds
            if game_state["white_time_left_seconds"] <= 0:
                game_state["white_time_left_seconds"] = 0
                game_state["status"] = "finished"
                game_state["winner"] = "black"
        else:
            game_state["black_time_left_seconds"] -= elapsed_seconds
            if game_state["black_time_left_seconds"] <= 0:
                game_state["black_time_left_seconds"] = 0
                game_state["status"] = "finished"
                game_state["winner"] = "white"

        # Broadcast the updated game state
        await manager.broadcast_game_update(game_id)

        if game_state["status"] == "finished":
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
