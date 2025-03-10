# WebSockets Multiplayer Chess Game

## • Description
This project is a simple multiplayer chess game built using FastAPI and WebSockets. It allows two players to play chess against each other in real-time, providing essential chess functionalities such as valid moves enforcement, checkmate detection, castling, and en passant.

## • Tech Stack

  Backend: FastAPI, WebSockets (Starlette)

  Frontend: HTML, CSS, JavaScript

## • Features

  Real-time two-player gameplay via WebSockets

  Real-time chat via Websockets

  Unique session-based game URLs

  Valid chess move detection (including special moves castling and en passant)

  Game clock functionality

  Automatic game state updates and broadcasting

## • Setup & Running

Clone the repository.

Install dependencies.

  fastapi
  uvicorn
  python-multipart
  jinja2
  websockets

Run the application:
  uvicorn main:app --host 0.0.0.0 --port 8000

Have fun.

## • Suggestions for Improvement

Record moves in database

Add user authentication and profiles

Add chess Engine to analyse positions: Stockfish [open-source], Lc0 (Leela Chess Zero), Komodo, etc

Feel free to fork, enhance, and contribute!
