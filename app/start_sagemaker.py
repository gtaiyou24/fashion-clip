import argparse

import uvicorn

from app import app

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str)
parser.add_argument("--local", action="store_true")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

if args.command == "serve":
    uvicorn.run(app, host=args.host, port=args.port)
else:
    raise ValueError(f"invalid command: {args.command}")
