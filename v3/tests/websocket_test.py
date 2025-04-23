#!/usr/bin/env python3
"""
ACDEK WebSocket Client Test Tool.

Test the WebSocket endpoints of the ACDEK API server.
"""
import asyncio
import json
import websockets
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def test_websocket(uri="ws://localhost:8000/ws"):
    """Test WebSocket connection to ACDEK API server."""
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Request status
            print("Requesting status...")
            await websocket.send(json.dumps({"type": "get_status"}))
            response = await websocket.recv()
            print(f"Status: {json.loads(response)}")
            
            # Request metrics
            print("Requesting metrics...")
            await websocket.send(json.dumps({"type": "get_metrics"}))
            response = await websocket.recv()
            print(f"Metrics: {json.loads(response)}")
            
            # Start monitoring command
            print("Sending start_monitor command...")
            await websocket.send(json.dumps({"type": "command", "command": "start_monitor"}))
            
            # Wait for updates
            print("Waiting for updates (press Ctrl+C to stop)...")
            count = 0
            while count < 5:  # Receive 5 messages then exit
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Received: {data['type']}")
                count += 1
                
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except KeyboardInterrupt:
        print("Test terminated by user")

def main():
    parser = argparse.ArgumentParser(description="Test ACDEK WebSocket API")
    parser.add_argument("--uri", default="ws://localhost:8000/ws",
                       help="WebSocket URI (default: ws://localhost:8000/ws)")
    args = parser.parse_args()
    
    asyncio.run(test_websocket(args.uri))

if __name__ == "__main__":
    main()