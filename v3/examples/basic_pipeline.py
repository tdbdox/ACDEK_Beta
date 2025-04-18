#!/usr/bin/env python3
"""
Basic ACDEK Pipeline Example.

Demonstrates SDK usage and CLI invocation.
"""
import json
import subprocess
import sys

# SDK Invocation
from sdk.acdek_client import Client

def sdk_demo():
    input_data = {"text": "The quick brown fox jumps over the lazy dog."}
    client = Client(config={"fallback_mode": "lenient"})
    result = client.full_pipeline(input_data, output_format="dict")
    print("=== SDK Result ===")
    print(json.dumps(result, indent=2))

def cli_demo():
    input_json = json.dumps({"text": "The quick brown fox"})
    cmd = [sys.executable, "cli/run_pipeline.py", "-f", "lenient", "-o", "pretty"]
    print("=== CLI Result ===")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate(input_json)
    if out:
        print(out)
    if err:
        print("Errors:", err)

if __name__ == '__main__':
    sdk_demo()
    print()
    cli_demo()