# scripts/app_bridge.py
import sys
import json

for line in sys.stdin:
    instance = json.loads(line)
    print(f"[PYTHON] Received: {instance}")
