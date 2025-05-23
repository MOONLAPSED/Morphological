import sys, json

for line in sys.stdin:
    instance = json.loads(line)
    print(f"[PYTHON] Received: {instance}")
    # You can expand with create/update logic here
