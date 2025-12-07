import subprocess

# HF Spaces will run this file (app_file: start.py in README metadata)
# It just starts your FastAPI app via uvicorn on port 7860.
subprocess.run(
    "uvicorn main:app --host 0.0.0.0 --port 7860",
    shell=True,
    check=True,
)
