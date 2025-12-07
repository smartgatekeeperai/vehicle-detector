import subprocess

subprocess.run(
    "uvicorn main:app --host 0.0.0.0 --port 7860",
    shell=True,
    check=True
)
