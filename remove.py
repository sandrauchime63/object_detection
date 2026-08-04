from pathlib import Path


for file in Path(".").glob("*.txt"):
    file.unlink()

print("All .txt files deleted.")

