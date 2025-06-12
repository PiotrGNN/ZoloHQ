import csv
import mimetypes
import os
from pathlib import Path

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "cache", "logs"}
BINARY_EXTS = {".db", ".whl", ".zip", ".tar", ".gz", ".pyc", ".exe", ".dll", ".pyd"}
LARGE_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
COUNT_LINES_EXTS = {".py", ".md", ".txt", ".csv", ".yml", ".yaml"}


def describe_file(path):
    ext = Path(path).suffix
    size = os.path.getsize(path)
    lines = "-"
    mime, _ = mimetypes.guess_type(path)
    if os.path.isdir(path):
        role = "directory"
    elif ext in [".py", ".ipynb"]:
        role = "code"
    elif ext in [".md", ".txt", ".rst"]:
        role = "doc"
    elif ext in [".csv", ".json", ".yaml", ".yml"]:
        role = "data"
    elif ext in [".log"]:
        role = "log"
    elif ext in [
        ".sh",
        ".ps1",
        ".bat",
        ".whl",
        ".zip",
        ".tar",
        ".gz",
        ".db",
        ".pyc",
        ".exe",
        ".dll",
        ".pyd",
    ]:
        role = "binary"
    else:
        role = mime or "other"
    if ext == ".py":
        desc = "Python code file"
    elif ext == ".md":
        desc = "Markdown documentation"
    elif ext == ".csv":
        desc = "CSV data file"
    elif ext == ".log":
        desc = "Log file"
    elif ext == ".json":
        desc = "JSON data/config file"
    elif ext == ".yml" or ext == ".yaml":
        desc = "YAML config file"
    elif ext == ".ipynb":
        desc = "Jupyter notebook"
    elif ext == ".db":
        desc = "Database file"
    elif ext == ".sh":
        desc = "Shell script"
    elif ext == ".whl":
        desc = "Python wheel package"
    elif os.path.isdir(path):
        desc = "Directory"
    else:
        desc = mime or "Other file"
    return [str(path), Path(path).name, ext, role, size, lines, desc]


rows = [["path", "name", "ext", "role", "size", "lines", "desc"]]
for root, dirs, files in os.walk(".", topdown=True):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for name in files:
        path = os.path.join(root, name)
        rows.append(describe_file(path))

with open("INVENTORY.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
