"""Atomic local JSON records in SQLite; no API keys or process lists persisted."""
import json
import os
from pathlib import Path
import sqlite3
from contextlib import contextmanager


def data_dir() -> Path:
    if os.environ.get("LLM_CONFIG_HOME"):
        return Path(os.environ["LLM_CONFIG_HOME"]).expanduser()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LLMConfigurator"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "llm-configurator"


class Store:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else data_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "cache.sqlite3"
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=20)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, key, default=None):
        with self.connect() as conn:
            row = conn.execute("SELECT payload FROM records WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        payload = json.dumps(value, allow_nan=False)
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO records VALUES (?, ?)", (key, payload))

    def update(self, key, change, default=None):
        """Atomic read-modify-write so concurrent jobs never lose each other's records."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload FROM records WHERE key=?", (key,)).fetchone()
            value = change(json.loads(row[0]) if row else default)
            conn.execute("INSERT OR REPLACE INTO records VALUES (?, ?)", (key, json.dumps(value, allow_nan=False)))
        return value

    def append(self, key, item, limit=1000):
        return self.update(key, lambda items: ((items or []) + [item])[-limit:], [])


def models_dir(store) -> Path:
    """Where downloads go: LLM_CONFIG_MODELS, then a CLI-saved setting, then the data directory."""
    configured = os.environ.get("LLM_CONFIG_MODELS") or (store.get("settings") or {}).get("models_dir")
    path = Path(configured).expanduser() if configured else store.directory / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_dir(store) -> Path:
    """Managed llama.cpp installs live here; never on PATH, never system-wide."""
    path = store.directory / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path
