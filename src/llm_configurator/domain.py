"""Validated domain objects; all memory calculations use bytes, not rounded GB."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math

GIB = 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Requirements:
    workload: str = "general"
    context: int = 8192
    users: int = 1
    min_tps: float = 15.0
    reserve_gib: float = 2.0
    gpu_reserve_gib: float = 0.5
    gpu_index: int = 0
    reclaim_pids: list[int] = field(default_factory=list)
    strict_speed: bool = False

    def __post_init__(self):
        if self.workload not in {"general", "coding", "agentic", "documents"}:
            raise ValueError("Supported workloads: general, coding, agentic, documents")
        for name, low, high in [("context", 256, 1048576), ("users", 1, 64), ("gpu_index", 0, 64)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        for name in ["min_tps", "reserve_gib", "gpu_reserve_gib"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100000:
                raise ValueError(f"{name} must be a finite non-negative number")
        if type(self.strict_speed) is not bool:
            raise ValueError("strict_speed must be a boolean")
        if not isinstance(self.reclaim_pids, list) or any(type(p) is not int or p <= 0 for p in self.reclaim_pids):
            raise ValueError("reclaim_pids must be a list of positive process IDs")


@dataclass
class Variant:
    id: str
    name: str
    base_repo: str
    repo: str
    revision: str
    base_revision: str
    filename: str
    sha256: str | None
    quant: str
    size_bytes: int
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int
    architecture: str
    scores: dict = field(default_factory=dict)
    score_source: str | None = None
    score_version: str | None = None
    score_settings: str | None = None
    fetched_at: str = field(default_factory=now)
    demo: bool = False

    def __post_init__(self):
        for name in ["size_bytes", "layers", "kv_heads", "head_dim", "max_context"]:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid model metadata: {name}")
        if self.architecture not in {"qwen2", "qwen3", "llama"}:
            raise ValueError(f"Unsupported memory architecture: {self.architecture}")

    def to_dict(self):
        return asdict(self)
