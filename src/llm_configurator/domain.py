"""Validated domain objects; all memory calculations use bytes, not rounded GB."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math

GIB = 1024**3

# Dense architectures use full attention unless sliding-window fields say otherwise.
DENSE_ARCHITECTURES = {"qwen2", "qwen3", "llama", "mistral", "gemma2", "gemma3", "gemma3_text", "phi3", "granite", "olmo2"}
# Mixture-of-experts: only `active_experts` of `experts` run per token, so fewer bytes are read per token.
MOE_ARCHITECTURES = {"qwen2_moe", "qwen3_moe", "mixtral", "gpt_oss"}
ARCHITECTURES = DENSE_ARCHITECTURES | MOE_ARCHITECTURES

# Approximate GGUF file bytes per parameter, including typical mixed-precision overhead.
QUANT_BYTES_PER_PARAMETER = {"Q2_K": 0.35, "Q3_K_M": 0.49, "IQ4_XS": 0.54, "Q4_0": 0.57, "Q4_K_M": 0.62,
                             "MXFP4": 0.54, "Q5_K_M": 0.73, "Q6_K": 0.84, "Q8_0": 1.1, "F16": 2.0, "BF16": 2.0}
# Bytes per stored K or V element; llama.cpp q8_0/q4_0 blocks hold 32 values in 34/18 bytes.
KV_BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Cancelled(Exception):
    """Raised by long operations when the user cancels; never an error to report."""


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled by user")


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
    priority: str = "balanced"
    include_rankings: bool = True
    kv_cache_type: str = "f16"

    def __post_init__(self):
        if self.kv_cache_type not in KV_BYTES_PER_ELEMENT:
            raise ValueError("kv_cache_type must be f16, q8_0 or q4_0")
        if self.priority not in {"balanced", "quality", "speed"}:
            raise ValueError("Priority must be balanced, quality or speed")
        if type(self.include_rankings) is not bool:
            raise ValueError("include_rankings must be a boolean")
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
    # v0.4 additions; defaults keep cached v0.3 records loadable.
    files: list = field(default_factory=list)  # sharded models: [{"filename", "size_bytes", "sha256"}], first shard first
    parameters: int | None = None
    active_parameters: int | None = None
    experts: int = 0
    active_experts: int = 0
    expert_fraction: float = 0.0  # share of weight bytes held in routed-expert tensors (0 for dense)
    sliding_window: int | None = None
    sliding_layers: int = 0  # layers whose KV cache is capped at sliding_window tokens
    family: str | None = None
    license: str | None = None
    source: str = "catalogue"  # catalogue | custom | local

    def __post_init__(self):
        for name in ["size_bytes", "layers", "kv_heads", "head_dim", "max_context"]:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid model metadata: {name}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported memory architecture: {self.architecture}")
        if self.source not in {"catalogue", "custom", "local"}:
            raise ValueError("Variant source must be catalogue, custom or local")
        if not isinstance(self.files, list) or any(not isinstance(f, dict) or not f.get("filename") or
                                                   type(f.get("size_bytes")) is not int or f["size_bytes"] <= 0 for f in self.files):
            raise ValueError("Invalid model metadata: files")
        if self.files and sum(f["size_bytes"] for f in self.files) != self.size_bytes:
            raise ValueError("Invalid model metadata: shard sizes must add up to size_bytes")
        if type(self.experts) is not int or type(self.active_experts) is not int or not 0 <= self.active_experts <= self.experts:
            raise ValueError("Invalid model metadata: experts")
        if isinstance(self.expert_fraction, bool) or not isinstance(self.expert_fraction, (int, float)) or not 0 <= self.expert_fraction < 1:
            raise ValueError("Invalid model metadata: expert_fraction")
        if type(self.sliding_layers) is not int or not 0 <= self.sliding_layers <= self.layers:
            raise ValueError("Invalid model metadata: sliding_layers")
        if self.sliding_layers and (type(self.sliding_window) is not int or self.sliding_window <= 0):
            raise ValueError("Invalid model metadata: sliding_window")

    @property
    def moe(self):
        return self.experts > 0

    def all_files(self):
        """Every file to download, verify or delete; single-file models return one entry."""
        return self.files or [{"filename": self.filename, "size_bytes": self.size_bytes, "sha256": self.sha256}]

    def to_dict(self):
        return asdict(self)
