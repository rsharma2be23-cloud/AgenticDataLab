"""Serializable Phase 2 experiment and explanation result contracts."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ExperimentResult:
    experiment_id: str
    dataset: Dict[str, Any]
    target: str
    task_type: str
    features: List[str]
    preprocessing: Dict[str, Any]
    model: str
    hyperparameters: Dict[str, Any]
    cv_results: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    test_metrics: Dict[str, Any]
    training_time_seconds: float
    warnings: List[str] = field(default_factory=list)
    critic_result: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None

    @classmethod
    def create(cls, **kwargs):
        kwargs.setdefault("experiment_id", str(uuid.uuid4()))
        return cls(**kwargs)

    def to_dict(self):
        return asdict(self)


@dataclass
class FeatureImportance:
    method: str
    features: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class LocalExplanation:
    method: str
    prediction: Any
    baseline_prediction: Any
    contributions: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
