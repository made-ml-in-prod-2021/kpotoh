from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=6322)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]
    use_log_trick: bool = field(default=True)


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestRegressor")
    random_state: int = field(default=632)
