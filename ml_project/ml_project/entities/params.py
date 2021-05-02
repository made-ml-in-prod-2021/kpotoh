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
    features_to_drop: Optional[List[str]]
    use_scaling_for_num_features: bool
    target_col: str


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestRegressor")
    random_state: int = field(default=632)
