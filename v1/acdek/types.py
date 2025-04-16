from typing import List, Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class QEnriched:
    intent: str
    dependencies: List[str]
    constraints: Dict[str, Union[str, int, float]]
    ambiguity_score: float

@dataclass
class ArItem:
    key: str
    value: Union[str, int, float]
    confidence: float
    age: float

@dataclass
class ArEnriched:
    slots: Dict[str, ArItem]

@dataclass
class CuRaw:
    traits: Dict[str, Union[str, float, int]]

@dataclass
class EfRaw:
    variables: Dict[str, Union[str, float, int]]

@dataclass
class ModulationVector:
    tone: float
    assertiveness: float
    confidence_floor: float
    fallback_policy: str