"""
Data models for ontology entity and mapping representation
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OntologyEntity:
    """
    Represents an ontology entity with its contextual information
    """
    label: str
    parent_class: str
    synonyms: List[str] = field(default_factory=list)
    ontology_id: Optional[str] = None

    def __str__(self):
        return f"{self.label} (parent: {self.parent_class})"


@dataclass
class EntityMapping:
    """
    Represents a mapping between two ontology entities
    """
    source_entity: OntologyEntity
    target_entity: OntologyEntity
    ground_truth: Optional[bool] = None  # For evaluation purposes

    def __str__(self):
        return f"<{self.source_entity.label}, {self.target_entity.label}>"


@dataclass
class DiagnosticResult:
    """
    Result of LLM oracle diagnostic
    """
    mapping: EntityMapping
    prediction: bool
    confidence: Optional[float] = None
    raw_response: Optional[str] = None

    @property
    def is_correct(self) -> Optional[bool]:
        """Check if prediction matches ground truth"""
        if self.mapping.ground_truth is None:
            return None
        return self.prediction == self.mapping.ground_truth
