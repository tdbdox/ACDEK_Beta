from dataclasses import dataclass
from acdek.registry import register_semantic_class


@dataclass
class CognitiveState:
    """Represents the user's cognitive state based on behavioral and psychological indicators."""
    attention_span: float
    focus_level: float
    stress_index: float


register_semantic_class("CognitiveState", CognitiveState)


@dataclass
class EmotionalProfile:
    joy: float
    anxiety: float
    anger: float
    sadness: float

register_semantic_class("EmotionalProfile", EmotionalProfile)


@dataclass
class TeamDynamics:
    cohesion: float
    conflict_level: float
    shared_vision_score: float

register_semantic_class("TeamDynamics", TeamDynamics)


@dataclass
class ProductivityPattern:
    work_cycles: int
    average_focus_minutes: float
    task_completion_rate: float

register_semantic_class("ProductivityPattern", ProductivityPattern)


@dataclass
class CommunicationStyle:
    assertiveness: float
    clarity: float
    listening_ratio: float

register_semantic_class("CommunicationStyle", CommunicationStyle)


@dataclass
class DecisionBiasProfile:
    risk_aversion: float
    confirmation_bias: float
    decisiveness: float

register_semantic_class("DecisionBiasProfile", DecisionBiasProfile)


@dataclass
class EnvironmentalState:
    noise_level: float
    lighting_quality: float
    physical_distractions: float

register_semantic_class("EnvironmentalState", EnvironmentalState)


@dataclass
class WorkflowEfficiency:
    task_switch_rate: float
    idle_time_ratio: float
    avg_task_completion_time: float

register_semantic_class("WorkflowEfficiency", WorkflowEfficiency)


@dataclass
class KnowledgeCoverage:
    domain_expertise_score: float
    knowledge_gaps: float
    learning_velocity: float

register_semantic_class("KnowledgeCoverage", KnowledgeCoverage)


@dataclass
class ToolUsagePattern:
    tool_switch_frequency: float
    preferred_toolset: float
    automation_utilization_score: float

register_semantic_class("ToolUsagePattern", ToolUsagePattern)


@dataclass
class InteractionLoad:
    active_sessions: int
    request_frequency: float
    interaction_burstiness: float

register_semantic_class("InteractionLoad", InteractionLoad)


@dataclass
class BehavioralRhythm:
    peak_performance_window: str
    fatigue_onset_time: str
    rest_interval_frequency: float

register_semantic_class("BehavioralRhythm", BehavioralRhythm)


@dataclass
class TaskResilience:
    retry_rate: float
    frustration_tolerance: float
    recovery_time_seconds: float

register_semantic_class("TaskResilience", TaskResilience)


@dataclass
class CollaborationPattern:
    delegation_ratio: float
    feedback_frequency: float
    cross_team_sync_score: float

register_semantic_class("CollaborationPattern", CollaborationPattern)


@dataclass
class AttentionDrift:
    drift_frequency: float
    recovery_efficiency: float
    distraction_source_entropy: float

register_semantic_class("AttentionDrift", AttentionDrift)


@dataclass
class TemporalAwareness:
    deadline_adherence_rate: float
    time_estimation_accuracy: float
    temporal_conflict_index: float

register_semantic_class("TemporalAwareness", TemporalAwareness)


@dataclass
class LearningStyle:
    visual_preference: float
    auditory_preference: float
    kinesthetic_preference: float

register_semantic_class("LearningStyle", LearningStyle)


@dataclass
class BusinessAnalysis:
    """Represents a business analysis profile with focus on comparative metrics and trend identification."""
    analytical_depth: float
    pattern_recognition: float
    decision_style: str
    comparative_focus: float
    
register_semantic_class("BusinessAnalysis", BusinessAnalysis)


@dataclass
class RiskProfile:
    """Represents a risk analysis profile focusing on assessment, quantification, and strategic planning."""
    risk_tolerance: float
    analytical_depth: float
    decision_style: str
    evidence_threshold: float
    assertiveness: float
    
register_semantic_class("RiskProfile", RiskProfile)