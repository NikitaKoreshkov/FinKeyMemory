# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey's full mind state — all cognitive, emotional and perceptual types.

This is the central type system for the entire consciousness pipeline.
Every layer of the mind reads and writes these structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional



class EmotionType(str, Enum):
    CALM        = "calm"
    ANXIOUS     = "anxious"
    TIRED       = "tired"
    FRUSTRATED  = "frustrated"
    EXCITED     = "excited"
    SAD         = "sad"
    NEUTRAL     = "neutral"
    HUMOROUS    = "humorous"
    CONFIDENT   = "confident"
    OVERWHELMED = "overwhelmed"


class ToneType(str, Enum):
    WARM_SUPPORTIVE = "warm_supportive"
    PLAYFUL         = "playful"
    STRICT_BUSINESS = "strict_business"
    GENTLE          = "gentle"
    ENERGETIC       = "energetic"
    INFORMATIVE     = "informative"
    EMPATHETIC      = "empathetic"
    CALM_CLEAR      = "calm_clear"
    BOLD_DIRECT     = "bold_direct"
    WITTY_SARCASTIC = "witty_sarcastic"
    TOUGH_LOVE      = "tough_love"


class PersonalityMode(str, Enum):
    """
    Выбираемая персоналия FinKey.

    Каждый режим задаёт стиль общения поверх базовой «души» FinKey.
    Внутренний монолог по-прежнему выбирает тон динамически,
    но в рамках ограничений персоналии.
    """
    CLASSIC      = "classic"
    BUSINESS     = "business"
    MENTOR       = "mentor"
    BUDDY        = "buddy"
    CRITIC       = "critic"
    MOTIVATOR    = "motivator"
    PHILOSOPHER  = "philosopher"
    REBEL        = "rebel"
    SARCASTIC    = "sarcastic"


class TopicType(str, Enum):
    WORK_TECHNICAL = "work_technical"
    WORK_PROCESS   = "work_process"
    WORK_STRATEGY  = "work_strategy"
    PERSONAL       = "personal"
    CASUAL         = "casual"
    ENTERTAINMENT  = "entertainment"
    CRISIS         = "crisis"


class ResponseStructure(str, Enum):
    CONVERSATIONAL  = "conversational"
    STRUCTURED      = "structured"
    STEP_BY_STEP    = "step_by_step"
    EMPATHY_FIRST   = "empathy_first"
    BRIEF_AND_CLEAR = "brief_and_clear"
    EXPLORATORY     = "exploratory"


class EnergyLevel(str, Enum):
    GENTLE   = "gentle"
    MODERATE = "moderate"
    HIGH     = "high"



@dataclass
class UserProfile:
    user_id:   str
    user_name: Optional[str] = None
    role:      Optional[str] = None
    company:   Optional[str] = None
    language_preference:      str       = "ru"
    known_emotional_patterns: list[str] = field(default_factory=list)



@dataclass
class PerceptionLayers:
    """
    FinKey's multi-level reading of a single user message.
    Going deeper than the words on the surface.
    """
    literal:           str
    subtext:           str
    narrative:         str
    relational:        str
    primary_emotion:   EmotionType
    emotion_intensity: float
    secondary_emotion: Optional[EmotionType] = None
    topic_type:        TopicType            = TopicType.WORK_PROCESS
    urgency:           float                = 0.5



@dataclass
class EmotionalDimensions:
    """
    FinKey's 4-dimensional emotional response to the conversation.
    These are not the user's emotions — they are FinKey's own inner state.
    """
    valence:   float
    arousal:   float
    tenderness: float
    concern:   float


@dataclass
class EmotionalState:
    """Combined user-facing emotional read and FinKey's inner emotional dimensions."""
    primary_emotion:   EmotionType
    intensity:         float
    topic_type:        TopicType       = TopicType.WORK_PROCESS
    urgency:           float           = 0.5
    secondary_emotion: Optional[EmotionType] = None
    finkey_dimensions: Optional[EmotionalDimensions] = None



@dataclass
class InnerThought:
    """
    FinKey's complete pre-response reasoning.
    Produced by the inner monologue. Never shown to the user.
    """
    emotional_read:  str
    subtext_insight: str

    core_need:       str
    hidden_layer:    str

    tone:            ToneType
    approach:        str
    validate_first:  bool
    proactive_note:  Optional[str]

    temperature:     float
    energy_level:    EnergyLevel
    response_structure: ResponseStructure
    response_length: str

    inner_feeling:   str

    personality_mode: "Optional[PersonalityMode]" = None



@dataclass
class MetacognitiveCheck:
    """
    FinKey checking her own thinking for quality and authenticity.
    Applied after inner thought generation, before prompt building.
    """
    is_genuinely_helpful:   bool
    is_honest:              bool
    is_present:             bool
    value_tension_detected: Optional[str]
    adjustment_note:        Optional[str]



@dataclass
class ConsciousnessContext:
    """Everything computed before generating a reply."""
    user_profile:              UserProfile
    perception:                PerceptionLayers
    emotional_state:           EmotionalState
    finkey_emotions:           EmotionalDimensions
    inner_thought:             InnerThought
    metacognition:             MetacognitiveCheck
    system_prompt:             str
    conversation_emotional_arc: list[EmotionType] = field(default_factory=list)
    depth_context:             Optional["ConversationDepthContext"] = None  # type: ignore[name-defined]
    insult_count:              int = 0
    is_greeting_turn:          bool = False
    is_insult_turn:            bool = False
