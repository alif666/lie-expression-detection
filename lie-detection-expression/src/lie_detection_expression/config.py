"""Shared configuration for model and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PipelineConfig:
    min_confidence: float = 0.7
    require_success: bool = True
    fps: int = 30
    window_seconds: float = 2.0
    hidden_layers: tuple[int, ...] = field(default_factory=lambda: (128, 64, 32))
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 200

    @property
    def window_size_frames(self) -> int:
        return int(self.fps * self.window_seconds)

