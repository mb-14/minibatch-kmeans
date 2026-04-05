"""Shared k-means backend method names for evaluation CLIs."""

from __future__ import annotations

from typing import Final, Literal

FLASHKMEANS: Final[str] = "flashkmeans"
FASTKMEANS: Final[str] = "fastkmeans"
MINIBATCHKMEANS: Final[str] = "minibatchkmeans"
SKLEARNKMEANS: Final[str] = "sklearnkmeans"
SKLEARNMINIBATCHKMEANS: Final[str] = "sklearnminibatchkmeans"
FAISSKMEANS: Final[str] = "faisskmeans"

MethodName = Literal[
    "flashkmeans",
    "fastkmeans",
    "minibatchkmeans",
    "sklearnkmeans",
    "sklearnminibatchkmeans",
    "faisskmeans",
]

VALID_METHODS: frozenset[str] = frozenset(
    {
        FLASHKMEANS,
        FASTKMEANS,
        MINIBATCHKMEANS,
        SKLEARNKMEANS,
        SKLEARNMINIBATCHKMEANS,
        FAISSKMEANS,
    }
)
