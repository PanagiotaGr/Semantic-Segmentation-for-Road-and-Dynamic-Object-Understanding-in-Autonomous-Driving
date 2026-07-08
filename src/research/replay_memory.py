"""Replay memory utilities for continual learning experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generic, Iterable, List, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ReplayItem(Generic[T]):
    """One stored replay sample."""

    item: T
    domain: str


class ReplayMemory(Generic[T]):
    """Fixed-size replay memory with random replacement.

    This lightweight utility stores examples from previous domains so that
    continual-learning experiments can mix old and new samples during training.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: List[ReplayItem[T]] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: T, domain: str) -> None:
        replay_item = ReplayItem(item=item, domain=domain)
        if len(self._items) < self.capacity:
            self._items.append(replay_item)
            return
        index = self._rng.randrange(self.capacity)
        self._items[index] = replay_item

    def extend(self, items: Iterable[T], domain: str) -> None:
        for item in items:
            self.add(item, domain=domain)

    def sample(self, batch_size: int) -> List[ReplayItem[T]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not self._items:
            return []
        size = min(batch_size, len(self._items))
        return self._rng.sample(self._items, size)

    def domains(self) -> Sequence[str]:
        return sorted({entry.domain for entry in self._items})

    def items(self) -> Sequence[ReplayItem[T]]:
        return tuple(self._items)


__all__ = ["ReplayItem", "ReplayMemory"]
