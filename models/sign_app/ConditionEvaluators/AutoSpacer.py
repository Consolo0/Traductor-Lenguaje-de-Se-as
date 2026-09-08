"""Agrega un espacio automáticamente cuando no se detecta ninguna mano por un tiempo."""

import time
from typing import Optional

from .InputBehavior import InputBehavior
from ..TextLogic import TextAction


class AutoSpacer(InputBehavior):
    """
    Agrega un espacio cuando no se detecta ninguna mano durante
    `gap_seconds` -- bajar la mano entre palabras sirve como señal de
    "fin de palabra", sin entrenar el modelo con una clase "espacio"
    inventada.
    """

    color = (0, 165, 255)
    label = "espacio auto"

    def __init__(self, gap_seconds: float):
        self.gap_seconds = gap_seconds
        self._hand_absent_since = None
        self._space_inserted_for_this_gap = False

    def update(self, hand_present, current_pred, text_so_far) -> Optional[TextAction.TextAction]:
        if hand_present:
            self._hand_absent_since = None
            self._space_inserted_for_this_gap = False
            return None

        now = time.time()
        if self._hand_absent_since is None:
            self._hand_absent_since = now
            return None

        elapsed = now - self._hand_absent_since
        should_add = (
            elapsed >= self.gap_seconds
            and not self._space_inserted_for_this_gap
            and bool(text_so_far)
            and not text_so_far.endswith(" ")
        )
        if should_add:
            self._space_inserted_for_this_gap = True
            return TextAction.TextAction(text_delta=" ")

        return None

    def progress(self) -> float:
        if self._hand_absent_since is None or self._space_inserted_for_this_gap:
            return 0.0
        elapsed = time.time() - self._hand_absent_since
        return min(elapsed / self.gap_seconds, 1.0)