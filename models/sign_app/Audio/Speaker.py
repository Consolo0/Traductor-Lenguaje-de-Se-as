"""Interfaz para salida de audio (texto -> voz)."""

from abc import ABC, abstractmethod


class Speaker(ABC):
    @abstractmethod
    def speak(self, text: str) -> bool:
        """Reproduce el texto como audio. Devuelve True si sonó."""
        raise NotImplementedError