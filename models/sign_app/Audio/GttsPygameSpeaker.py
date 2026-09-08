"""Genera el audio con gTTS en memoria (sin archivo en disco) y lo reproduce con pygame."""

import io

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pygame
except ImportError:
    pygame = None

from .Speaker import Speaker


class GttsPygameSpeaker(Speaker):
    """
    Genera el audio con gTTS directo en memoria (io.BytesIO, sin tocar
    disco) y lo reproduce con pygame -- no abre ninguna app externa ni
    deja ningún .mp3 dando vueltas.
    """

    def __init__(self, lang: str = "es"):
        self.lang = lang

    def speak(self, text: str) -> bool:
        if not text.strip():
            print("(nada para reproducir todavía)")
            return False
        if gTTS is None or pygame is None:
            print("Falta instalar gTTS y/o pygame: pip install gTTS pygame")
            return False

        print(f"Generando audio para: '{text}'")
        tts = gTTS(text=text, lang=self.lang)

        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(buffer, "mp3")  # "mp3" como hint de formato, no hay archivo real
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

        return True