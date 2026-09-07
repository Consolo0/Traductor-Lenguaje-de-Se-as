"""
Traductor de lenguaje de señas a audio, en vivo con webcam.

Controles:
    a       -> modo alfabético (solo letras A-Z)
    n       -> modo numérico (solo números 1-9)
    b       -> borrar el último carácter (backspace)
    x       -> agregar un espacio al texto (separador de palabras)
    c       -> limpiar todo el texto
    s       -> reproducir el texto acumulado como audio (TTS, en memoria, sin generar mp3)
    q       -> salir

La letra/número se agrega SOLO al texto automáticamente: sostené la seña
sin moverla durante --hold-seconds (default 0.8s) y se confirma sola.
Para repetir la misma letra dos veces seguidas (ej. "SS"), soltá la mano
o cambiá de seña brevemente entre una y otra -- si no, se interpreta
como "seguís sosteniendo la misma", no como dos confirmaciones distintas.

Uso:
    python realtime_infer.py --model "mlp_model.pt" --classes "mlp_model.classes.json"
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from LandmarkExtractor import LandmarkExtractor
from train import SignMLP

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pygame
except ImportError:
    pygame = None


def load_model(model_path: Path, classes_path: Path, device: str):
    with open(classes_path) as f:
        classes = json.load(f)

    model = SignMLP(input_dim=63, n_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, classes


def build_mode_masks(classes: list[str]):
    numeric_idx = [i for i, c in enumerate(classes) if c.isdigit()]
    alpha_idx = [i for i, c in enumerate(classes) if c.isalpha()]
    return {"numeric": numeric_idx, "alpha": alpha_idx}


def predict(model, features, device, allowed_idx):
    x = torch.tensor([features], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)[0]

    masked_logits = torch.full_like(logits, float("-inf"))
    masked_logits[allowed_idx] = logits[allowed_idx]

    probs = torch.softmax(masked_logits, dim=0)
    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())

    return pred_idx, confidence


class HoldConfirmer:
    """
    Reemplaza la confirmación manual (ESPACIO) por una automática basada
    en tiempo: si la misma predicción se mantiene estable por
    `hold_seconds`, se considera "confirmada".

    Para evitar que sostener la mano quieta 3 segundos agregue la misma
    letra varias veces, una vez confirmada una predicción hay que ver
    OTRA predicción (o ninguna) antes de poder volver a confirmar esa
    misma letra.
    """

    def __init__(self, hold_seconds: float):
        self.hold_seconds = hold_seconds
        self._stable_pred = None
        self._stable_since = None
        self._last_confirmed = None

    def update(self, current_pred):
        """
        current_pred: la letra/número predicho este frame, o None si no
        hay predicción confiable.

        Devuelve la letra a confirmar/agregar este frame, o None si
        todavía no corresponde confirmar nada.
        """
        now = time.time()

        if current_pred != self._stable_pred:
            # cambió la predicción (o se perdió la mano) -> reiniciar el conteo
            self._stable_pred = current_pred
            self._stable_since = now
            if current_pred is None:
                self._last_confirmed = None  # soltó la mano -> permite repetir letra
            return None

        if current_pred is None:
            return None

        held_for = now - self._stable_since
        if held_for >= self.hold_seconds and current_pred != self._last_confirmed:
            self._last_confirmed = current_pred
            return current_pred

        return None

    def progress(self) -> float:
        """Fracción [0, 1] de cuánto falta para confirmar (para mostrar en pantalla)."""
        if self._stable_pred is None or self._stable_since is None:
            return 0.0
        elapsed = time.time() - self._stable_since
        return min(elapsed / self.hold_seconds, 1.0)


def speak(text: str) -> bool:
    """
    Genera el audio con gTTS directo en memoria (io.BytesIO, sin tocar
    disco) y lo reproduce con pygame -- no se abre ninguna app externa
    ni queda ningún .mp3 dando vueltas.

    Devuelve True si el audio se generó y reprodujo, False si no había
    nada que decir o faltan librerías (para no borrar el texto en esos casos).
    """
    if not text.strip():
        print("(nada para reproducir todavía)")
        return False
    if gTTS is None or pygame is None:
        print("Falta instalar gTTS y/o pygame: pip install gTTS pygame")
        return False

    print(f"Generando audio para: '{text}'")
    tts = gTTS(text=text, lang="es")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlp_model.pt")
    parser.add_argument("--classes", default="mlp_model.classes.json")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-model-confidence", type=float, default=0.6)
    parser.add_argument("--hold-seconds", type=float, default=0.8,
                         help="Segundos que hay que sostener una seña estable para confirmarla (default 0.8)")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(Path(args.model), Path(args.classes), device)
    masks = build_mode_masks(classes)
    confirmer = HoldConfirmer(hold_seconds=args.hold_seconds)

    mode = "alpha"
    accumulated_text = ""

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara (index={args.camera_index})")

    print("Traductor iniciado. Presioná 'q' para salir. Ver controles en el encabezado del script.")

    with LandmarkExtractor(
        min_detection_confidence=args.min_detection_confidence,
        static_image_mode=False,
    ) as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer el frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            annotated, features = extractor.process_frame_for_display(frame)

            current_pred, current_conf = None, 0.0
            if features is not None:
                pred_idx, confidence = predict(model, features, device, masks[mode])
                current_conf = confidence
                if confidence >= args.min_model_confidence:
                    current_pred = classes[pred_idx]

            confirmed = confirmer.update(current_pred)
            if confirmed is not None:
                accumulated_text += confirmed
                print(f"Agregado: '{confirmed}' -> texto: '{accumulated_text}'")

            # --- overlay ---
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 110), (30, 30, 30), -1)

            mode_label = "ALFABETICO (A-Z)" if mode == "alpha" else "NUMERICO (1-9)"
            cv2.putText(annotated, f"Modo: {mode_label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if current_pred is not None:
                pred_text = f"Deteccion: {current_pred}  ({current_conf*100:.0f}%)"
                color = (0, 220, 0)
            else:
                pred_text = f"Deteccion: --  ({current_conf*100:.0f}%)"
                color = (0, 0, 220)
            cv2.putText(annotated, pred_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # barra de progreso de "sostener para confirmar"
            progress = confirmer.progress()
            bar_w = int(200 * progress)
            cv2.rectangle(annotated, (10, 65), (210, 75), (80, 80, 80), 1)
            cv2.rectangle(annotated, (10, 65), (10 + bar_w, 75), (0, 220, 0), -1)

            cv2.putText(annotated, f"Texto: {accumulated_text}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Traductor de senas", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("a"):
                mode = "alpha"
            elif key == ord("n"):
                mode = "numeric"
            elif key == ord("b"):
                accumulated_text = accumulated_text[:-1]
            elif key == ord("x"):
                accumulated_text += " "
            elif key == ord("c"):
                accumulated_text = ""
                print("Texto limpiado.")
            elif key == ord("s"):
                if speak(accumulated_text):
                    accumulated_text = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()