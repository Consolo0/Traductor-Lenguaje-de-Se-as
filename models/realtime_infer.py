"""
Traductor de lenguaje de señas a audio, en vivo con webcam.

Controles:
    a       -> modo alfabético (solo letras A-Z)
    n       -> modo numérico (solo números 1-9)
    b       -> borrar el último carácter (backspace)
    x       -> agregar un espacio manual al texto
    c       -> limpiar todo el texto
    s       -> reproducir el texto acumulado como audio (TTS, en memoria, sin generar mp3)
    q       -> salir

La letra/número y el espacio se agregan automáticamente -- ver
sign_app/HoldConfirmer.py y sign_app/AutoSpacer.py.

Este archivo es solo el "punto de composición": arma las piezas
(clasificador, comportamientos de entrada, speaker, renderer) y corre
el loop de la cámara. Para agregar un comportamiento nuevo, no hace
falta tocar este archivo más que por la lista `behaviors` de abajo --
la lógica va en su propia clase dentro de sign_app/.

Uso:
    python realtime_infer.py --model "mlp_model.pt" --classes "mlp_model.classes.json"
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sign_app.FrameAnalyzer import LandmarkExtractor
from sign_app.SignModelLogic import SignClassifier
from sign_app.ConditionEvaluators import HoldConfirmer, AutoSpacer
from sign_app.Audio import GttsPygameSpeaker
from sign_app.ScreenLogic import OverlayRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlp_model.pt")
    parser.add_argument("--classes", default="mlp_model.classes.json")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-model-confidence", type=float, default=0.6)
    parser.add_argument("--hold-seconds", type=float, default=0.8)
    parser.add_argument("--space-gap-seconds", type=float, default=1.5)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    classifier = SignClassifier.SignClassifier(Path(args.model), Path(args.classes))
    speaker = GttsPygameSpeaker.GttsPygameSpeaker(lang="es")
    renderer = OverlayRenderer.OverlayRenderer(panel_height=110)

    # Comportamientos automáticos de entrada de texto. Para agregar uno
    # nuevo: crear la clase en su propio archivo dentro de sign_app/
    # (heredando de InputBehavior) y agregar la instancia acá -- nada
    # más cambia.
    behaviors = [
        HoldConfirmer.HoldConfirmer(hold_seconds=args.hold_seconds),
        AutoSpacer.AutoSpacer(gap_seconds=args.space_gap_seconds),
    ]

    mode = "alpha"
    accumulated_text = ""

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara (index={args.camera_index})")

    print("Traductor iniciado. Presioná 'q' para salir.")
    cv2.namedWindow("Traductor de senas", cv2.WINDOW_NORMAL)

    with LandmarkExtractor.LandmarkExtractor(
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
            hand_present = features is not None

            current_pred, current_conf = None, 0.0
            if features is not None:
                label, confidence = classifier.predict(features, mode)
                current_conf = confidence
                if confidence >= args.min_model_confidence:
                    current_pred = label

            for behavior in behaviors:
                action = behavior.update(hand_present, current_pred, accumulated_text)
                if action is not None:
                    accumulated_text += action.text_delta
                    print(f"[{behavior.label}] -> texto: '{accumulated_text}'")

            mode_label = "ALFABETICO (A-Z)" if mode == "alpha" else "NUMERICO (1-9)"
            canvas = renderer.render(
                annotated, mode_label, current_pred, current_conf, behaviors, accumulated_text
            )
            cv2.imshow("Traductor de senas", canvas)

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
                if speaker.speak(accumulated_text):
                    accumulated_text = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()