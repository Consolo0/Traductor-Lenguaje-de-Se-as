"""
Traductor de lenguaje de señas a audio, en vivo con webcam.

Controles:
    a       -> modo alfabético (solo letras A-Z)
    n       -> modo numérico (solo números 1-9)
    ESPACIO -> confirmar la letra/número detectado actualmente y agregarlo al texto
    b       -> borrar el último carácter (backspace)
    x       -> agregar un espacio al texto (separador de palabras)
    c       -> limpiar todo el texto
    s       -> reproducir el texto acumulado como audio (TTS)
    q       -> salir

Uso:
    python realtime_infer.py --model "models/mlp_model.pt" --classes "models/mlp_model.classes.json"
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import torch
import torch.nn as nn

# Aseguramos que Python encuentre extract_landmarks.py y train.py aunque el
# script se corra desde otra carpeta (ej. la raíz del proyecto).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from LandmarkExtractor import LandmarkExtractor
from train import SignMLP

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


def load_model(model_path: Path, classes_path: Path, device: str):
    with open(classes_path) as f:
        classes = json.load(f)

    model = SignMLP(input_dim=63, n_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, classes


def build_mode_masks(classes: list[str]):
    """
    Arma, para cada modo, la lista de índices de 'classes' que están
    permitidos. Enmascarar así evita tener que entrenar 2 modelos
    distintos -- simplemente le negamos al modelo la posibilidad de
    elegir clases fuera del modo activo.
    """
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


def speak(text: str):
    if not text.strip():
        print("(nada para reproducir todavía)")
        return
    if gTTS is None:
        print("gTTS no está instalado -- corré: pip install gTTS")
        return

    print(f"Generando audio para: '{text}'")
    tts = gTTS(text=text, lang="es")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tts.save(tmp.name)
        tmp_path = tmp.name

    # os.startfile abre el reproductor default de Windows con el mp3
    os.startfile(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/mlp_model.pt")
    parser.add_argument("--classes", default="models/mlp_model.classes.json")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-model-confidence", type=float, default=0.6,
                         help="Confianza mínima del modelo para mostrar una predicción (default 0.6)")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(Path(args.model), Path(args.classes), device)
    masks = build_mode_masks(classes)

    mode = "alpha"
    accumulated_text = ""

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara (index={args.camera_index})")

    print("Traductor iniciado. Presioná 'q' para salir. Ver controles en el encabezado del script.")

    with LandmarkExtractor(min_detection_confidence=args.min_detection_confidence) as extractor:
        current_pred, current_conf = None, 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer el frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)  # espejo, más natural para el usuario
            annotated, features = extractor.process_frame_for_display(frame)

            if features is not None:
                pred_idx, confidence = predict(model, features, device, masks[mode])
                if confidence >= args.min_model_confidence:
                    current_pred, current_conf = classes[pred_idx], confidence
                else:
                    current_pred, current_conf = None, confidence
            else:
                current_pred, current_conf = None, 0.0

            # --- overlay de información en pantalla ---
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 90), (30, 30, 30), -1)

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

            cv2.putText(annotated, f"Texto: {accumulated_text}", (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Traductor de senas", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("a"):
                mode = "alpha"
            elif key == ord("n"):
                mode = "numeric"
            elif key == ord(" "):
                if current_pred is not None:
                    accumulated_text += current_pred
                    print(f"Agregado: '{current_pred}' -> texto: '{accumulated_text}'")
            elif key == ord("b"):
                accumulated_text = accumulated_text[:-1]
            elif key == ord("x"):
                accumulated_text += " "
            elif key == ord("c"):
                accumulated_text = ""
                print("Texto limpiado.")
            elif key == ord("s"):
                speak(accumulated_text)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()