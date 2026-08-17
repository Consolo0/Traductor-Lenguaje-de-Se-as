"""
Extrae landmarks de mano (MediaPipe) de un dataset de imágenes organizado como:

    data/asl_dataset/<letra_o_numero>/imagen1.jpeg
    data/asl_dataset/<letra_o_numero>/imagen2.jpeg
    ...

Genera un CSV con 63 features (21 landmarks x,y,z normalizados) + columna label.

Nota: la clase "0" fue excluida del dataset. Se determinó (con el notebook
mediapipe_diagnostic.ipynb) que esa seña se forma con 2 manos superpuestas
(una plana de fondo, otra formando el círculo), y ni con max_num_hands=2
la detección resultaba confiable de forma consistente -- la mayoría de las
imágenes detectaban solo 1 mano, y a veces era la mano equivocada (la de
fondo, no la que forma la seña). Con esa clase afuera, ninguna otra seña
del dataset requiere 2 manos, así que el extractor vuelve a la versión
simple de 1 mano.

Uso:
    python extract_landmarks.py --input "data/asl_dataset" --output "data/landmarks/landmarks.csv"
"""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class LandmarkExtractor:
    """
    Encapsula la extracción y normalización de landmarks de mano con MediaPipe.

    Se usa como clase (y no funciones sueltas) porque la misma instancia se
    va a reusar en realtime_infer.py para procesar frames de la webcam en
    vivo -> así garantizamos que el preprocesamiento de entrenamiento e
    inferencia sea idéntico (evita distribution shift).
    """

    HEADER = (
        [f"x{i}" for i in range(21)]
        + [f"y{i}" for i in range(21)]
        + [f"z{i}" for i in range(21)]
    )

    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.5, static_image_mode: bool = True):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._hands.close()

    @staticmethod
    def _normalize(landmarks) -> list[float]:
        """
        Normaliza los 21 landmarks (x, y, z) para que el modelo sea invariante a:
          - posición de la mano en el frame (traslación)
          - distancia de la mano a la cámara (escala)
        """
        coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        wrist = coords[0]

        translated = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in coords]

        ref_x, ref_y, ref_z = translated[9]
        scale = (ref_x ** 2 + ref_y ** 2 + ref_z ** 2) ** 0.5
        if scale < 1e-6:
            scale = 1e-6

        normalized = [(x / scale, y / scale, z / scale) for x, y, z in translated]
        return [v[0] for v in normalized] + [v[1] for v in normalized] + [v[2] for v in normalized]

    def extract_from_image(self, image_bgr) -> list[float] | None:
        """
        Recibe una imagen BGR, devuelve el vector normalizado de 63 features,
        o None si no se detectó ninguna mano.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        return self._normalize(results.multi_hand_landmarks[0])

    def extract_from_path(self, image_path: Path) -> list[float] | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return self.extract_from_image(image)

    def process_dataset(self, input_dir: Path, output_path: Path, exclude: set[str] | None = None) -> dict:
        exclude = exclude or set()

        letter_dirs = sorted(
            d for d in input_dir.iterdir() if d.is_dir() and d.name not in exclude
        )
        if not letter_dirs:
            raise SystemExit(f"No se encontraron subcarpetas en {input_dir} (después de excluir {exclude})")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_ok, total_failed = 0, 0
        failed_by_letter = {}

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER + ["label"])

            for letter_dir in letter_dirs:
                label = letter_dir.name
                image_paths = [p for p in letter_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
                failed_by_letter[label] = 0

                for img_path in tqdm(image_paths, desc=f"Procesando '{label}'"):
                    features = self.extract_from_path(img_path)
                    if features is None:
                        total_failed += 1
                        failed_by_letter[label] += 1
                        continue

                    writer.writerow(features + [label])
                    total_ok += 1

        return {
            "total_ok": total_ok,
            "total_failed": total_failed,
            "failed_by_letter": failed_by_letter,
            "output_path": output_path,
        }


def print_summary(stats: dict):
    total = stats["total_ok"] + stats["total_failed"]
    rate = (stats["total_ok"] / total * 100) if total else 0.0

    print("\n=== Resumen de extracción ===")
    print(f"Imágenes procesadas con éxito: {stats['total_ok']}")
    print(f"Imágenes descartadas (sin mano detectada o ilegibles): {stats['total_failed']}")
    print(f"Tasa de detección: {rate:.1f}%")
    print("\nDescartes por letra/número:")
    for label, count in sorted(stats["failed_by_letter"].items()):
        if count > 0:
            print(f"  {label}: {count}")
    print(f"\nCSV guardado en: {stats['output_path']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Carpeta raíz del dataset (data/asl_dataset)")
    parser.add_argument("--output", required=True, help="Ruta del CSV de salida")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["0"],
        help="Nombres de carpetas/clases a excluir (default: '0')",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Umbral de confianza de detección de MediaPipe (default 0.5)",
    )
    args = parser.parse_args()

    with LandmarkExtractor(min_detection_confidence=args.min_detection_confidence) as extractor:
        stats = extractor.process_dataset(
            Path(args.input), Path(args.output), exclude=set(args.exclude)
        )

    print_summary(stats)


if __name__ == "__main__":
    main()