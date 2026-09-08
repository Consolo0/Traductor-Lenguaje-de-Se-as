"""
Dibuja el panel de información (modo, detección, barras de progreso de
cada InputBehavior, texto acumulado) como una franja separada ARRIBA
del frame de la cámara -- nunca tapa el video real.
"""

import cv2
import numpy as np


class OverlayRenderer:
    def __init__(self, panel_height: int = 110):
        self.panel_height = panel_height

    def render(self, frame, mode_label: str, current_pred, current_conf: float, behaviors, accumulated_text: str):
        h, w = frame.shape[:2]
        canvas = np.zeros((h + self.panel_height, w, 3), dtype=np.uint8)
        canvas[self.panel_height:, :] = frame
        panel = canvas[: self.panel_height, :]
        panel[:] = (30, 30, 30)

        cv2.putText(panel, f"Modo: {mode_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if current_pred is not None:
            text = f"Deteccion: {current_pred}  ({current_conf*100:.0f}%)"
            color = (0, 220, 0)
        else:
            text = f"Deteccion: --  ({current_conf*100:.0f}%)"
            color = (0, 0, 220)
        cv2.putText(panel, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # una barra de progreso por cada InputBehavior, generada
        # automáticamente -- agregar un behavior nuevo le agrega su
        # barra sin tocar este archivo.
        bar_x = 10
        for behavior in behaviors:
            progress = behavior.progress()
            bar_w = int(150 * progress)
            cv2.rectangle(panel, (bar_x, 65), (bar_x + 150, 75), (80, 80, 80), 1)
            cv2.rectangle(panel, (bar_x, 65), (bar_x + bar_w, 75), behavior.color, -1)
            bar_x += 160

        cv2.putText(panel, f"Texto: {accumulated_text}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return canvas