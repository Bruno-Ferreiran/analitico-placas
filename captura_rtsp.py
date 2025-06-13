
import os
import time
import hashlib
import re                                     
from datetime import datetime
import subprocess as sp
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from easyocr import Reader

# ===== CONFIGURAÇÕES =====
RTSP_URL = 'rtsp://admin:marina100@170.238.163.34:8105/cam/realmonitor?channel=1&subtype=0'
MODEL_PATH = 'models/plate_det.pt'
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45
IMG_SIZE = (640, 640)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
ROI_Y1, ROI_Y2 = 300, 1080
SNAPSHOT_INTERVAL = 10.0
FRAME_SKIP = 2

# ===== GPU =====
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
print(f"[INFO] GPU disponível: {USE_CUDA}")

# ===== DIRETÓRIOS =====
os.makedirs('output/plates', exist_ok=True)
os.makedirs('output/snapshots', exist_ok=True)
os.makedirs('output/debug', exist_ok=True)

# ===== MODELOS =====
detector = YOLO(MODEL_PATH)
reader = Reader(['pt', 'en'], gpu=USE_CUDA)

# ===== FFMPEG (RTSP) =====
ffmpeg_cmd = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',
    '-i', RTSP_URL,
    '-flags', '+low_delay',
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-loglevel', 'quiet',
    '-'
]
pipe = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, bufsize=10**6)

# ===== CONTROLE =====
last_plate = None
last_snapshot_time = time.time()
last_frame_hash = None
frame_count = 0

try:
    while True:
        loop_start = time.time()
        raw_image = pipe.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
        if len(raw_image) != FRAME_WIDTH * FRAME_HEIGHT * 3:
            print(f"[{datetime.now():%H:%M:%S}] ⚠️ Frame corrompido. Pulando...")
            continue

        # Verifica frame repetido
        frame_hash = hashlib.md5(raw_image).hexdigest()
        if frame_hash == last_frame_hash:
            continue
        last_frame_hash = frame_hash

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        roi = frame[ROI_Y1:ROI_Y2, :]
        roi_draw = roi.copy()

        now = time.time()
        if now >= last_snapshot_time + SNAPSHOT_INTERVAL:
            ts_snap = datetime.now().strftime('%Y%m%d_%H%M%S')
            snap_path = f"output/snapshots/roi_{ts_snap}.jpg"
            cv2.imwrite(snap_path, roi)
            print(f"[{ts_snap}] 📸 Snapshot salvo em {snap_path}")
            last_snapshot_time += SNAPSHOT_INTERVAL

        # Redimensiona para YOLO
        img = cv2.resize(roi, IMG_SIZE)

        try:
            results = detector.predict(
                source=img,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=5,
                device=DEVICE,
                verbose=False
            )
        except Exception as e:
            print(f"[YOLO] Erro: {e}")
            continue

        # Processa resultados
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(boxes, scores):
                if score < CONF_THRESHOLD:
                    continue

                # Ajusta coordenadas de volta ao ROI
                h_ratio = roi.shape[0] / IMG_SIZE[1]
                w_ratio = roi.shape[1] / IMG_SIZE[0]
                x1o, y1o, x2o, y2o = [
                    int(v * w_ratio) if i % 2 == 0 else int(v * h_ratio)
                    for i, v in enumerate([x1, y1, x2, y2])
                ]
                plate_img = roi[y1o:y2o, x1o:x2o]

                # ===== EXTRAÇÃO E LIMPEZA DO TEXTO =====
                try:
                    texts_gray = reader.readtext(
                        cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY),
                        detail=0,
                        paragraph=False
                    )
                    texts_color = reader.readtext(
                        plate_img,
                        detail=0,
                        paragraph=False
                    )
                    texts = texts_gray + texts_color

                    # limpa e extrai padrão AAA-0000
                    candidates = []
                    for t in texts:
                        raw = t.strip().upper()
                        clean = re.sub(r'[^A-Z0-9]', '', raw)      # tira tudo que não for letra ou número
                        m = re.search(r'[A-Z]{3}\d{4}', clean)     # busca três letras + quatro dígitos
                        if m:
                            p = m.group()
                            # insere hífen no meio: 'ABC1234' → 'ABC-1234'
                            p = p[:3] + '-' + p[3:]
                            candidates.append(p)

                    placa = candidates[0] if candidates else None

                except Exception as e:
                    print(f"[OCR] Erro: {e}")
                    placa = None

                # só salva/nomeia se diferente da última e não for None
                if placa and placa != last_plate:
                    last_plate = placa
                    ts_plate = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_path = f"output/plates/{placa}_{ts_plate}.jpg"
                    cv2.imwrite(out_path, plate_img)
                    print(f"[{ts_plate}] ✅ Placa: {placa}, conf={score:.2f} → {out_path}")

                # desenha retângulo e label
                cv2.rectangle(roi_draw, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                label = placa if placa else "Desconhecida"
                cv2.putText(
                    roi_draw, label, (x1o, y1o - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        # debug e tempo de loop
        timestamp_dbg = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        cv2.putText(
            roi_draw, timestamp_dbg, (10, roi.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        debug_path = f"output/debug/frame_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(debug_path, roi_draw)

        print(f"[DEBUG] Loop: {time.time() - loop_start:.2f}s")

except KeyboardInterrupt:
    print(f"[{datetime.now():%H:%M:%S}] 🛑 Interrompido pelo usuário.")
finally:
    pipe.terminate()
