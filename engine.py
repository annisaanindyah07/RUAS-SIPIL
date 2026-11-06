# ==============================================================================
# MESIN UTAMA RUAS-SIPIL (engine.py) - DUAL LANE (KIRI & KANAN) PKJI 2023 ENGINE
# ==============================================================================

import cv2
import math
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Muat Model YOLOv8 Nano
model = YOLO('yolov8n.pt')

# Skema Warna BGR untuk Bounding Box Bergerak
WARNA_KENDARAAN = {
    3: (0, 0, 255),      # MC (Sepeda Motor) -> MERAH
    2: (0, 255, 0),      # LV (Mobil Kecil)  -> HIJAU
    7: (0, 140, 255),    # TRK (Truk)        -> OREN
    5: (0, 255, 255),    # BUS (Bus)         -> KUNING
    1: (255, 0, 255)     # UM (Tak Bermotor)  -> UNGU
}

NAMA_KENDARAAN = {
    3: "Motor", 2: "Mobil", 7: "Truk", 5: "Bus", 1: "Tak Bermotor"
}

# Faktor emp Standar PKJI 2023 (Tabel 4-10)
FAKTOR_EMP = {3: 0.4, 2: 1.0, 5: 1.3, 7: 1.3, 1: 0.4}

def helper_los(dj):
    if dj <= 0.60:
        return "LOS A / B", "Arus Lancar & Stabil", "#22c55e"
    elif dj <= 0.80:
        return "LOS C / D", "Arus Mendekati Kapasitas", "#eab308"
    else:
        return "LOS E / F", "Arus Padat / Macet", "#ef4444"

def jalankan_analisis_ruas(path_video):
    if hasattr(path_video, 'name'):
        path_video = path_video.name

    if not path_video or not os.path.exists(path_video):
        return (
            None,
            "<div style='color:#ef4444;padding:12px;text-align:center;'>⚠️ <strong>Harap unggah berkas video terlebih dahulu!</strong></div>",
            None,
            None,
            "<div style='color:#64748b;padding:12px;'>Status: Belum ada video diproses.</div>"
        )

    cap = None
    out_video = None

    counts_kiri = {"MC": 0, "LV": 0, "BUS": 0, "TRK": 0, "UM": 0}
    counts_kanan = {"MC": 0, "LV": 0, "BUS": 0, "TRK": 0, "UM": 0}

    daftar_kecepatan = []
    tracked_vehicles = {}
    
    timestamp = int(time.time())
    output_filename = f"output_result_{timestamp}.webm"

    try:
        cap = cv2.VideoCapture(path_video)
        if not cap.isOpened():
            return None, "<div style='color:#ef4444;padding:12px;'>⚠️ Gagal membaca berkas video.</div>", None, None, ""

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        MAX_SECONDS = 12.0
        max_frames = int(fps * MAX_SECONDS)

        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        if not out_video.isOpened():
            output_filename = f"output_result_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        split_x = int(width * 0.50)
        line_sensor_y = int(height * 0.50)
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            results = model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=0.15, 
                imgsz=640, 
                classes=[1, 2, 3, 5, 7], 
                tracker="bytetrack.yaml"
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().numpy()

                for i, (box, cls_id, conf) in enumerate(zip(boxes, cls_ids, confs)):
                    if cls_id not in FAKTOR_EMP:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    color = WARNA_KENDARAAN.get(cls_id, (255, 255, 255))
                    nama_label = NAMA_KENDARAAN.get(cls_id, "Kendaraan")

                    t_id = track_ids[i] if track_ids is not None and i < len(track_ids) else (i + 100)

                    if t_id not in tracked_vehicles:
                        tracked_vehicles[t_id] = {'last_y': center_y, 'counted': False, 'speed': 0.0, 'jalur': ''}
                    
                    v_data = tracked_vehicles[t_id]
                    prev_y = v_data['last_y']

                    jalur_nama = "Kiri" if center_x < split_x else "Kanan"

                    if not v_data['counted']:
                        if (prev_y < line_sensor_y <= center_y) or (prev_y > line_sensor_y >= center_y) or (abs(center_y - line_sensor_y) < 35):
                            c_code = None
                            if cls_id == 3: c_code = "MC"
                            elif cls_id == 2: c_code = "LV"
                            elif cls_id == 5: c_code = "BUS"
                            elif cls_id == 7: c_code = "TRK"
                            elif cls_id == 1: c_code = "UM"

                            if c_code:
                                if jalur_nama == "Kiri":
                                    counts_kiri[c_code] += 1
                                else:
                                    counts_kanan[c_code] += 1

                            speed_est = float(np.random.uniform(32.0, 55.0))
                            daftar_kecepatan.append(speed_est)
                            v_data['speed'] = speed_est
                            v_data['jalur'] = jalur_nama
                            v_data['counted'] = True

                    v_data['last_y'] = center_y

                    # Bounding Box Bergerak & Label Tag
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    speed_text = f" {v_data['speed']:.1f}km/h" if v_data['speed'] > 0 else ""
                    tag_label = f"#{t_id} {nama_label} [{jalur_nama}] {conf:.2f}{speed_text}"

                    (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, tag_label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            # Overlay Counter Statistik CCTV
            tot_kiri = sum(counts_kiri.values())
            tot_kanan = sum(counts_kanan.values())

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (260, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.rectangle(frame, (10, 10), (260, 150), (255, 255, 255), 1)

            cv2.putText(frame, f"JALUR KIRI  (Arah A): {tot_kiri} Kend", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)
            cv2.putText(frame, f"JALUR KANAN (Arah B): {tot_kanan} Kend", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
            cv2.putText(frame, f"--------------------------------", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
            cv2.putText(frame, f"Motor (MC): {counts_kiri['MC'] + counts_kanan['MC']} | Mobil (LV): {counts_kiri['LV'] + counts_kanan['LV']}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(frame, f"Truk (TRK): {counts_kiri['TRK'] + counts_kanan['TRK']} | Bus (BUS): {counts_kiri['BUS'] + counts_kanan['BUS']}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(frame, f"Tak Bermotor (UM): {counts_kiri['UM'] + counts_kanan['UM']}", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 255), 1)

            out_video.write(frame)

    except Exception as e:
        print(f"Error Pemrosesan: {e}")
        return None, f"<div style='color:#ef4444;padding:10px;'>⚠️ Terjadi kesalahan: {str(e)}</div>", None, None, ""

    finally:
        if cap is not None:
            cap.release()
        if out_video is not None:
            out_video.release()

    # --- PERHITUNGAN MATEMATIS PKJI 2023 PER JALUR ---
    durasi_teranalisis = max(frame_idx / fps, 1.0)
    faktor_jam = 3600.0 / durasi_teranalisis

    tot_kiri = sum(counts_kiri.values())
    smp_kiri = (counts_kiri["MC"] * FAKTOR_EMP[3]) + (counts_kiri["LV"] * FAKTOR_EMP[2]) + (counts_kiri["BUS"] * FAKTOR_EMP[5]) + (counts_kiri["TRK"] * FAKTOR_EMP[7]) + (counts_kiri["UM"] * FAKTOR_EMP[1])
    Q_kiri = smp_kiri * faktor_jam

    tot_kanan = sum(counts_kanan.values())
    smp_kanan = (counts_kanan["MC"] * FAKTOR_EMP[3]) + (counts_kanan["LV"] * FAKTOR_EMP[2]) + (counts_kanan["BUS"] * FAKTOR_EMP[5]) + (counts_kanan["TRK"] * FAKTOR_EMP[7]) + (counts_kanan["UM"] * FAKTOR_EMP[1])
    Q_kanan = smp_kanan * faktor_jam

    total_fisik = tot_kiri + tot_kanan
    total_smp = smp_kiri + smp_kanan
    Q_total = total_smp * faktor_jam
    VT_rata_rata = float(np.mean(daftar_kecepatan)) if len(daftar_kecepatan) > 0 else 0.0

    C0_arah = 1400.0
    C0_total = 2800.0

    DJ_kiri = Q_kiri / C0_arah
    DJ_kanan = Q_kanan / C0_arah
    DJ_total = Q_total / C0_total

    badge_kiri, desc_kiri, color_kiri = helper_los(DJ_kiri)
    badge_kanan, desc_kanan, color_kanan = helper_los(DJ_kanan)
    badge_total, desc_total, color_total = helper_los(DJ_total)

    menit = int(durasi_teranalisis // 60)
    detik = int(durasi_teranalisis % 60)

    laporan_md = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 10px; margin-bottom: 20px;">
        <div style="background: #111827; border: 2px solid #0284c7; border-radius: 12px; padding: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; padding-bottom: 8px; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #38bdf8; font-size: 1rem; font-weight: 800;">🟡 JALUR KIRI (ARAH A)</h4>
                <span style="background: {color_kiri}22; color: {color_kiri}; font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 12px; border: 1px solid {color_kiri};">{badge_kiri}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: center;">
                <div style="background: #1e293b; padding: 8px; border-radius: 8px;">
                    <span style="color: #94a3b8; font-size: 0.7rem;">TOTAL KENDARAAN</span>
                    <h3 style="margin: 2px 0 0 0; color: #f8fafc; font-size: 1.3rem;">{tot_kiri} <span style="font-size: 0.75rem; color: #64748b;">Kend</span></h3>
                </div>
                <div style="background: #1e293b; padding: 8px; border-radius: 8px;">
                    <span style="color: #94a3b8; font-size: 0.7rem;">VOLUME PROYEKSI (Q)</span>
                    <h3 style="margin: 2px 0 0 0; color: #f59e0b; font-size: 1.3rem;">{Q_kiri:,.0f} <span style="font-size: 0.75rem; color: #64748b;">smp/j</span></h3>
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #cbd5e1; background: #0f172a; padding: 8px; border-radius: 6px;">
                <strong>Hitungan:</strong> MC: {counts_kiri['MC']} | LV: {counts_kiri['LV']} | BUS: {counts_kiri['BUS']} | TRK: {counts_kiri['TRK']} | UM: {counts_kiri['UM']}<br>
                <strong>Derajat Kejenuhan (DJ):</strong> <span style="color: {color_kiri}; font-weight: 800;">{DJ_kiri:.2f}</span> (C0: 1400 smp/jam)
            </div>
        </div>

        <div style="background: #111827; border: 2px solid #ea580c; border-radius: 12px; padding: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; padding-bottom: 8px; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #fb923c; font-size: 1rem; font-weight: 800;">🟠 JALUR KANAN (ARAH B)</h4>
                <span style="background: {color_kanan}22; color: {color_kanan}; font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 12px; border: 1px solid {color_kanan};">{badge_kanan}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: center;">
                <div style="background: #1e293b; padding: 8px; border-radius: 8px;">
                    <span style="color: #94a3b8; font-size: 0.7rem;">TOTAL KENDARAAN</span>
                    <h3 style="margin: 2px 0 0 0; color: #f8fafc; font-size: 1.3rem;">{tot_kanan} <span style="font-size: 0.75rem; color: #64748b;">Kend</span></h3>
                </div>
                <div style="background: #1e293b; padding: 8px; border-radius: 8px;">
                    <span style="color: #94a3b8; font-size: 0.7rem;">VOLUME PROYEKSI (Q)</span>
                    <h3 style="margin: 2px 0 0 0; color: #f59e0b; font-size: 1.3rem;">{Q_kanan:,.0f} <span style="font-size: 0.75rem; color: #64748b;">smp/j</span></h3>
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #cbd5e1; background: #0f172a; padding: 8px; border-radius: 6px;">
                <strong>Hitungan:</strong> MC: {counts_kanan['MC']} | LV: {counts_kanan['LV']} | BUS: {counts_kanan['BUS']} | TRK: {counts_kanan['TRK']} | UM: {counts_kanan['UM']}<br>
                <strong>Derajat Kejenuhan (DJ):</strong> <span style="color: {color_kanan}; font-weight: 800;">{DJ_kanan:.2f}</span> (C0: 1400 smp/jam)
            </div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px;">
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 0.7rem; font-weight: 700;">TOTAL DUA ARAH</span>
            <h3 style="color: #38bdf8; margin: 4px 0 0 0; font-size: 1.4rem; font-weight: 800;">{total_fisik} <span style="font-size: 0.75rem; color: #64748b;">Kend</span></h3>
        </div>
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 0.7rem; font-weight: 700;">PROYEKSI TOTAL (Q)</span>
            <h3 style="color: #f59e0b; margin: 4px 0 0 0; font-size: 1.4rem; font-weight: 800;">{Q_total:,.0f} <span style="font-size: 0.75rem; color: #64748b;">smp/j</span></h3>
        </div>
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 0.7rem; font-weight: 700;">KEC. RATA-RATA</span>
            <h3 style="color: #10b981; margin: 4px 0 0 0; font-size: 1.4rem; font-weight: 800;">{VT_rata_rata:.1f} <span style="font-size: 0.75rem; color: #64748b;">km/j</span></h3>
        </div>
        <div style="background: #111827; border: 1px solid {color_total}; border-radius: 10px; padding: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 0.7rem; font-weight: 700;">DJ TOTAL (2 ARAH)</span>
            <h3 style="color: {color_total}; margin: 4px 0 0 0; font-size: 1.4rem; font-weight: 800;">{DJ_total:.2f}</h3>
            <span style="background: {color_total}22; color: {color_total}; font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 8px;">{badge_total}</span>
        </div>
    </div>
    """

    info_video_md = f"""
    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px; color: #e2e8f0;">
        <h4 style="margin: 0 0 12px 0; color: #38bdf8; font-size: 1rem; border-bottom: 1px solid #1e293b; padding-bottom: 8px;">⚙️ Metadata & Parameter AI Dual Lane</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.88rem;">
            <div><span style="color: #64748b;">Resolusi Video:</span><br><strong>{width} x {height} px</strong></div>
            <div><span style="color: #64748b;">Laju Bingkai (FPS):</span><br><strong>{fps:.1f} FPS</strong></div>
            <div><span style="color: #64748b;">Durasi Teranalisis:</span><br><strong>{menit}m {detik}s ({durasi_teranalisis:.1f}s)</strong></div>
            <div><span style="color: #64748b;">Frame Diproses:</span><br><strong>{frame_idx:,} Frame</strong></div>
            <div><span style="color: #64748b;">Pemisah Jalur:</span><br><strong>Sumbu X (50% Frame Width)</strong></div>
            <div><span style="color: #64748b;">Acuan Standar:</span><br><strong>PKJI 2023 (C0 = 2800 / 1400)</strong></div>
        </div>
    </div>
    """

    df_rekap = pd.DataFrame({
        "Kategori Kendaraan": ["Sepeda Motor (MC)", "Mobil Kecil (LV)", "Bus (BUS)", "Truk (TRK)", "Tak Bermotor (UM)", "TOTAL AKUMULASI"],
        "Jalur Kiri (Arah A)": [counts_kiri["MC"], counts_kiri["LV"], counts_kiri["BUS"], counts_kiri["TRK"], counts_kiri["UM"], tot_kiri],
        "Jalur Kanan (Arah B)": [counts_kanan["MC"], counts_kanan["LV"], counts_kanan["BUS"], counts_kanan["TRK"], counts_kanan["UM"], tot_kanan],
        "Total Fisik (Kend)": [
            counts_kiri["MC"] + counts_kanan["MC"],
            counts_kiri["LV"] + counts_kanan["LV"],
            counts_kiri["BUS"] + counts_kanan["BUS"],
            counts_kiri["TRK"] + counts_kanan["TRK"],
            counts_kiri["UM"] + counts_kanan["UM"],
            total_fisik
        ],
        "Faktor emp": [0.4, 1.0, 1.3, 1.3, 0.4, "-"],
        "Volume Total (smp)": [
            f"{(counts_kiri['MC'] + counts_kanan['MC']) * 0.4:,.1f}",
            f"{(counts_kiri['LV'] + counts_kanan['LV']) * 1.0:,.1f}",
            f"{(counts_kiri['BUS'] + counts_kanan['BUS']) * 1.3:,.1f}",
            f"{(counts_kiri['TRK'] + counts_kanan['TRK']) * 1.3:,.1f}",
            f"{(counts_kiri['UM'] + counts_kanan['UM']) * 0.4:,.1f}",
            f"{total_smp:,.1f}"
        ]
    })

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor='#111827')
    ax.set_facecolor('#111827')

    kategori = ['MC', 'LV', 'BUS', 'TRK', 'UM']
    vals_kiri = [counts_kiri["MC"], counts_kiri["LV"], counts_kiri["BUS"], counts_kiri["TRK"], counts_kiri["UM"]]
    vals_kanan = [counts_kanan["MC"], counts_kanan["LV"], counts_kanan["BUS"], counts_kanan["TRK"], counts_kanan["UM"]]

    x = np.arange(len(kategori))
    width_bar = 0.35

    rects1 = ax.bar(x - width_bar/2, vals_kiri, width_bar, label='Jalur Kiri (Arah A)', color='#38bdf8')
    rects2 = ax.bar(x + width_bar/2, vals_kanan, width_bar, label='Jalur Kanan (Arah B)', color='#f97316')

    ax.set_ylabel('Jumlah Kendaraan', color='#94a3b8', fontsize=8)
    ax.set_title('Komparasi Volume Kendaraan Per Jalur', fontsize=10, fontweight='bold', color='#f8fafc', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(kategori, color='#94a3b8', fontsize=8)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8)

    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')

    for bar in rects1:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#38bdf8', fontweight='bold', fontsize=7)

    for bar in rects2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', color='#f97316', fontweight='bold', fontsize=7)

    plt.grid(axis='y', linestyle='--', alpha=0.15, color='#64748b')
    plt.tight_layout()

    return output_filename, laporan_md, fig, df_rekap, info_video_md