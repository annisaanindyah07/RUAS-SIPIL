# ==============================================================================
# ANTARMUKA WEB RUAS-SIPIL (app.py) - STABLE DUAL LANE EDITION
# ==============================================================================

import gradio as gr
from engine import jalankan_analisis_ruas

custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}
.main-header h1 {
    color: #38bdf8 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.card-panel {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}
.gr-button-primary {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css, title="RUAS-SIPIL AI Dashboard") as demo:
    
    # Header Utama
    with gr.Row(elem_classes="main-header"):
        gr.Markdown(
            """
            # 🚦 RUAS-SIPIL
            **Sistem Analisis Lalu Lintas Berbasis AI & Evaluasi Kinerja Jalan (PKJI 2023)**
            """
        )

    # GRID UTAMA ATAS
    with gr.Row():
        # Kolom Kiri: Unggah File
        with gr.Column(scale=5, elem_classes="card-panel"):
            gr.Markdown("### 📹 **Unggah Berkas Video**")
            input_video = gr.File(
                label="Pilih File Video Lalu Lintas (.mp4 / .avi / .mov / .mkv)",
                file_count="single",
                file_types=[".mp4", ".avi", ".mov", ".mkv"]
            )
            btn_proses = gr.Button("🚀 Mulai Analisis Otomatis & Visualisasi Deteksi", variant="primary", size="lg")

        # Kolom Kanan: Video Hasil Deteksi
        with gr.Column(scale=7, elem_classes="card-panel"):
            gr.Markdown("### 🎥 **Hasil Rekaman Deteksi AI & Sensor Virtual**")
            output_video_processed = gr.Video(
                label="Pemutaran Video Hasil Deteksi (Kotak Warna-Warni)",
                interactive=False
            )

    # BARIS TENGAH: KARTU KPI STATISTIK
    output_laporan = gr.HTML(
        value="<div style='text-align: center; color: #64748b; padding: 15px;'>Unggah file video di atas lalu klik <strong>Mulai Analisis Otomatis</strong>.</div>"
    )

    # GRID UTAMA BAWAH
    with gr.Row():
        # Kolom Kiri Bawah: Tabs Grafik & Rekapitulasi
        with gr.Column(scale=7, elem_classes="card-panel"):
            with gr.Tabs():
                with gr.TabItem("📈 Grafik Komposisi Kendaraan"):
                    output_grafik = gr.Plot(label="Distribusi Kendaraan Per Jalur")
                
                with gr.TabItem("📋 Tabel Rekapitulasi (Format PKJI 2023)"):
                    output_tabel = gr.Dataframe(interactive=False)

        # Kolom Kanan Bawah: Panel Metadata AI
        with gr.Column(scale=5, elem_classes="card-panel"):
            output_info_video = gr.HTML()

    # Event Trigger
    btn_proses.click(
        fn=jalankan_analisis_ruas,
        inputs=[input_video],
        outputs=[
            output_video_processed, 
            output_laporan, 
            output_grafik, 
            output_tabel, 
            output_info_video
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)