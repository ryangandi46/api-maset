from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# --- Muat Model dan Data ---
try:
    pipeline = joblib.load('rekomendasi_pipeline.pkl')
    df_clustered = pd.read_csv('data_makanan_clustered.csv')
    print("✅ Model dan data berhasil dimuat.")
except FileNotFoundError:
    print("❌ Pastikan file 'rekomendasi_pipeline.pkl' dan 'data_makanan_clustered.csv' ada.")
    pipeline = None
    df_clustered = pd.DataFrame()

# --- Definisikan variabel nama kolom ---
# Nama kolom dari file CSV Anda 
HARGA_COL = 'Harga (Rp)'
DIET_COL = 'Tipe Diet'

# --- Endpoint untuk Rekomendasi ---
@app.route('/rekomendasi', methods=['POST'])
def get_rekomendasi():
    if pipeline is None or df_clustered.empty:
        return jsonify({'error': 'Model atau data tidak tersedia'}), 500

    data = request.get_json()
    if not data or 'budget' not in data or 'preferensi_diet' not in data:
        return jsonify({'error': 'Input tidak lengkap. Butuh "budget" dan "preferensi_diet".'}), 400

    try:
        budget = int(data['budget'])
        preferensi_diet = data['preferensi_diet']
    except (ValueError, TypeError):
        return jsonify({'error': 'Tipe data input salah.'}), 400

    input_pengguna = pd.DataFrame({
        HARGA_COL: [budget],
        DIET_COL: [preferensi_diet]
    })

    try:
        predicted_cluster = pipeline.predict(input_pengguna)[0]
    except Exception as e:
        return jsonify({'error': f"Gagal melakukan prediksi. Detail: {e}"}), 400

    # ✅ Tambahkan filter diet juga
    rekomendasi_df = df_clustered[
        (df_clustered['Cluster'] == predicted_cluster) &
        (df_clustered[HARGA_COL] <= budget) &
        (df_clustered[DIET_COL] == preferensi_diet)
    ]

    if rekomendasi_df.empty:
        saran_df = df_clustered[
            (df_clustered['Cluster'] == predicted_cluster) &
            (df_clustered[DIET_COL] == preferensi_diet)
        ]
        hasil_saran = saran_df.to_dict(orient='records')
        return jsonify({
            'message': 'Tidak ada menu sesuai budget, berikut saran dari diet dan cluster yang sama.',
            'rekomendasi': [],
            'saran': hasil_saran
        })

    hasil_rekomendasi = rekomendasi_df.to_dict(orient='records')
    return jsonify({'rekomendasi': hasil_rekomendasi})


# --- Jalankan API Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)