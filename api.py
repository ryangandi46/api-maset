# app.py — API clustering (model baru, 2 endpoint)

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)

# ---------------------------
# Konfigurasi & konstanta
# ---------------------------
FEATURES = ['kalori', 'karbohidrat', 'protein', 'harga']   # urutan fitur numerik
DATA_PATH = 'foodeat_cleaned.csv'                          # opsional, untuk hitung mean

# ---------------------------
# Muat model & siapkan data
# ---------------------------
try:
    bundle = joblib.load('model_rekomendasi_makanan.pkl')
    kmeans = bundle['kmeans']
    scaler = bundle['scaler']
    # Validasi urutan fitur jika disimpan
    if 'features' in bundle:
        if list(bundle['features']) != FEATURES:
            print(f"⚠️  Peringatan: Urutan FEATURES di bundle {bundle['features']} "
                  f"berbeda dengan API {FEATURES}. Pastikan konsisten!")
    print("✅ Model (kmeans, scaler) berhasil dimuat.")
except FileNotFoundError:
    raise SystemExit("❌ 'model_rekomendasi_makanan.pkl' tidak ditemukan.")
except KeyError as e:
    raise SystemExit(f"❌ Kunci '{e}' tidak ada dalam bundle model.")

# Hitung rata-rata nutrisi dari data (untuk profil user di /get-user-cluster)
try:
    df_ref = pd.read_csv(DATA_PATH)
    for col in FEATURES:
        if col not in df_ref.columns:
            raise KeyError(f"Kolom '{col}' tidak ada di {DATA_PATH}")
    means = df_ref[FEATURES].mean(numeric_only=True)
    print("✅ Rata-rata nutrisi dari data referensi berhasil dihitung.")
except Exception as e:
    print(f"⚠️  Tidak bisa memuat '{DATA_PATH}' atau kolom tidak lengkap ({e}). "
          "Pakai fallback mean sederhana.")
    means = pd.Series(
        {'kalori': 350.0, 'karbohidrat': 30.0, 'protein': 20.0, 'harga': 25000.0}
    )

# ---------------------------
# Utilitas kecil
# ---------------------------
def derive_level_harga(harga: float) -> str:
    if harga < 18000:
        return 'Normal'
    elif harga <= 35000:
        return 'Mahal'
    return 'Premium'

def to_float(v, name):
    try:
        return float(v)
    except Exception:
        raise ValueError(f"'{name}' harus numerik.")

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return "API Clustering Makanan (model baru, 2 endpoint) aktif."

@app.route('/predict-food-cluster', methods=['POST'])
def predict_food_cluster():
    """
    Input JSON: { kalori, karbohidrat, protein, harga }
    Output: { cluster, level_harga }
    """
    try:
        data = request.get_json(force=True)
        req_keys = ['kalori', 'karbohidrat', 'protein', 'harga']
        if not all(k in data for k in req_keys):
            return jsonify({'error': f'Key wajib: {req_keys}'}), 400

        kal = to_float(data['kalori'], 'kalori')
        kar = to_float(data['karbohidrat'], 'karbohidrat')
        pro = to_float(data['protein'], 'protein')
        har = to_float(data['harga'], 'harga')

        X = [[kal, kar, pro, har]]
        Xs = scaler.transform(X)
        cluster_id = int(kmeans.predict(Xs)[0])
        level_harga = derive_level_harga(har)

        return jsonify({'cluster': cluster_id, 'level_harga': level_harga})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/get-user-cluster', methods=['POST'])
def get_user_cluster():
    """
    Input JSON: { budget, tipe_diet }
      - 'tipe_diet' diterima agar bisa dipakai di layer aplikasi (Laravel), TIDAK dipakai ke model.
    Output: { cluster, level_harga }
    """
    try:
        data = request.get_json(force=True)
        if 'budget' not in data:
            return jsonify({'error': 'Key "budget" wajib ada.'}), 400

        budget = to_float(data['budget'], 'budget')

        # Profil user sederhana: pakai mean nutrisi dari data + budget user
        in_vec = [[
            float(data.get('kalori', means['kalori'])),
            float(data.get('karbohidrat', means['karbohidrat'])),
            float(data.get('protein', means['protein'])),
            budget
        ]]
        Xs = scaler.transform(in_vec)
        cluster_id = int(kmeans.predict(Xs)[0])
        level_harga = derive_level_harga(budget)

        return jsonify({'cluster': cluster_id, 'level_harga': level_harga})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    # host='0.0.0.0' agar bisa diakses lintas host/container
    app.run(host='0.0.0.0', port=5000, debug=True)
