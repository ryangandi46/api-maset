# Impor library yang diperlukan
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Muat bundle model dari satu file saat aplikasi pertama kali dijalankan ---
try:
    # Muat bundle yang berisi semua komponen model
    model_bundle = joblib.load('model_rekomendasi_makanan.pkl')

    # Ekstrak setiap komponen dari bundle ke variabel masing-masing
    kmeans = model_bundle['kmeans']
    scaler = model_bundle['scaler']
    le_diet = model_bundle['label_encoder_diet']
    le_level = model_bundle['label_encoder_level']
    
    # Catatan: Akan lebih baik jika nilai rata-rata ini juga disimpan di dalam bundle
    # Untuk sekarang, kita definisikan secara manual sebagai placeholder
    # Anda bisa mendapatkan nilai ini dari notebook saat training: df[['kalori', 'karbohidrat', 'protein']].mean()
    avg_features = {
        'kalori': 350.5,
        'karbohidrat': 30.2,
        'protein': 18.9
    }
    
    print("✅ Model bundle (kmeans, scaler, encoders) berhasil dimuat.")

except FileNotFoundError:
    print("❌ FATAL ERROR: File 'model_rekomendasi_makanan.pkl' tidak ditemukan.")
    print("Pastikan file model berada di folder yang sama dengan app.py")
    exit() # Hentikan aplikasi jika file model tidak ada
except KeyError as e:
    print(f"❌ FATAL ERROR: Komponen '{e}' tidak ditemukan di dalam bundle model.")
    exit()
# -------------------------------------------------------------------------


@app.route('/')
def index():
    return "API Model Rekomendasi Makanan Aktif."


# Endpoint #1: Mendapatkan CLUSTER REKOMENDASI untuk PENGGUNA
@app.route('/get-user-cluster', methods=['POST'])
def get_user_cluster():
    """
    Menerima budget dan tipe diet, lalu mengembalikan nomor cluster yang paling cocok.
    """
    try:
        # Ambil data JSON yang dikirim dari client (Laravel)
        data = request.get_json()

        # Validasi input
        if not data or 'budget' not in data or 'tipe_diet' not in data:
            return jsonify({'error': 'Input tidak valid! Pastikan ada key "budget" dan "tipe_diet".'}), 400

        budget_user = int(data['budget'])
        tipe_diet_user = data['tipe_diet']

        # --- Logika untuk membuat profil pengguna ---
        # 1. Ubah input teks menjadi angka (encoded)
        diet_encoded = le_diet.transform([tipe_diet_user])[0]

        # 2. Tentukan level harga dari budget
        if budget_user < 18000:
            level_harga = 'Normal'
        elif 18000 <= budget_user <= 35000:
            level_harga = 'Mahal'
        else:
            level_harga = 'Premium'
        
        level_encoded = le_level.transform([level_harga])[0]

        # 3. Buat array fitur untuk diprediksi
        # Gunakan nilai rata-rata dari data training sebagai placeholder
        input_user = [[
            avg_features['kalori'],
            avg_features['karbohidrat'],
            avg_features['protein'],
            budget_user,
            diet_encoded,
            level_encoded
        ]]

        # 4. Scaling input
        input_user_scaled = scaler.transform(input_user)

        # 5. Prediksi cluster
        predicted_cluster = kmeans.predict(input_user_scaled)[0]

        # Kembalikan hasil dalam format JSON
        return jsonify({'cluster': int(predicted_cluster)})

    except ValueError:
        return jsonify({'error': f"Tipe diet '{tipe_diet_user}' tidak dikenali oleh model."}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan internal: {str(e)}'}), 500


# Endpoint #2: Memprediksi CLUSTER untuk MAKANAN BARU
@app.route('/predict-food-cluster', methods=['POST'])
def predict_food_cluster():
    """
    Menerima fitur-fitur dari satu makanan, lalu mengembalikan nomor cluster untuk makanan tsb.
    """
    try:
        data = request.get_json()

        # Validasi input
        required_keys = ['kalori', 'karbohidrat', 'protein', 'harga', 'tipe_diet']
        if not all(key in data for key in required_keys):
            return jsonify({'error': f'Input tidak valid! Key yang dibutuhkan: {required_keys}.'}), 400

        # Tentukan level harga berdasarkan harga makanan
        if data['harga'] < 18000: level_harga = 'Normal'
        elif data['harga'] <= 35000: level_harga = 'Mahal'
        else: level_harga = 'Premium'
            
        # Ubah input teks menjadi angka
        diet_encoded = le_diet.transform([data['tipe_diet']])[0]
        level_encoded = le_level.transform([level_harga])[0]

        # Buat array fitur dari data makanan baru
        food_features = [[
            data['kalori'],
            data['karbohidrat'],
            data['protein'],
            data['harga'],
            diet_encoded,
            level_encoded
        ]]

        # Scaling fitur
        food_features_scaled = scaler.transform(food_features)

        # Prediksi cluster
        predicted_cluster = kmeans.predict(food_features_scaled)[0]

        return jsonify({'cluster': int(predicted_cluster)})

    except ValueError:
         return jsonify({'error': f"Tipe diet '{data.get('tipe_diet')}' tidak dikenali oleh model."}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan internal: {str(e)}'}), 500


if __name__ == '__main__':
    # Jalankan aplikasi. Port 5000 adalah default.
    # host='0.0.0.0' agar bisa diakses dari luar container Docker (jika digunakan)
    app.run(host='0.0.0.0', port=5000, debug=True)