# car-plane-image-autoencoder
Dimensionality reduction dan rekonstruksi citra mobil dan pesawat menggunakan convolutional autoencoder, dengan perbandingan arsitektur dan hyperparameter tuning berdasarkan SSIM.

# Car & Plane Image Reconstruction with Autoencoders

Proyek akademik untuk mengeksplorasi **dimensionality reduction** dan **image reconstruction** menggunakan convolutional autoencoder pada citra mobil dan pesawat. Eksperimen membandingkan baseline, modifikasi arsitektur, dan manual hyperparameter tuning berdasarkan kualitas rekonstruksi gambar.

## Gambaran Proyek

Autoencoder mempelajari representasi ringkas dari gambar melalui encoder, kemudian membangun kembali gambar tersebut melalui decoder. Input berupa citra grayscale berukuran 28 × 28 piksel direpresentasikan dalam latent space berukuran 128 atau 256 dimensi.

Fokus proyek ini adalah rekonstruksi gambar, bukan klasifikasi mobil dan pesawat. Label kelas digunakan untuk menjaga proporsi kelas dalam pembagian dataset.

## Dataset

Dataset mobil dan pesawat disediakan sebagai bagian dari tugas akademik. Notebook merujuknya sebagai dataset **Car Plane version 2**; sumber publik aslinya belum dicantumkan.

| Kategori | Jumlah gambar |
|---|---:|
| Car | 8.012 |
| Plane | 8.012 |
| **Total** | **16.024** |

Folder train dan test bawaan digabung, kemudian dibagi ulang menggunakan stratified random split dengan `random_state=42`:

| Subset | Jumlah gambar | Proporsi sekitar |
|---|---:|---:|
| Training | 12.819 | 80% |
| Validation | 1.602 | 10% |
| Test | 1.603 | 10% |

Hasil eksperimen menggunakan pembagian ulang tersebut, bukan test set bawaan dataset.

## Alur Analisis

1. Memeriksa jumlah gambar per kelas dan file gambar yang rusak.
2. Mengubah gambar menjadi grayscale, melakukan resize ke 28 × 28 piksel, dan scaling nilai piksel ke rentang 0–1.
3. Membagi data menjadi training, validation, dan test set.
4. Melatih autoencoder dengan gambar input sebagai target rekonstruksi.
5. Membandingkan modifikasi arsitektur dan konfigurasi training.
6. Mengevaluasi rata-rata Structural Similarity Index (SSIM) pada test set dan memvisualisasikan gambar hasil rekonstruksi.

<img width="1160" height="366" alt="Screenshot 2026-09-28 at 18 54 34" src="https://github.com/user-attachments/assets/8e4101a8-f47b-453e-8964-ea4ad648bf21" />


## Arsitektur dan Eksperimen

Baseline menggunakan alur berikut:

```text
Input (28 × 28 × 1)
→ Conv2D (32 filters, ReLU)
→ MaxPooling2D → Flatten
→ Dense (128): latent representation
→ Dense (6272) → Reshape (14 × 14 × 32)
→ UpSampling2D
→ Conv2D (32 filters, ReLU)
→ Conv2D (1 filter, Sigmoid)
→ Reconstructed image (28 × 28 × 1)
```

Semua model menggunakan optimizer **Adam**, loss **MSE**, dan **MAE** sebagai metric selama training.

| Konfigurasi | Perubahan utama | Latent dim. | Batch size | Maks. epoch |
|---|---|---:|---:|---:|
| Baseline | Satu convolutional layer pada encoder | 128 | 32 | 30 |
| Modified 1 | Dua convolutional layer dengan 64 filters pada encoder; Batch Normalization | 128 | 32 | 50 |
| Modified 2 | Encoder dengan 128 filters, tambahan Batch Normalization, Dropout 0,2 | 128 | 32 | 50 |
| Tuning 1 | Konfigurasi Modified 2 dengan Dropout 0,3, batch size 16, dan patience 5 | 128 | 16 | 50 |
| Tuning 2 | Konfigurasi Tuning 1 dengan latent dimension 256 | 256 | 16 | 50 |

Modified 1 dan Modified 2 menggunakan Early Stopping dengan patience 10. Kedua eksperimen tuning menggunakan patience 5. Early Stopping memonitor validation loss dan mengembalikan bobot terbaik.

## Hasil Evaluasi

SSIM dihitung untuk setiap gambar pada test set, kemudian dirata-ratakan. Nilai yang lebih tinggi menunjukkan kemiripan struktur yang lebih baik antara gambar input dan hasil rekonstruksi.

| Model | Latent dimension | Mean test SSIM |
|---|---:|---:|
| Baseline | 128 | 0,7321 |
| Modified 1 | 128 | 0,7432 |
| Modified 2 | 128 | 0,8814 |
| **Tuning 1** | **128** | **0,8917** |
| Tuning 2 | 256 | 0,7110 |

Nilai di atas berasal dari output eksperimen yang tersimpan dalam notebook.

### Baseline Reconstruction

<img width="1160" height="468" alt="Screenshot 2026-09-28 at 18 55 34" src="https://github.com/user-attachments/assets/0fe8f817-c52b-43ca-9efa-410957378518" />

### Tuning 1 Reconstruction

<img width="1135" height="468" alt="Screenshot 2026-09-28 at 18 55 55" src="https://github.com/user-attachments/assets/789733d2-081a-4a3b-8acb-baad40e5c222" />

### Tuning 2 Reconstruction

<img width="1117" height="471" alt="Screenshot 2026-09-28 at 18 58 37" src="https://github.com/user-attachments/assets/2e4c2ce5-e226-4bd5-8ffa-7a7659e7a167" />

<img width="699" height="463" alt="Screenshot 2026-09-28 at 18 58 09" src="https://github.com/user-attachments/assets/7ea0952e-5e53-450d-812d-6c5f9735b9d4" />

### Hyperparameter Tuning 1
<img width="714" height="474" alt="Screenshot 2026-09-28 at 18 59 14" src="https://github.com/user-attachments/assets/8584daaf-c8bf-450c-bfc4-e98fd6268a74" />

training loss mengalami penurunan secara konsisten selama proses pelatihan. Validation loss juga menurun pada beberapa epoch awal, namun setelah mencapai nilai terbaik pada epoch ke-27 sebesar 0,00454, validation loss cenderung berfluktuasi dan tidak menunjukkan penurunan yang signifikan. Meskipun demikian, selisih antara training loss dan validation loss masih relatif kecil sehingga model masih mampu melakukan generalisasi dengan cukup baik dan tidak mengalami overfitting yang berlebihan.

Dibandingkan dengan Modified 2, perubahan hyperparameter berupa peningkatan Dropout dari 0,2 menjadi 0,3, pengurangan batch size dari 32 menjadi 16, serta penerapan Early Stopping dengan patience 5 menghasilkan nilai validation loss yang sedikit lebih rendah dan model mencapai performa terbaik lebih cepat, yaitu pada epoch ke-27. Namun, peningkatan kualitas rekonstruksi yang diperoleh relatif kecil dibandingkan Modified 2 sehingga perubahan hyperparameter ini belum memberikan peningkatan performa yang signifikan.

<img width="1137" height="475" alt="Screenshot 2026-09-28 at 18 59 29" src="https://github.com/user-attachments/assets/967c965b-d4c1-4576-8280-1ff87aa28850" />

Hyperparameter Tuning memiliki tingkat kemiripan yang tinggi dengan gambar asli. Bentuk utama objek, baik pada kelas Car maupun Plane, masih dapat dipertahankan dengan baik meskipun terdapat sedikit penghalusan (smoothing) pada beberapa bagian dan detail berukuran kecil.

Namun, pada Hyperparameter Tuning 1 masih terlihat sedikit efek smoothing sehingga beberapa detail halus tampak lebih kabur. Sebaliknya, hasil rekonstruksi Modified 2 cenderung mempertahankan kontur objek dengan lebih konsisten dan menghasilkan tampilan yang sedikit lebih tajam pada beberapa sampel. Meskipun perbedaan visualnya tidak terlalu signifikan, hasil rekonstruksi Modified 2 terlihat lebih stabil secara keseluruhan.

### Hyperparameter Tuning 2

<img width="707" height="472" alt="Screenshot 2026-09-28 at 19 00 31" src="https://github.com/user-attachments/assets/8f2685fd-098b-440e-827a-a26aa213926d" />

Grafik, training loss dan validation loss mengalami penurunan pada awal proses pelatihan, kemudian menurun secara bertahap hingga akhir epoch. Nilai validation loss terbaik diperoleh pada epoch ke-5, yaitu sebesar 0,00647. Setelah epoch tersebut, validation loss mulai berfluktuasi dan tidak menunjukkan perbaikan. Hal ini menunjukkan adanya kecenderungan model mulai mengalami overfitting ringan setelah epoch ke-5.



## Temuan Utama

- Tuning 1 menghasilkan mean test SSIM tertinggi dalam eksperimen yang ditampilkan, meningkat dari **0,7321 menjadi 0,8917**, atau sekitar **21,80% secara relatif** dibanding baseline. Angka ini merupakan peningkatan SSIM, bukan classification accuracy.
- Representasi 128 dimensi mampu mempertahankan struktur gambar lebih baik pada konfigurasi Tuning 1 dibanding baseline, meskipun keduanya menggunakan latent dimension yang sama.
- Menambah latent dimension menjadi 256 pada Tuning 2 tidak meningkatkan kualitas rekonstruksi dalam eksperimen ini; SSIM turun menjadi 0,7110.
- Perubahan kapasitas jaringan dan konfigurasi training perlu dievaluasi bersama. Eksperimen ini belum memisahkan pengaruh masing-masing perubahan melalui ablation study.

## Keterbatasan

- Evaluasi menggunakan gambar grayscale beresolusi 28 × 28, sehingga tidak mengukur kemampuan mempertahankan warna atau detail resolusi tinggi.
- Hasil berasal dari satu pembagian data dan run yang tersimpan, tanpa pengulangan beberapa random seed.
- Test SSIM dibandingkan pada beberapa konfigurasi. Validasi akhir pada holdout yang belum digunakan untuk pemilihan konfigurasi diperlukan untuk penilaian generalisasi lebih lanjut.
- Perbandingan waktu training, penggunaan memori, dan ukuran file hasil kompresi belum dilakukan. Pengurangan latent dimension tidak sama dengan penghematan ukuran file yang telah diukur.

## Tools

Python, TensorFlow/Keras, NumPy, pandas, OpenCV, Pillow, scikit-learn, scikit-image, Matplotlib, dan Google Colab.

## Dataset 
https://drive.google.com/drive/folders/1aouIW0R7PwOJoO4BZPpI-wRJdgqaVA6O?usp=sharing


