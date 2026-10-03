# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | **Bahasa Indonesia** | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

**Perpustakaan video lokal khusus Windows.** Jelajahi dan putar video, lalu temukan adegan berdasarkan orang yang muncul. Saran pencocokan wajah dapat Anda periksa dan koreksi di aplikasi.

## Yang dapat dilakukan

- Menjelajahi dan memutar video di folder pilihan
- Mencari kandidat orang, video terkait, dan waktu kemunculan wajah
- Menjeda dan melanjutkan analisis, serta menetapkan, memisahkan, menggabungkan, atau mengecualikan hasil wajah secara manual
- Menyimpan indeks di PC tanpa memindahkan video asli

Pengelompokan otomatis adalah saran berdasarkan kemiripan visual dan dapat keliru. Periksa dan koreksi hasilnya. Akurasi pencocokan wajah belum diukur. Pemrosesan berlangsung lokal; penyiapan pertama mengunduh perangkat lunak dan model yang diperlukan.

## Penyiapan Windows

Perlu Windows 64-bit, Python 3.12, dan PowerShell. Jalankan dari folder proyek:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

Penyiapan standar menggunakan CPU. Untuk akselerasi NVIDIA CUDA, jalankan `.\Scripts\setup_windows.cmd -Gpu`. TensorRT harus dipasang terpisah; lihat [petunjuk penyiapan Windows](docs/WINDOWS_SETUP.md). Implementasi Windows belum diuji saat dijalankan: tidak ada tes, peluncuran aplikasi, atau analisis video nyata.

## Privasi dan model

Indeks dan gambar yang dibuat disimpan di data aplikasi lokal Windows. Aplikasi tidak mengunggah video atau data wajah. Penyiapan mengunduh model FACE01 wajah Jepang dan MediaPipe Face Landmarker yang versinya sudah ditetapkan, lalu memeriksa SHA-256. Model FACE01 memiliki ketentuan tersendiri; bacalah sebelum digunakan. [Lisensi MIT](LICENSE) proyek tidak mencakup model atau dependensi pihak ketiga.

## Status proyek

Implementasi Windows belum diuji saat dijalankan: tidak ada tes, peluncuran aplikasi, atau analisis video nyata. Rilis khusus Windows ini tidak mendukung aplikasi dan penyiapan macOS sebelumnya.
