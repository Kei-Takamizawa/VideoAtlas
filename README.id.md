# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | **Bahasa Indonesia** | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

VideoAtlas adalah aplikasi lokal untuk Windows yang mengatur video dan membantu mencari adegan berdasarkan orang yang muncul. Format yang didukung: MP4, MOV, AVI, MKV, M4V, dan WebM. Video dikenali dari hash kontennya, sehingga mengganti nama file saja tidak memicu analisis ulang.

## Yang dapat dilakukan

- Mengumpulkan beberapa sampel wajah per video dan melacak orang di dalamnya
- Mencocokkan orang di video berbeda dengan hati-hati
- Meninjau saran lalu memilih “orang yang sama”, “orang berbeda”, atau “nanti” (tombol S, D, L)
- Mengubah nama, menggabungkan atau memisahkan grup, mengecualikan orang, dan memilih gambar perwakilan
- Menyimpan pasangan yang ditinjau untuk mengevaluasi hasil pencocokan

Penggabungan otomatis lintas video nonaktif secara default. Koleksi video nyata kecil sudah dievaluasi, tetapi ambang saat ini melewatkan banyak kecocokan. Akurasi dengan masker asli belum diverifikasi. Nama berkas tidak digunakan untuk mengenali orang. Lihat hasil dan batas mode wajah atas pada [status implementasi](docs/IMPLEMENTATION_STATUS.md).

## Memulai di Windows

Perlu Windows 64-bit, Python 3.12, dan PowerShell. Jalankan dari folder proyek:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

Pengaturan standar memakai CPU dan mengunduh model resmi dengan verifikasi checksum. Antarmuka tersedia dalam bahasa Inggris dan Jepang. Baca [panduan penyiapan Windows](docs/WINDOWS_SETUP.md). Video asli tetap di tempatnya dan indeks tersimpan di PC. Aplikasi tidak mengunggah video atau data wajah. Model dan perangkat lunak memiliki ketentuan penggunaan masing-masing.
