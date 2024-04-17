# Perbedaan membuat library dan file utama di rust

1. File Utama:

- File utama adalah tempat di mana Anda menulis kode yang ingin Anda jalankan sebagai program utama.
- File utama tempat di mana Anda mendefinisikan fungsi main(), yang akan dieksekusi saat program Anda dijalankan.
- Bayangkan file utama seperti "pintu masuk" ke aplikasi Anda. Jadi File utama adalah tempat pertama yang akan dijalankan ketika Anda memulai aplikasi Anda.

2. Library:

- Library adalah kumpulan kode yang dirancang untuk digunakan kembali di berbagai proyek.
- Library adalah tempat di mana Anda mendefinisikan berbagai fungsi, struct, atau fitur lain yang ingin Anda bagi dengan proyek lain.
- Bayangkan library seperti "kotak alat" yang berisi berbagai alat yang dapat Anda gunakan di berbagai proyek Anda.
- Library tidak memiliki fungsi main() karena itu tidak dimaksudkan untuk dijalankan sebagai program mandiri. Sebaliknya, kode dalam library tersebut dimaksudkan untuk digunakan oleh program utama atau proyek lain.

Dalam konteks pengembangan perangkat lunak, membuat file utama berarti membuat program utama yang akan dijalankan oleh pengguna akhir. Sedangkan membuat library berarti membuat kumpulan kode yang dapat digunakan kembali dalam proyek-proyek lain untuk memudahkan pengembangan perangkat lunak yang lebih besar dan lebih kompleks.
