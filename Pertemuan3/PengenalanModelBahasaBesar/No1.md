# Introduction to Large Language Models and the Transformer Architecture - Pradeep Menon

## Model Bahasa Besar/ Large Language Model (LLM)

- Dilatih pada sejumlah besar data teks. Hasilnya, mereka dapat menghasilkan teks yang koheren(berhubungan) dan lancar.

- GPT adalah contoh dari Model Bahasa Besar. Model-model ini disebut "besar" karena mereka memiliki miliaran parameter yang membentuk respons mereka.

- Premis dasar dari model bahasa adalah kemampuannya untuk memprediksi kata atau sub-kata berikutnya (disebut token) berdasarkan teks yang telah diamati sejauh ini.

- Arsitektur pembelajaran mendalam yang membuat proses ini lebih mirip manusia adalah arsitektur Transformer.

## The Transformer Architecture: The Building Block

- Pertama-tama, dalam pemrosesan bahasa alami, model pembelajaran mesin perlu mengubah kata-kata menjadi angka agar bisa dipahami. Ini dilakukan melalui "penyematan input", di mana kata-kata direpresentasikan sebagai angka untuk diproses oleh model.

- Kemudian, untuk memahami urutan kata dalam kalimat, kita menggunakan "pengkodean posisi". Ini memberi setiap kata angka yang mewakili posisinya dalam kalimat.

- Selanjutnya, ada bagian "encoder" dalam model seperti GPT. Ini menerjemahkan teks input ke dalam representasi numerik yang menangkap makna dan konteks.

- Prosesnya melibatkan "lapisan-lapisan perhatian diri" yang membantu model memahami konteks teks pada tingkat abstraksi yang berbeda.

- Ketika kita melatih model, kita memindahkan urutan output untuk memastikan decoder hanya melihat kata-kata sebelumnya. Ini membantu model dalam memprediksi kata-kata berikutnya dengan benar.

- Selama pelatihan, model juga belajar bagaimana mengubah output menjadi format numerik yang disebut "penyematan output". Ini melibatkan fungsi kerugian yang mengukur perbedaan antara prediksi model dan nilai target.

- Selama inferensi, model menggunakan representasi input dan output yang dikodekan untuk menghasilkan teks bahasa alami sebagai output.

- Terakhir, ada langkah penggunaan "lapisan linear" untuk memetakan output ke ruang dimensi yang sesuai dengan input aslinya. Ini diikuti oleh penggunaan fungsi softmax untuk menghasilkan distribusi probabilitas untuk setiap token output dalam kosakata, yang membantu model memilih token output dengan benar.

## The Concept of Attention Mechanism

- memungkinkan model fokus pada bagian yang berbeda dari urutan input saat membuat setiap token output.

- Ini memungkinkan model secara selektif fokus pada bagian yang berbeda dari urutan input alih-alih memperlakukan semuanya dengan cara yang sama.

- Ini dapat menangkap hubungan antara input yang jauh satu sama lain dalam urutan, yang berguna untuk tugas bahasa alami.

- Perlu lebih sedikit parameter untuk memodelkan dependensi jangka panjang karena hanya perlu memperhatikan input yang penting.

- Ini sangat bagus dalam menangani input dengan panjang yang berbeda karena dapat menyesuaikan perhatiannya berdasarkan panjang urutan.

## Kesimpulan

Model Bahasa Besar seperti GPT (Generative Pre-trained Transformer) merupakan teknologi yang telah dilatih dengan data teks yang sangat luas, memungkinkannya menghasilkan teks yang koheren dan lancar.

Dengan miliaran parameter, model ini mampu merespons dengan cara yang kompleks dan bervariasi, meningkatkan kemampuannya dalam memprediksi kata atau sub-kata berikutnya berdasarkan teks yang diamati.

Arsitektur Transformer adalah inovasi penting dalam pembelajaran mesin yang memungkinkan proses prediksi menjadi lebih efisien dan mirip dengan pemahaman manusia.

Fitur utama dari model ini termasuk:

- Penyematan input yang mengubah kata-kata menjadi angka, dan pengkodean posisi untuk memahami urutan kata dalam kalimat.
- Encoder yang menerjemahkan teks menjadi representasi numerik.
- Decoder yang menghasilkan teks berdasarkan representasi tersebut.
- Mekanisme perhatian yang memungkinkan fokus pada bagian penting dari input, meningkatkan kemampuan dalam memahami dan menghasilkan teks yang relevan.
- Kemampuan model untuk menyesuaikan pendekatannya tergantung pada panjang input, menjadikannya efisien dalam pengolahan informasi.

Secara keseluruhan, Model Bahasa Besar dengan arsitektur Transformer memberikan terobosan dalam pemrosesan bahasa alami, membuka pintu bagi kemampuan yang sangat maju dan intuitif dalam menghasilkan dan memahami teks. 🤖✨
