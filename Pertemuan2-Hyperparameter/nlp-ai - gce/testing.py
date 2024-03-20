from iteung import reply # Import modul reply dari paket iteung untuk menggunakan fungsi botReply
# # import pandas as pd # Import modul pandas untuk manipulasi data DataFrame

while True:
    # Memulai loop tak terbatas untuk berinteraksi dengan bot
    message = input("Kamu: ")
    # Meminta input pesan dari pengguna
    return_message, status, _, _ = reply.botReply(message)
    # Mendapatkan balasan dari bot menggunakan fungsi botReply dari modul reply
    print(f"ITeung: {return_message}")
    # Menampilkan balasan dari bot


# data = reply.get_val_data()
# val_q = []
# val_a = []
# bot_a = []
# for i, v in data.iterrows():
#     val_question = v['0']
#     val_answer = v['1'].replace('<START>', '').replace('<END>', '').strip()
#     val_q.append(val_question)
#     val_a.append(val_answer)
#     return_message, _ = reply.botReply(val_question)
#     bot_a.append(return_message)

# bot_test_dataframe = pd.DataFrame(
#     {
#         'question': val_q,
#         'answer': val_a,
#         'bot': bot_a
#     }
# )

# bot_test_dataframe.to_csv('output_dir2/test_result.csv', index=False)


# import pandas as pd
# from iteung import reply

# # Daftar pertanyaan
# listPertanyaan = [
#     "Hai",
#     "Apa kabar?",
#     "Siapa nama kamu?",
#     "kamu sehat",
#     "kamu suka makan apa?",
#     "udah makan?",
#     "kamu suka makanan apa?",
#     "kamu suka minuman apa?",
# ]

# # Inisialisasi list jawaban dan akurasi
# listJawaban = []
# listAkurasi = []

# # Mendapatkan jawaban dan akurasi untuk setiap pertanyaan
# for message in listPertanyaan:
#     return_message, status , dec_outputs, akurasi= reply.botReply(message)
#     listJawaban.append(return_message)
#     listAkurasi.append(akurasi)

# # Membuat DataFrame menggunakan pandas
# df = pd.DataFrame({
#     'Pertanyaan': listPertanyaan,
#     'Jawaban': listJawaban,
#     'Akurasi': listAkurasi
# })

# # Menyimpan DataFrame ke dalam file Excel
# df.to_excel('hasil_bot.xlsx', index=False)


# from iteung import reply
# import pandas as pd

# data = reply.get_val_data()
# val_q = []
# val_a = []
# bot_a = []
# for i, v in data.iterrows():
#     val_question = v['0']
#     val_answer = v['1'].replace('<START>', '').replace('<END>', '').strip()
#     val_q.append(val_question)
#     val_a.append(val_answer)
#     # Mendapatkan balasan dari bot untuk setiap pertanyaan validasi
#     return_message,_,_,kecocokan = reply.botReply(val_question)
#     bot_a.append(return_message)

# Membuat DataFrame yang berisi pertanyaan validasi, jawaban asli, dan jawaban bot
# serta tingkat kecocokan antara jawaban bot dan jawaban asli
# bot_test_dataframe = pd.DataFrame(
#     {
#         'question': val_q,
#         'answer': val_a,
#         'bot': bot_a,
#         'kecocokan':kecocokan
#     }
# )

# Menyimpan DataFrame ke dalam file Excel
# bot_test_dataframe.to_excel('test_result.xlsx', index=False)


# from iteung import reply
# import pandas as pd

# data = reply.get_val_data()
# val_q = []
# val_a = []
# bot_a = []
# validity = []  # List untuk menyimpan validitas jawaban
# for i, v in data.iterrows():
#     val_question = v['0']
#     val_answer = v['1'].replace('<START>', '').replace('<END>', '').strip()
#     val_q.append(val_question)
#     val_a.append(val_answer)
#     # Mendapatkan balasan dari bot untuk setiap pertanyaan validasi
#     return_message,_,_,kecocokan = reply.botReply(val_question)
#     bot_a.append(return_message)
#     # Menentukan validitas jawaban
#     is_valid = 'valid' if return_message.strip() == val_answer else 'tidak valid'
#     validity.append(is_valid)

# # Membuat DataFrame yang berisi pertanyaan validasi, jawaban asli, jawaban bot,
# # serta tingkat kecocokan antara jawaban bot dan jawaban asli
# bot_test_dataframe = pd.DataFrame(
#     {
#         'question': val_q,
#         'answer': val_a,
#         'bot': bot_a,
#         'wordIndex': kecocokan,
#         'valid': validity
#     }
# )

# # Menyimpan DataFrame ke dalam file Excel
# bot_test_dataframe.to_excel('test_result3.xlsx', index=False)
