import json
import os
import pickle
import random
import re
import nltk

import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate
from keras_preprocessing.sequence import pad_sequences


# Fungsi untuk mengatur konfigurasi awal
def setConfig(file_name):
    factory = StemmerFactory()  # Membuat objek factory dari StemmerFactory untuk penggunaan stemmer
    stemmer = factory.create_stemmer()  # Membuat objek stemmer dari factory
    punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))  # Regex untuk karakter punctuation
    unknowns = ["gak paham", "kurang ngerti", "I don't know", "apaan cuyyy"]  # Daftar jawaban jika model tidak mengerti
    path = file_name + "/"  # Path relatif untuk file
    return factory, stemmer, punct_re_escape, unknowns, file_name, path


# Fungsi untuk memuat konfigurasi dari file
def load_config(path, config_path):
    data = {}
    if os.path.exists(path + config_path):  # Memeriksa apakah file konfigurasi ada
        with open(path + config_path) as json_file:  # Membuka file konfigurasi
            data = json.load(json_file)  # Memuat data konfigurasi dari file JSON
    return data


# Fungsi untuk memuat tokenizer dari file
def load_tokenizer(path, tokenizer_path):
    tokenizer = None
    if os.path.exists(path + tokenizer_path):  # Memeriksa apakah file tokenizer ada
        with open(path + tokenizer_path, 'rb') as handle:  # Membuka file tokenizer dalam mode binary
            tokenizer = pickle.load(handle)  # Memuat tokenizer dari file pickle
    return tokenizer


# Fungsi untuk mengatur parameter awal
def setParams(path, slang_path, config_path, tokenizer_path):
    list_indonesia_slang = pd.read_csv(path + slang_path, header=None).to_numpy()  # Memuat daftar slang kata
    config = load_config(path, config_path)  # Memuat konfigurasi
    VOCAB_SIZE = config['VOCAB_SIZE']  # Ukuran vokabular
    maxlen_questions = config['maxlen_questions']  # Panjang maksimum pertanyaan
    maxlen_answers = config['maxlen_answers']  # Panjang maksimum jawaban
    tokenizer = load_tokenizer(path, tokenizer_path)  # Memuat tokenizer
    return list_indonesia_slang, VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer


# Fungsi untuk memeriksa kata normal atau slang
def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)  # Memeriksa apakah kata adalah slang
    if slang_result:
        return slang_result
    return word_input


# Fungsi untuk normalisasi kalimat
def normalize_sentence(sentence):
    sentence = punct_re_escape.sub('', sentence.lower())  # Menghapus punctuation dan merubah menjadi lowercase
    sentence = sentence.replace('iiteung', '').replace('\n', '')  # Menghapus kata tertentu
    sentence = sentence.replace('iteung', '')  # Menghapus kata tertentu
    sentence = sentence.replace('teung', '')  # Menghapus kata tertentu
    sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)  # Menghapus tawa
    sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)  # Menghapus ekspresi tertentu
    sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)  # Menghapus ekspresi tertentu
    sentence = ' '.join(sentence.split())  # Menghapus spasi berlebih
    if sentence:
        sentence = sentence.strip().split(" ")  # Memisahkan kata
        normal_sentence = " "
        for word in sentence:
            normalize_word = check_normal_word(word)  # Memeriksa kata normal atau slang
            root_sentence = stemmer.stem(normalize_word)  # Melakukan stemming
            normal_sentence += root_sentence + " "  # Menggabungkan kata yang sudah distem
        return punct_re_escape.sub('', normal_sentence)  # Menghapus punctuation
    return sentence


# Fungsi untuk mengubah string menjadi token
def str_to_tokens(sentence, tokenizer, maxlen_questions):
    words = sentence.lower().split()  # Memisahkan kata dan mengubah menjadi lowercase
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')  # Mengambil indeks kata dari tokenizer
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')  # Padding tokens dengan maxlen


# Fungsi untuk mengatur encoder-decoder model
def setEncoderDecoder(VOCAB_SIZE):
    enc_inputs = Input(shape=(None,))
    enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inputs)
    _, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    enc_states = [state_h, state_c]

    dec_inputs = Input(shape=(None,))
    dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inputs)
    dec_lstm = LSTM(256 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)

    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

    dec_dense = Dense(VOCAB_SIZE, activation=softmax)
    output = dec_dense(dec_outputs)

    return dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, output


# Fungsi untuk membuat model inference
def make_inference_models(dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states):
    dec_state_input_h = Input(shape=(256 * 2,))
    dec_state_input_c = Input(shape=(256 * 2,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_outputs, state_h, state_c = dec_lstm(dec_embedding, initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]

    dec_outputs = dec_dense(dec_outputs)

    dec_model = Model(inputs=[dec_inputs] + dec_states_inputs, outputs=[dec_outputs] + dec_states)
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)

    return enc_model, dec_model


# Fungsi untuk mengatur model
def setModel(enc_inputs, dec_inputs, output, dec_lstm, dec_embedding, dec_dense, enc_states, file_path):
    model = Model([enc_inputs, dec_inputs], output)  # Membuat model dengan input dan output
    model.load_weights(file_path)  # Memuat berat model dari file

    enc_model, dec_model = make_inference_models(dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states)
    return model, enc_model, dec_model


# Fungsi untuk melakukan percakapan
def chat(input_value, tokenizer, maxlen_answers, enc_model, dec_model):
    input_value = stemmer.stem(
        normalize_sentence(normalize_sentence(input_value))  # Normalisasi dan stemming input
    )

    states_values = enc_model.predict(
        str_to_tokens(input_value, tokenizer, maxlen_questions)  # Mengonversi input menjadi token
    )

    empty_target_seq = np.zeros((1, 1))  # Membuat matriks kosong untuk sequence target
    empty_target_seq[0, 0] = tokenizer.word_index['start']  # Mengatur indeks token awal

    stop_condition = False
    decoded_translation = ''
    status = "false"

    kecocokan = 0

    while not stop_condition:
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )  # Prediksi output dengan model decoder
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])  # Mengambil indeks kata dengan probabilitas tertinggi
        kecocokan = dec_outputs[0, -1, sampled_word_index]
        if dec_outputs[0, -1, sampled_word_index] < 0.1:  # Jika probabilitas rendah
            decoded_translation = unknowns[random.randint(0, (len(unknowns) - 1))]  # Menggunakan jawaban acak dari daftar
            break
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:  # Jika sampai di akhir atau panjang jawaban maksimum tercapai
            stop_condition = True  # Berhenti
            print(len(decoded_translation.split()));  # Menampilkan panjang jawaban

        empty_target_seq = np.zeros((1, 1))  # Membuat matriks kosong untuk sequence target
        empty_target_seq[0, 0] = sampled_word_index  # Mengatur indeks token yang dihasilkan sebagai input berikutnya
        states_values = [h, c]  # Menyimpan status untuk state selanjutnya
        status = "true"

    return decoded_translation.strip(), str(status).lower(), dec_outputs, kecocokan


# Fungsi untuk mendapatkan slang data
def get_val_data(dir='output_dir2'):
    data = pd.read_csv(dir+'/val.csv')  # Memuat data dari file CSV
    return data


# Fungsi untuk memeriksa kata dalam dictionary
def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)


# Fungsi untuk memberikan balasan bot
def botReply(message):
    return chat(message, tokenizer, maxlen_answers, enc_model, dec_model)


# Inisialisasi konfigurasi awal
factory, stemmer, punct_re_escape, unknowns, file_name, path = setConfig("output_dir2")

# Mendapatkan daftar slang, ukuran vokabular, panjang maksimum pertanyaan dan jawaban, dan tokenizer
list_indonesia_slang, VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer = setParams(path,
                                                                                          'daftar-slang-bahasa-indonesia.csv',
                                                                                          'config.json',
                                                                                          'tokenizer.pickle')

# Mengatur encoder-decoder
dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, output = setEncoderDecoder(VOCAB_SIZE)

# Mengatur model
model, enc_model, dec_model = setModel(enc_inputs,
                                       dec_inputs,
                                       output,
                                       dec_lstm,
                                       dec_embedding,
                                       dec_dense,
                                       enc_states,
                                       path + 'model-' + file_name + '.h5')

data_slang = {}  # Dictionary untuk slang data
for key, value in list_indonesia_slang:
    data_slang[key] = value



# def chat(input_value, tokenizer, maxlen_answers, enc_model, dec_model):
#     input_value = stemmer.stem(
#         normalize_sentence(normalize_sentence(input_value))
#     )

#     states_values = enc_model.predict(
#         str_to_tokens(input_value, tokenizer, maxlen_questions)
#     )

#     empty_target_seq = np.zeros((1, 1))
#     empty_target_seq[0, 0] = tokenizer.word_index['start']

#     stop_condition = False
#     decoded_translation = ''
#     status = "false"

#     while not stop_condition:
#         dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )

#         sampled_word_index = np.argmax(dec_outputs[0, -1, :])
#         if dec_outputs[0, -1, sampled_word_index] < 0.03:
#             decoded_translation = unknowns[random.randint(0, (len(unknowns) - 1))]
#             break
#         sampled_word = None
#         for word, index in tokenizer.word_index.items():
#             if sampled_word_index == index:
#                 if word != 'end':
#                     decoded_translation += ' {}'.format(word)
#                 sampled_word = word

#         if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
#             stop_condition = True

#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = sampled_word_index
#         states_values = [h, c]
#         status = "true"

#     return decoded_translation.strip(), str(status).lower()