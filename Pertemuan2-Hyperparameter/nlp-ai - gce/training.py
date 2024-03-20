
# Mengimpor library yang diperlukan
import json  # Untuk bekerja dengan data JSON
import os  # Untuk berinteraksi dengan sistem operasi
import pickle  # Untuk menyimpan dan memuat objek Python

import pandas as pd  # Untuk bekerja dengan data tabular dalam bentuk DataFrame
import tensorflow as tf  # Library pembelajaran mesin yang digunakan sebagai backend untuk Keras
from keras import Input, Model  # Untuk membangun model neural network dengan Keras
from keras.activations import softmax  # Fungsi aktivasi softmax
from keras.callbacks import ModelCheckpoint, TensorBoard  # Callbacks untuk menyimpan model dan log
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate  # Jenis-jenis layer dalam model neural network
from keras.optimizers import RMSprop  # Optimizer untuk melatih model
from keras.utils import to_categorical  # Untuk melakukan one-hot encoding pada data output
from keras.preprocessing.sequence import pad_sequences  # Untuk melakukan padding pada sequence
from keras.preprocessing.text import Tokenizer  # Untuk tokenisasi teks

# Konfigurasi sesi TensorFlow
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))  # Membuat sesi TensorFlow

# Path untuk menyimpan model dan log
path = "output_dir2/"  # Path untuk menyimpan model dan log
try:
    os.makedirs(path)  # Membuat direktori jika belum ada
except:
    pass

# Membaca dataset pertanyaan dan jawaban yang telah dibersihkan
dataset = pd.read_csv('./dataset/clean_qa2.txt', delimiter="|", header=None, lineterminator='\n')  # Membaca dataset dari file CSV

# Memisahkan dataset menjadi data validasi dan data pelatihan
dataset_val = dataset.iloc[400:].to_csv('output_dir2/val.csv')  # Memisahkan data validasi dan menyimpannya sebagai file CSV
dataset_train = dataset.iloc[:400]  # Data pelatihan

# Memisahkan pertanyaan dan jawaban dari dataset pelatihan dan uji
questions_train = dataset_train.iloc[:, 0].values.tolist()  # Pertanyaan dari data pelatihan
answers_train = dataset_train.iloc[:, 1].values.tolist()  # Jawaban dari data pelatihan
questions_test = dataset_train.iloc[:, 0].values.tolist()  # Pertanyaan dari data uji
answers_test = dataset_train.iloc[:, 1].values.tolist()  # Jawaban dari data uji

# Fungsi untuk menyimpan tokenizer ke file pickle
def save_tokenizer(tokenizer):
    with open('output_dir2/tokenizer.pickle', 'wb') as handle:  # Menyimpan tokenizer sebagai file pickle
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Fungsi untuk menyimpan konfigurasi ke file JSON
def save_config(key, value):
    data = {}  # Dictionary untuk data konfigurasi
    if os.path.exists(path + 'config.json'):  # Memeriksa apakah file konfigurasi sudah ada
        with open(path + 'config.json') as json_file:  # Membuka file konfigurasi
            data = json.load(json_file)  # Memuat data konfigurasi
    data[key] = value  # Menambahkan konfigurasi baru ke dictionary
    with open(path + 'config.json', 'w') as outfile:  # Menyimpan data konfigurasi ke file JSON
        json.dump(data, outfile)

# Inisialisasi Tokenizer
target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'  # Regex untuk karakter target yang akan dihapus
tokenizer = Tokenizer(filters=target_regex, lower=True)  # Membuat objek Tokenizer
tokenizer.fit_on_texts(questions_train + answers_train + questions_test + answers_test)  # Menggunakan data pelatihan untuk tokenisasi
save_tokenizer(tokenizer)  # Menyimpan tokenizer ke file pickle

# Menghitung ukuran vocab
VOCAB_SIZE = len(tokenizer.word_index) + 1  # Menghitung ukuran vokabular
save_config('VOCAB_SIZE', VOCAB_SIZE)  # Menyimpan ukuran vokabular ke dalam konfigurasi
print('Ukuran vocab : {}'.format(VOCAB_SIZE))  # Menampilkan ukuran vokabular

# Tokenisasi data pertanyaan dan jawaban pelatihan
tokenized_questions_train = tokenizer.texts_to_sequences(questions_train)  # Tokenisasi pertanyaan pelatihan
maxlen_questions_train = max([len(x) for x in tokenized_questions_train])  # Menentukan panjang maksimum pertanyaan pelatihan
save_config('maxlen_questions', maxlen_questions_train)  # Menyimpan panjang maksimum pertanyaan pelatihan
encoder_input_data_train = pad_sequences(tokenized_questions_train, maxlen=maxlen_questions_train, padding='post')  # Padding sequence pertanyaan pelatihan

# Tokenisasi data pertanyaan dan jawaban uji
tokenized_questions_test = tokenizer.texts_to_sequences(questions_test)  # Tokenisasi pertanyaan uji
maxlen_questions_test = max([len(x) for x in tokenized_questions_test])  # Menentukan panjang maksimum pertanyaan uji
save_config('maxlen_questions', maxlen_questions_test)  # Menyimpan panjang maksimum pertanyaan uji
encoder_input_data_test = pad_sequences(tokenized_questions_test, maxlen=maxlen_questions_test, padding='post')  # Padding sequence pertanyaan uji

# Tokenisasi data jawaban pelatihan
tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)  # Tokenisasi jawaban pelatihan
maxlen_answers_train = max([len(x) for x in tokenized_answers_train])  # Menentukan panjang maksimum jawaban pelatihan
save_config('maxlen_answers', maxlen_answers_train)  # Menyimpan panjang maksimum jawaban pelatihan
decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')  # Padding sequence jawaban pelatihan

# Tokenisasi data jawaban uji
tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)  # Tokenisasi jawaban uji
maxlen_answers_test = max([len(x) for x in tokenized_answers_test])  # Menentukan panjang maksimum jawaban uji
save_config('maxlen_answers', maxlen_answers_test)  # Menyimpan panjang maksimum jawaban uji
decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')  # Padding sequence jawaban uji

# Membuat data output pelatihan dengan one-hot encoding
for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:]  # Menghilangkan token 'start'
padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')  # Padding sequence jawaban pelatihan
decoder_output_data_train = to_categorical(padded_answers_train, num_classes=VOCAB_SIZE)  # Melakukan one-hot encoding

# Membuat data output uji dengan one-hot encoding
for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]  # Menghilangkan token 'start'
padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')  # Padding sequence jawaban uji
decoder_output_data_test = to_categorical(padded_answers_test, num_classes=VOCAB_SIZE)  # Melakukan one-hot encoding

# Inisialisasi input encoder
enc_inp = Input(shape=(None,))  # Mendefinisikan input encoder
enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inp)  # Membuat layer embedding
enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)  # Mendefinisikan layer LSTM dengan keadaan maju dan mundur
state_h = Concatenate()([forward_h, backward_h])  # Menggabungkan keadaan maju dan mundur untuk keadaan tersembunyi
state_c = Concatenate()([forward_c, backward_c])  # Menggabungkan keadaan maju dan mundur untuk sel memori
enc_states = [state_h, state_c]  # Mendefinisikan keadaan encoder

# Inisialisasi input decoder
dec_inp = Input(shape=(None,))  # Mendefinisikan input decoder
dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inp)  # Membuat layer embedding
dec_lstm = LSTM(256 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)  # Mendefinisikan layer LSTM dengan keadaan maju dan mundur
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)  # Mendefinisikan layer LSTM decoder
dec_dense = Dense(VOCAB_SIZE, activation=softmax)  # Membuat layer dense dengan fungsi aktivasi softmax
output = dec_dense(dec_outputs)  # Output dari model

# Konfigurasi callback untuk tensorboard dan checkpoint model
logdir = os.path.join(path, "logs")  # Menentukan direktori log
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)  # Callback TensorBoard
checkpoint = ModelCheckpoint(os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),  # Callback ModelCheckpoint
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='auto', period=100)

# Inisialisasi dan kompilasi model
model = Model([enc_inp, dec_inp], output)  # Mendefinisikan model
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])  # Kompilasi model dengan optimizer dan fungsi loss

# Menampilkan ringkasan model
model.summary()

# Pelatihan model
batch_size = 10  # Ukuran batch
epochs = 1000  # Jumlah epoch
model.fit([encoder_input_data_train, decoder_input_data_train],  # Melatih model
          decoder_output_data_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),  # Data validasi
          callbacks=[tensorboard_callback, checkpoint])  # Callbacks

# Menyimpan model
model.save(os.path.join(path, 'model-' + path.replace("/", "") + '.h5'))  # Menyimpan model
