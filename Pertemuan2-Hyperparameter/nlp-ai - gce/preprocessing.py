
"""ITeung

# Pra-pemrosesan
"""

# Mengimpor library yang diperlukan
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import io
import os
import re
import requests
import csv
import datetime
import numpy as np
import pandas as pd
import random
import pickle

print('# Membuat objek Stemmer dari factory')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

print('# Membuat regular expression untuk tanda baca')
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
print('# Kata-kata yang tidak dikenal')

print('# Membaca daftar slang Bahasa Indonesia dari file CSV')
list_indonesia_slang = pd.read_csv('./dataset/daftar-slang-bahasa-indonesia.csv', header=None).to_numpy()

print('# Membuat kamus untuk slang')
data_slang = {}
for key, value in list_indonesia_slang:
    data_slang[key] = value

print('# Fungsi untuk beralih secara dinamis antara kata slang dan kata normal')
def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)

print('# Fungsi untuk memeriksa apakah sebuah kata adalah kata normal atau slang')
def check_normal_word(word_input):
    print('# Memanggil fungsi dynamic_switcher dengan data_slang dan kata input sebagai argumen')
    slang_result = dynamic_switcher(data_slang, word_input)
    print('# Jika slang_result memiliki nilai, artinya kata input adalah kata slang, maka kembalikan hasil slang')
    if slang_result:
        return slang_result
    print('# Jika tidak, kembalikan kata input karena kata tersebut adalah kata normal')
    return word_input

print('# Fungsi untuk normalisasi kalimat')
def normalize_sentence(sentence):
  sentence = punct_re_escape.sub('', sentence.lower()) # Menghapus tanda baca dan mengonversi huruf kecil
  sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','') # Menghapus kata-kata khusus
  sentence = sentence.replace('teung', '') # Menghapus kata "teung"
  sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence) # Menghapus kata tertawa (e.g., "wkwk", "wk", "wkk")
  sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence) # Menghapus kata tertawa (e.g., "xi", "xiixi")
  sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence) # Menghapus kata tertawa (e.g., "hahahi", "hehe", "haha", "hihi")
  sentence = ' '.join(sentence.split()) # Menghapus spasi berlebih
  if sentence:
    sentence = sentence.strip().split(" ")
    normal_sentence = " "
    for word in sentence:
      normalize_word = check_normal_word(word)
      root_sentence = stemmer.stem(normalize_word)
      normal_sentence += root_sentence+" "
    return punct_re_escape.sub('',normal_sentence) # Menghapus tanda baca lagi
  return sentence

print('# Membaca dataset pertanyaan dan jawaban dari file CSV')
df = pd.read_csv('./dataset/qa_gce.csv', sep='|',usecols= ['question','answer'])
df.head()

print('# Membuat kamus untuk panjang pertanyaan')
question_length = {}
print('# Membuat kamus untuk panjang jawaban')
answer_length = {}

# Mengiterasi melalui baris-baris dataset
for index, row in df.iterrows():
  question = normalize_sentence(row['question']) # Normalisasi pertanyaan
  question = normalize_sentence(question)
  question = stemmer.stem(question) # Stemming pertanyaan

  # Memperbarui kamus panjang pertanyaan
  if question_length.get(len(question.split())):
    question_length[len(question.split())] += 1
  else:
    question_length[len(question.split())] = 1

  # Memperbarui kamus panjang jawaban
  if answer_length.get(len(str(row['answer']).split())):
    answer_length[len(str(row['answer']).split())] += 1
  else:
    answer_length[len(str(row['answer']).split())] = 1

# Menampilkan kamus panjang pertanyaan
question_length

# Menampilkan kamus panjang jawaban
answer_length

# Membuat list untuk panjang pertanyaan dan jumlah kalimat dengan panjang tersebut
val_question_length = list(question_length.values())
key_question_length = list(question_length.keys())
key_val_question_length = list(zip(key_question_length, val_question_length))
# Membuat DataFrame dari list tersebut
df_question_length = pd.DataFrame(key_val_question_length, columns=['length_data', 'total_sentences'])
df_question_length.sort_values(by=['length_data'], inplace=True)
df_question_length.describe()

# Membuat list untuk panjang jawaban dan jumlah kalimat dengan panjang tersebut
val_answer_length = list(answer_length.values())
key_answer_length = list(answer_length.keys())
key_val_answer_length = list(zip(key_answer_length, val_answer_length))
# Membuat DataFrame dari list tersebut
df_answer_length = pd.DataFrame(key_val_answer_length, columns=['length_data', 'total_sentences'])
df_answer_length.sort_values(by=['length_data'], inplace=True)
df_answer_length.describe()

data_length = 0

#filename = open('./dataset/clean_qa.txt', 'a+')
filename= './dataset/clean_qa2.txt'
# Membuka file clean_qa.txt untuk menulis
with open(filename, 'w', encoding='utf-8') as f:
  # Mengiterasi melalui baris-baris dataset
  for index, row in df.iterrows():
    question = normalize_sentence(str(row['question'])) # Normalisasi pertanyaan
    question = normalize_sentence(question)
    question = stemmer.stem(question) # Stemming pertanyaan

    answer = str(row['answer']).lower().replace('iteung', 'aku').replace('\n', ' ') # Normalisasi jawaban

    # Memeriksa panjang pertanyaan dan jawaban
    if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
      body="{"+question+"}|<START> {"+answer+"} <END>"
      # Menulis pasangan pertanyaan dan jawaban ke file clean_qa.txt
      print(body, file=f)
      #filename.write(f"{question}\t<START> {answer} <END>\n")

#filename.close()
