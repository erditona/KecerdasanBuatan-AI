# ITeung AI
Indonesia ChatBot using Seq2Seq (LSTM and BiLSTM) with Bahdanau Attention Mechanism

# Minimum Requirement Environment
## Minimum Python Sudah Terinstall
Python: 3.7.9
## Opsional  
CUDA Toolkit: 11.0.3 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)\
CuDNN: v8.0.5 [CuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

# Usage
1. Install all required depedency
```console
$ pip install -r requirements.txt
```

2. run preprocessing.py
```console
$ python preprocessing.py
```

3. run training.py
```console
$ python training.py
```
4. check output_dir get 4 file from there to iteung-ai plus daftar-slang from dataset
   ![image](https://github.com/mymyid/nlp/assets/11188109/33ffb73f-5c42-4ac5-8a1b-2d44349bf3a2)
5. Coba memakai model
```python
from iteung import reply
message = input("Kamu: ")
return_message, status = reply.botReply(message)
print(f"ITeung: {return_message}")
```
6. run testing.py if you want to test automatically
```console
$ python testing.py
```
