#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../Desktop/seq2seq_mm/seq2seq_NMT'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # seq2seq 学習用モデルの構築
#%% [markdown]
# ### import
# 本プログラムで必要になるライブラリのimportを行う.

#%%
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

#%% [markdown]
# ### 各種パラメータの設定
# 学習時のバッチサイズ,エポック数,中間層のユニット数の設定を行う.

#%%
batch_size = 64   # バッチサイズ
epochs = 100     # エポック数
latent_dim = 256  # 中間層ユニット数

#%% [markdown]
# ## データの読み込み
# csvファイルをpandasの`read_csv()`によって読み込みを行う.
# 
# また,先頭5行を表示し,中身の確認を行う.

#%%
texts = pd.read_csv("fra-eng/fra-eng.csv", index_col=0)
texts.head()

#%% [markdown]
# ## 入力文の定義
# 本プログラムでは英語を入力文とするため,変数に英語文を格納しておく.

#%%
input_texts = texts["english"]
input_texts.head()

#%% [markdown]
# ### 入力文の文字の辞書を作成
# 後に文字列を各文字ごとのindexに置き換えるために,各文字に数字(index)を割り振る.
# 
# その定義のために{char : index}の辞書を作成する.
# 
# なお、`0` はパディング用に使用するためここでは使用しない.

#%%
input_char_dict = {}
input_char_cnt = 1
for input_text in input_texts:
    for char in input_text:
        if char not in input_char_dict.keys():
            input_char_dict[char] = input_char_cnt
            input_char_cnt += 1
print(input_char_dict)

#%% [markdown]
# ## 出力文の定義
# 本プログラムではフランス語を出力とする.

#%%
target_texts = texts["french"]
target_texts.head()

#%% [markdown]
# ### 出力文の開始文字と終端文字の定義
# 出力文には文の始め(開始文字)を示す `\t` と文の終わり(終端文字)を示す `\n` を付与する.

#%%
target_texts = ["\t"+text+"\n" for text in target_texts]
target_texts[:5]

#%% [markdown]
# ### 出力文の文字の辞書の作成
# 入力文と同様,後に文字列をindexに置き換えるために,各文字に数字(index)を割り振る.
# 
# その定義のために{char : index}の辞書を作成する.

#%%
target_char_dict = {}
# target_char_cnt = 1
for target_text in target_texts:
    for char in target_text:
        if char not in target_char_dict.keys():
            # target_char_dict[char] = target_char_cnt
            # target_char_cnt += 1
            target_char_dict[char] = input_char_cnt
            input_char_cnt += 1
print(target_char_dict)

#%% [markdown]
# ### 確認
# データ数,入力の語彙数,出力の語彙数,入力の最大長,出力の最大長の確認.

#%%
num_encoder_tokens = len(input_char_dict)
num_decoder_tokens = len(target_char_dict)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('データ数    ：', len(input_texts))
print('入力の語彙数：', num_encoder_tokens)
print('出力の語彙数：', num_decoder_tokens)
print('入力の最大長：', max_encoder_seq_length)
print('出力の最大長：', max_decoder_seq_length)

#%% [markdown]
# ### 文章をindexの系列に置換
# 作成した辞書に従い,元の文字列を文字のindexの系列に置換する.

#%%
encoder_input_texts  = []
decoder_input_texts  = []
decoder_target_texts = []
for text in input_texts:
    encoder_input_texts.append([input_char_dict[char] for char in text])
for text in target_texts:
    decoder_input_texts.append([target_char_dict[char] for char in text])
for text in decoder_input_texts:
    decoder_target_texts.append(text[1:]) # targetは1step先から。開始文字は含まない
print(decoder_target_texts)

#%% [markdown]
# ### 文章を固定長にパディング
# 文章の長さがそれぞれ異なっているため,最大の長さを持つ文章の長さに合わせる.
# 
# このとき,長さが最大長に満たない文章に関しては0でパディングを行い,長さをあわせる.

#%%
MAX_SENTENCE_SIZE = 60


#%%
padded_input1_texts  = pad_sequences(encoder_input_texts,  maxlen=MAX_SENTENCE_SIZE, padding='post')
padded_input2_texts  = pad_sequences(decoder_input_texts,  maxlen=MAX_SENTENCE_SIZE, padding='post')
# decoder_target_texts = pad_sequences(decoder_target_texts, maxlen=max_decoder_seq_length, padding='post')
print(padded_input1_texts.shape)

#%% [markdown]
# ## tensorflow eager

#%%
import tensorflow as tf
## eager
tf.enable_eager_execution()

#%% [markdown]
# ## データセットの整形

#%%

WORD_VECTOR_SIZE = 100
BATCH_SIZE = 100
print("データセット準備中...")
print("データの準備中...")
dataset = tf.data.Dataset.from_tensor_slices(
      (tf.cast(padded_input1_texts, tf.int32),
       tf.cast(padded_input2_texts, tf.int32)))
print(padded_input1_texts.shape)
dataset = dataset.batch(BATCH_SIZE)

#%% [markdown]
# ## モデルの定義

#%%
from my_model import MyLSTM
# model = MyLSTM(vocab_size=vocab_size, vector_size=WORD_VECTOR_SIZE, embedding_matrix=embedding_matrix, max_length=MAX_SENTENCE_SIZE)
model = MyLSTM(vocab_size=num_decoder_tokens+num_encoder_tokens, vector_size=WORD_VECTOR_SIZE, max_length=MAX_SENTENCE_SIZE)
model.lstm_layer1 = tf.contrib.eager.defun(model.lstm_layer1) # コンパイル
model.lstm_layer2 = tf.contrib.eager.defun(model.lstm_layer2) # コンパイル

#%% [markdown]
# ## 損失関数

#%%
# 損失関数の定義
def loss(model, input1, input2):
    # print(input1)
    lstm1_output = model.lstm_layer1(input1)
    lstm2_output = model.lstm_layer2(input2)
    # print(lstm1_output[0])
    # print(lstm2_output[0])
    loss_v = tf.losses.mean_squared_error(lstm1_output, lstm2_output)
    # print("loss_v", loss_v)
    return loss_v

#%% [markdown]
# ## 勾配

#%%
# 勾配の計算 よしなにやってくれる
def grad(model, input1, input2):
    with tf.GradientTape() as tape:
        loss_value = loss(model, input1, input2)
    return tape.gradient(loss_value, model.variables)

#%% [markdown]
# ### optimizer

#%%
## optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

#%% [markdown]
# ### 保存

#%%
# 保存
import os
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  ## 続きから

#%% [markdown]
# ## モデルの学習

#%%
from tqdm import tqdm_notebook as tqdm


#%%
## モデルの学習
print('学習開始...')
EPOCH = 5
loss_history = [] # lossの履歴
val_loss_history = [] # 検証用
for i in range(EPOCH):
    print(i+1,"えぽっく目の開始")
#     dataset = make_dataset(i, df_train)
    loss_values = []
    for (batch, (input1, input2)) in tqdm(enumerate(dataset.take(-1))): # -1は全てとってくる
        # print(input1.shape)
        # print(input2.shape)
        # print(batch)
        loss_v = loss(model, input1, input2)
        loss_values.append(loss_v)
        if batch % 10 == 0: ## 学習中の確認用
            print(" Loss at step {:03d}: {:.5f}".format(batch, np.mean(loss_values)))
        grads = grad(model, input1, input2)
        optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
    loss_history.append(np.mean(loss_values)) # えぽっくごとの平均損失を記録
    print("平均損失：", np.mean(loss_values))
    checkpoint.save(file_prefix = checkpoint_prefix) # 1エポックごとにsave

#%% [markdown]
# # loss の可視化

#%%
# loss の可視化
import matplotlib.pyplot as plt
plt.plot(loss_history, label="close")
# plt.plot(val_loss_history, label = "validation")
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.show()


