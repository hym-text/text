import tensorflow as tf

# DENSE_UNIT = 200

class MyLSTM(tf.keras.Model):
  def __init__(self, vocab_size, vector_size, max_length):
    super(MyLSTM, self).__init__()
    self.vocab_size = vocab_size
    # self.embed1 = tf.keras.layers.Embedding(vocab_size, vector_size, input_length=max_length, embeddings_initializer=embedding_matrix, trainable=True, mask_zero=False)
    self.embed1 = tf.keras.layers.Embedding(vocab_size, vector_size, input_length=max_length, mask_zero=False)
    self.embed2 = tf.keras.layers.Embedding(vocab_size, vector_size, input_length=max_length, mask_zero=False)
    
    self.lstm1 = tf.keras.layers.CuDNNLSTM(units=100)
    self.lstm2 = tf.keras.layers.CuDNNLSTM(units=100)

    # self.dropout = tf.keras.layers.Dropout(0.3)
    # self.dense2 = tf.keras.layers.Dense(units=1, activation="sigmoid")

    # # self.reshape = tf.keras.layers.Reshape((max_length, DENSE_UNIT, 1))
    # self.conv1 = tf.keras.layers.Conv1D(1000, kernel_size=2, strides=1, activation="tanh", padding="valid")
    # # self.conv2 = tf.keras.layers.Conv2D(141, kernel_size=(2, vector_size), strides=(1, 1), activation="tanh", padding="valid")
    # self.maxpool1 = tf.keras.layers.GlobalMaxPooling1D()
    # self.maxpool2 = tf.keras.layers.GlobalMaxPooling1D()


  def lstm_layer1(self, input1):
    print("input", input1.shape)
    result1 = self.embed1(input1)
    print("emb", result1.shape)
    result1 = self.lstm1(result1)
    print("lstm1", result1)
    # result1 = input1
    return result1

  def lstm_layer2(self, input2):
    print("input2", input2.shape)
    # result2 = self.lstm1(input2)
    result2 = self.embed2(input2)
    print("embed2", result2.shape)
    result2 = self.lstm2(result2)
    # result = tf.nn.tanh(result)
    # return input2
    return result2
