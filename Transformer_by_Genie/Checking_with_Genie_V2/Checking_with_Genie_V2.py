# This code implements a Transformer model for sentiment analysis on the IMDB movie reviews dataset using Keras and TensorFlow.
# It includes token and position embedding, multiple transformer blocks, and a final dense layer for classification.
# It uses as input the IMDB dataset, preprocesses it, and trains the model to predict sentiment from movie reviews.
# Of each review, the first 200 words are considered, and the top 20,000 words are used for vocabulary.
# The model is then evaluated on a test set, and training/validation accuracy and loss are plotted.
# Now we are going to susbstitute GlobalAveragePooling1D with teh BERT strategy (using only the embedding of the first token
# as input for the classification

# Import necessary libraries
import numpy as np
import keras
import tensorflow as tf

from keras import Sequential

from keras.datasets import imdb

from keras.models import Model

from keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from keras.layers import MultiHeadAttention
from keras.layers import Layer
from keras.layers import Lambda

#from keras.layers import GlobalAveragePooling1D
#from keras import backend as K
#from keras import initializers, regularizers, constraints
#from keras.layers import Add
#from keras.layers import Flatten
#from keras.layers import Activation
#from keras.layers import Concatenate
#from keras.layers import Reshape
#from keras.layers import Lambda
#from keras.optimizers import Adam
#from keras.layers import LSTM
#from keras.layers import Bidirectional
#from keras.layers import TimeDistributed
#from keras.layers import GRU
#from keras.layers import SimpleRNN
#from keras.layers import Conv1D
#from keras.layers import MaxPooling1D
#from keras.layers import GlobalMaxPooling1D
#from keras.layers import BatchNormalization


print ("Libraries imported successfully.")

vocab_size = 20000  # Only consider the top 20k words
CLS_ID = vocab_size
vocab_size_with_CLS = vocab_size + 1
maxlen = 200  # Only consider the first 200 words of each movie review

# Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# Prepend the CLS token to each sequence
def prepend_cls_token(sequences, cls_id=CLS_ID, maxlen=200):
    new_sequences = []
    for seq in sequences:
        seq = [cls_id] + list(seq)
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        else:
            seq = seq + [0] * (maxlen - len(seq))
        new_sequences.append(seq)
    return np.array(new_sequences)

X_train = prepend_cls_token(X_train, cls_id=CLS_ID, maxlen=maxlen)
X_test = prepend_cls_token(X_test, cls_id=CLS_ID, maxlen=maxlen)

X_train = keras.utils.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.utils.pad_sequences(X_test, maxlen=maxlen)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary size with CLS token: {vocab_size_with_CLS}")
print(f"Max sequence length: {maxlen}")

# Define the Token and Position Embedding layer
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
# Define the Transformer Block
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the model
embed_dim = 128
num_heads = 4
num_transformer_blocks = 4
ff_dim = 128

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size_with_CLS, embed_dim)
x = embedding_layer(inputs)
for _ in range(num_transformer_blocks):
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
# ... después de tus bloques Transformer
# x = GlobalAveragePooling1D()(x)  # QUITA esta línea
x = Lambda(lambda t: t[:, 0, :])(x)  # Usa solo el embedding del primer token (CLS)
    
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=x)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# Print the model summary
model.summary()

# Print the model architecture
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Check if the model is built correctly
# Train the model
history = model.fit(X_train, y_train, batch_size=8, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Plot training & validation accuracy values
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()