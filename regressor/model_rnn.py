import tensorflow as tf
from tensorflow.keras import layers, models

class BIRNN(models.Model):
    def __init__(self, num_hidden, num_layers, num_classes, embed_dim, dict_size=32, dropout_rate=0.5):
        super(BIRNN, self).__init__()

        self.embedding = layers.Embedding(dict_size, num_hidden, input_length=embed_dim, mask_zero=True)

        # Bidirectional LSTM layers
        self.bidirectional_rnn = []
        for _ in range(num_layers):
            self.bidirectional_rnn.append(layers.Bidirectional(
                layers.LSTM(num_hidden, dropout=dropout_rate, return_sequences=True),
                merge_mode='concat'
            ))
        

        self.pool = layers.GlobalAveragePooling1D()# default input shape: (batch_size, sequence_length, num_hidden*2)

        # Fully Connected Layers
        self.dense1 = layers.Dense(128, activation=None)#change to num hidden?
        self.prelu = layers.PReLU(shared_axes=[1])
        self.dropout_mlp = layers.Dropout(dropout_rate)
        self.dense_output = layers.Dense(num_classes, activation=None)

    def call(self, inputs, training=False):
        x, c = inputs # shape: (batch_  size, sequence_length), (batch_size, )

        x = self.embedding(x) # shape: (batch_size, sequence_length, num_hidden)

        # Bidirectional LSTM layers
        for layer in self.bidirectional_rnn:
            x = layer(x) # shape: (batch_size, sequence_length, num_hidden*2)
        
        x = self.pool(x) # shape: (batch_size, num_hidden*2)

        #convert c to a column vector
        c = tf.reshape(c, (-1,1)) # shape: (batch_size, 1)

        #cast c to float32
        c = tf.cast(c, tf.float32)

        x = tf.concat([x, c], axis=1) # shape: (batch_size, sequence_length*num_hidden*2 + 1)

        x = self.dense1(x) # shape: (batch_size, num_hidden)

        x = self.prelu(x)

        x = self.dropout_mlp(x, training=training)

        output = self.dense_output(x) # shape: (batch_size, num_classes)


        return output

if __name__ == '__main__':
    # Example usage:
    num_hidden = 128
    num_layers = 2
    num_classes = 1
    embed_dim = 66
    dict_size = 32

    # Create an instance of the BIRNN model
    birnn_model = BIRNN(num_hidden, num_layers, num_classes, embed_dim,  dict_size)

    # Display the model summary
    birnn_model.build(input_shape=[(None, embed_dim), (None, )])

    birnn_model.summary()

# q: what is the recommended way of adding dropout to an RNN?
# a: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN