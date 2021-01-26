import tensorflow as tf
import numpy as np

import time


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence.
    The mask indicates where pad value 0 is present:
    it outputs a 1 at those locations, and a 0 otherwise.
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # (batch_size, seq_len) => (batch_size, 1, 1, seq_len)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    mask the future tokens in a sequence. a value of 1 indicates future step
    used for scaled dot product attention
    """
    return 1.0 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    if mask is not None:
        # send logits to -inf for masked positions
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0

        self.depth = self.d_model//self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension (width) into (num_heads, depth), then transpose:
            (batch_size, seq_len, width) => (batch_size, num_heads, seq_len, depth)
        Note: this is NOT splitting the seq_len dimension
        """
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # split q, k, v into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        scaled_attention, _ = scaled_dot_product_attention(
            q, k, v, mask)

        # => (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # => (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(
            scaled_attention, [batch_size, -1, self.d_model])

        output = self.dense(concat_attention)

        return output


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_prob):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_prob)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_prob)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)

        # apply dropout for training
        attn_output = self.dropout1(attn_output, training=training)

        # residual connection and layer norm
        out1 = self.layernorm1(x + attn_output)

        out2 = self.ffn(out1)
        ffn_out = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out2 + ffn_out)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_prob):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_prob)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_prob)
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout_prob)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        attn_output1 = self.mha1(x, x, x, look_ahead_mask)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(attn_output1 + x)

        attn_output2 = self.mha2(enc_out, enc_out, out1, padding_mask)
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(attn_output2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximum_position_encoding,
                 dropout_prob):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(
            d_model, num_heads, dff, dropout_prob) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate=dropout_prob)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))    # why?
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x    # (batch_size, seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 dropout_prob):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model,
                                        num_heads, dff, dropout_prob) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate=dropout_prob)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))    # why?
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_out, training,
                                   look_ahead_mask, padding_mask)

        return x    # (batch_size, seq_len, d_model)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, dropout_prob=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, dropout_prob)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, dropout_prob)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, targ, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        enc_out = self.encoder(inp, training, enc_padding_mask)

        dec_out = self.decoder(targ, enc_out, training,
                               look_ahead_mask, dec_padding_mask)

        return self.final_layer(dec_out)


if __name__ == "__main__":
    sample_transformer = Transformer(
        num_layers=4, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=12000, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform(
        (64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform(
        (64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out = sample_transformer(temp_input, temp_target, training=False,
                                enc_padding_mask=None,
                                look_ahead_mask=None,
                                dec_padding_mask=None)

    t0 = time.time()
    fn_out = sample_transformer(temp_input, temp_target, training=False,
                                enc_padding_mask=None,
                                look_ahead_mask=None,
                                dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
    print(time.time() - t0)
