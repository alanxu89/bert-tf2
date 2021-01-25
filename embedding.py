import tensorflow as tf


class WordEmbedding(tf.keras.layers.Layer):
    """Performs an embedding lookup suitable for accelerator devices.

    This layer uses either tf.gather or tf.one_hot to translate integer indices to
    float embeddings.

    Args:
    vocab_size: Number of elements in the vocabulary.
    embedding_width: Output size of the embedding layer.
    initializer: The initializer to use for the embedding weights. Defaults to
        "glorot_uniform".
    use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
        lookup. Defaults to False (that is, using tf.gather). Setting this option
        to True may improve performance, especially on small vocabulary sizes, but
        will generally require more memory.
    scale_factor: Whether to scale the output embeddings. Defaults to None (that
        is, not to scale). Setting this option to a float will let values in
        output embeddings multiplied by scale_factor.
    """

    def __init__(self,
                 vocab_size,
                 embedding_width,
                 initializer="glorot_uniform",
                 use_one_hot=False,
                 scale_factor=None,
                 **kwargs):
        super(WordEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_width = embedding_width
        self.initializer = tf.keras.initializer.get(initializer)
        self.use_one_hot = use_one_hot
        self.scale_factor = scale_factor

    def build(self, input_shape):
        self.embedding_table = self.add_weight(
            "embedding_table",
            shape=[self.vocab_size, self.embedding_width],
            initializer=self.initializer,
            dtype=tf.float32)

    def call(self, inputs):
        # input shape: (batch_size, seq_length)

        flat_inputs = tf.reshape(inputs, [-1])
        if self.use_one_hot:
            # (batch_size*seq_length, vocab_size)
            one_hot_data = tf.one_hot(
                flat_inputs, depth=self.vocab_size, dtype=tf.float32)
            embeddings = tf.matmul(one_hot_data, self.embedding_table)
        else:
            embeddings = tf.gather(self.embedding_table, flat_inputs)
        to_shape = tf.concat(
            [tf.shape(inputs), [self.embedding_width]], axis=0)
        embeddings = tf.reshape(embeddings, to_shape)

        if self.scale_factor is not None:
            return embeddings *= self.scale_factor
        return embeddings

    @property
    def vocab_size(self):
        return self.vocab_size

    @property
    def embedding_width(self):
        return self.embedding_width

    @property
    def embedding_table(self):
        return self.embedding_table


class TokenTypeEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 token_type_vocab_size,
                 width,
                 initializer,
                 **kwargs):
        super(TokenTypeEmbedding, self).__init__(**kwargs)
        self.token_type_vocab_size = token_type_vocab_size
        self.width = width
        self.initializer = initializer

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        if len(input_shape) != 2:
            raise ValueError("TokenTypeEmbedding expects a 2-dimensional input tensor "
                             "of shape [batch, sequence], got "
                             "{}".format(input_shape))
        self.batch_size = input_shape[0]
        self.seq_length = input_shape[1]

        self.kernel = self.add_weight(
            "kernel",
            shape=[self.token_type_vocab_size, self.width],
            initializer=self.initializer)
        super(PositionEmbedding, self).build(input_shape)

    def call(self, input):
        # This vocab will be small so we always do one-hot here,
        # since it is always faster for a small vocabulary.
        # input shape: [batch_size, seq_length]
        # output_shape: [batch_size, seq_length, width]

        # [batch_size, seq_length] => [batch_size*seq_length]
        flat_token_type_ids = tf.reshape(input, [-1])

        # [batch_size*seq_length] => [batch_size*seq_length, vocab_size]
        one_hot_ids = tf.one_hot(
            flat_token_type_ids, depth=self.token_type_vocab_size)

        # get shape [batch_size*seq_length, width]
        token_type_embeddings = tf.matmul(one_hot_ids, self.kernel)

        return tf.reshape(token_type_embeddings,
                          [self.batch_size, self.seq_length, self.width])


class PositionEmbedding(tf.keras.layers.Layer):
    """create position embedding with respect to input shape """

    def __init__(self,
                 max_length,
                 initializer,
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        if max_length is None:
            raise ValueError("`max_length` must be an Integer, not `None`.")
        self.max_length = max_length
        self.initializer = initializer

    def build(self, input_shape):
        dimension_list = input_shape.as_list()
        if len(dimension_list) != 3:
            raise ValueError("PositionEmbedding expects a 3-dimensional input tensor "
                             "of shape [batch, sequence, width], got "
                             "{}".format(input_shape))
        width = dimension_list[2]

        self.position_embeddings = self.add_weight(
            "embeddings",
            shape=[self.max_length, width],
            initializer=self.initializer)
        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        position_embeddings = self.position_embeddings[:input_shape[1], :]
        return tf.broadcast_to(position_embeddings, input_shape)


if __name__ == "__main__":
    emb = PositionEmbedding(100)
    print(emb.get_config())
    position_embedding = PositionEmbedding(max_length=100)
    inputs = tf.keras.Input((50, 32), dtype=tf.float32)
    inputs = tf.constant(np.random.randn(2, 3, 4))
    outputs = position_embedding(inputs)
    print(outputs)
