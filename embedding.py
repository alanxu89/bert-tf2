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
