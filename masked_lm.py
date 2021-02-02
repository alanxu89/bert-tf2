import tensorflow as tf


class MaskedLM(tf.keras.layers.Layer):
    """Masked language model network head for BERT modeling.
    Args:
        embedding_table: The embedding table from encoder network.
        activation: The activation, if any, for the dense layer.
        initializer: The initializer for the dense layer. Defaults to a Glorot
        uniform initializer.
        output: The output style for this layer. Can be either 'logits' or
        'predictions'.
    """

    def __init__(self,
                 embedding_table,
                 activation,
                 initializer="glorot_uniform",
                 output="logits",
                 name=None,
                 **kwargs
                 ):
        super(MaskedLM, self).__init__()
        self.embedding_table = embedding_table
        self.activation = activation
        self.initializer = tf.keras.initializers.get(initializer)

        if output not in ("predictions", "logits"):
            raise ValueError(
                "`output` can be either `logits` or `predictions`")
        self.output_type = output

    def build(self, input_shape):
        self.vocab_size, self.hidden_size = self.embedding_table.shape.as_list()
        self.dense = tf.keras.layers.Dense(
            self.hidden_size,
            activation=self.activation,
            kernel_initializer=self.initializer,
            name="transform/dense")
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-2, name='transform/LayerNorm')
        self.bias = self.add_weight('output_bias/bias',
                                    shape=(self.vocab_size,),
                                    initializer="zeros",
                                    trainable=True
                                    )
        super(MaskedLM, self).build(input_shape)

    def call(self, sequence_data, masked_positions):
        # => (batch_size * num_predictions, num_hidden)
        masked_lm_input = self._gather_indexes(sequence_data, masked_positions)

        # => (batch_size * num_predictions, num_hidden)
        lm_data = self.dense(masked_lm_input)
        lm_data = self.layer_norm(lm_data)

        # embedding_table shape: [vocab_size, embedding_size]
        # embedding = tf.gather(embedding_table, flat_input_ids)
        # embedding_size = num_hidden
        # (batch_size * num_predictions, num_hidden) => (batch_size * num_predictions, vocab_size)
        lm_data = tf.matmul(lm_data, self.embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(lm_data, self.bias)

        masked_positions_length = masked_positions.shape.as_list()[1]
        # => (batch_size, num_predictions, vocab_size)
        logits = tf.reshape(
            logits, [-1, masked_positions_length, self.vocab_size])

        if self.output_type == "logits":
            return logits
        else:
            return tf.nn.log_softmax(logits)

    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions.

        Args:
            sequence_tensor: Sequence output of `BertModel` layer of shape
            (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
            hidden units of `BertModel` layer.
            positions: Positions ids of tokens in sequence to mask for pretraining
            of with dimension (batch_size, num_predictions) where
            `num_predictions` is maximum number of tokens to mask out and predict
            per each sequence.

        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            num_hidden).
        """
        batch_size, seq_length, width = sequence_tensor.shape.as_list()
        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32), [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor, [-1, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

        return output_tensor
