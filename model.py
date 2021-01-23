# coding=utf-8

import copy
import json

import numpy as np
import tensorflow as tf


class BertConfig:
    """Config for BERT model"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02
                 ):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=0)
        for key, val in json_object.items():
            config.__dict__[key] = val
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=False, indent=2) + "\n"


class BertModel:
    """BERT model ("Bidirectional Encoder Representations from Transformers")."""

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None
                 ):
        """Constructor for BertModel.

        Args:
        config: `BertConfig` instance.
        is_training: bool. true for training model, false for eval model. 
            Controls whether dropout will be applied.
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
        scope: (optional) variable scope. Defaults to "bert".

        Raises:
        ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = input_ids.shape.as_list()
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        self.word_embedding = tf.keras.layers.Embedding(
            config.vocab_size,
            config.embedding_size,
            create_initializer(config.initializer_range))

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)
        self.token_type_embedding = TokenTypeEmbedding(
            config.type_vocab_size,
            config.embedding_size,
            create_initializer(config.initializer_range))

        self.position_embedding = PositionEmbedding(
            config.maximum_position_encoding,
            create_initializer(config.initializer_range))


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


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
    d = {"vocab_size": 100, "hidden_size": 1024}
    config = BertConfig.from_dict(d)
    print(config.to_json_string())

    emb = PositionEmbedding(100)
    print(emb.get_config())
    position_embedding = PositionEmbedding(max_length=100)
    inputs = tf.keras.Input((50, 32), dtype=tf.float32)
    inputs = tf.constant(np.random.randn(2, 3, 4))
    outputs = position_embedding(inputs)
    print(outputs)
