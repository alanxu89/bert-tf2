# coding=utf-8

import copy
import json

import numpy as np
import tensorflow as tf

from embedding import WordEmbedding, PositionEmbedding, TokenTypeEmbedding


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


class BertModel(tf.keras.Model):
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

        self.word_embedding = WordEmbedding(
            config.vocab_size,
            config.embedding_size,
            create_initializer(config.initializer_range),
            use_one_hot_embeddings)

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

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = encoder_output[:, 0, :]
        pooler_layer = tf.keras.layers.Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=create_initializer(config.initializer_range),
            name='pooler_transform')
        cls_output = pooler_layer(first_token_tensor)


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


if __name__ == "__main__":
    d = {"vocab_size": 100, "hidden_size": 1024}
    config = BertConfig.from_dict(d)
    print(config.to_json_string())
