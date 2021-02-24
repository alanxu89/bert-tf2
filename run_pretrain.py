import tensorflow as tf
import numpy as np
import time

from model import BertConfig, BertModel


def loss_function(bert_config, log_probs, label_ids, label_weights):
    """
        log_probs: (batch_size, seq_len, vocab_size)
        label_ids, label_weights: (batch_size, seq_len)
    """
    log_probs = tf.reshape(log_probs, [-1, bert_config.vocab_size])
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    return loss


train_loss = tf.keras.metrics.Mean(name='train_loss')


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(768)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


@tf.function()
def train_step(input_ids, masked_lm_positions):
    with tf.GradientTape() as tape:
        masked_lm_outputs = bert_model(
            input_ids, masked_lm_positions, is_training=True)
        label_weights = tf.ones(masked_lm_outputs.shape.as_list()[:2])
        loss = loss_function(config, masked_lm_outputs,
                             masked_lm_positions, label_weights)

    gradients = tape.gradient(loss, bert_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bert_model.trainable_variables))

    train_loss(loss)
    # train_accuracy(accuracy_function(tar_real, predictions))


if __name__ == "__main__":
    d = {"vocab_size": 2000, "hidden_size": 768}
    config = BertConfig.from_dict(d)
    print(config.to_json_string())

    bert_model = BertModel(config)

    input_ids = tf.constant(np.random.randint(
        2000, size=(16, 128)), dtype=tf.int32)

    masked_lm_positions = tf.constant(
        np.random.randint(129, size=(16, 32)), dtype=tf.int32)

    output = bert_model(input_ids, masked_lm_positions)

    bert_model.summary()

    t0 = time.time()

    train_step(input_ids, masked_lm_positions)

    print("time for one train_step: ", time.time() - t0)
