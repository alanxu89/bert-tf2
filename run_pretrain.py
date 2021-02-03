import tensorflow as tf


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
