import tensorflow as tf


class SelfAttentionMask(tf.keras.layers.Layer):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        inputs[0]: from_tensor: 2D or 3D Tensor of shape:
            [batch_size, from_seq_length, ...].
        inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Output:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """

    def call(self, inputs, to_mask):
        batch_size, from_seq_length = inputs.shape.as_list()[:2]

        to_seq_length = to_mask.shape.as_list()[1]

        to_mask = tf.cast(tf.reshape(
            to_mask, [batch_size, 1, to_seq_length]), inputs.dtype)

        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_length, 1], dtype=inputs.dtype)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask


if __name__ == "__main__":
    attention_mask = SelfAttentionMask()
    inputs = tf.random.uniform([2, 4, 3])
    to_mask = tf.constant([[0, 1, 0], [1, 1, 0]])
    new_mask = attention_mask(inputs, to_mask)
    print(to_mask)
    print(new_mask)
