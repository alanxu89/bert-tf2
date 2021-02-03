import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self,
                 initial_learning_rate,
                 decay_schedule_fn,
                 warmup_steps,
                 power=1.0):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_percent_done = step / warmup_steps

        if warmup_percent_done < 1.0:
            warmup_learning_rate = self.initial_learning_rate * \
                tf.math.pow(warmup_percent_done, self.power)
            return warmup_learning_rate
        else:
            return self.decay_schedule_fn(step)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
        }
