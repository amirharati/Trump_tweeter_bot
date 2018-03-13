"""
    A simple charecter model. This is not a conditional LM and is created for
    debugging.
"""
import tensorflow as tf
import numpy as np


class CharLmModel():
  def __init__(self, inputs, vocabs, num_layers, model_size, gs):
    self.sequence = inputs['seq']
    self.lengths = inputs['length']
    self.model_size = model_size
    self.num_layers = num_layers
    self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
    self.vocabs = vocabs
    self.vocab_size = len(vocabs)

    with tf.variable_scope("main", initializer=tf.contrib.layers.xavier_initializer()):
      # embadding for chars
      embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
      embed_seq = tf.nn.embedding_lookup(embedding, self.sequence)
      cells = []
      for i in range(self.num_layers):
        c = tf.nn.rnn_cell.GRUCell(self.model_size)
        c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=self.keep_prob,
                                          output_keep_prob=self.keep_prob)
        cells.append(c)
      cell = tf.nn.rnn_cell.MultiRNNCell(cells)

      # run the rnn
      outputs, state = tf.nn.dynamic_rnn(cell, embed_seq, dtype=tf.float32, sequence_length=self.lengths)

      # compute the loss and also predictions
      loss, preds = self._loss(outputs)

      self.preds = preds
      self.loss = tf.reduce_mean(loss)
      tf.summary.scalar("loss", self.loss)
      with tf.variable_scope("train_op"):
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.initial_learning_rate = tf.constant(0.005)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                       self.global_step,
                                                        200, 0.9)
        tf.summary.scalar("learning_rate", learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Visualise gradients
        vis_grads = [0 if i is None else i for i in grads]
        for g in vis_grads:
          tf.summary.histogram("gradients_" + str(g), g)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._sample_ops()

  def _loss(self, outputs):
    targets = self.sequence[:, 1:]
    outputs = outputs[:, :-1, :]
    logits = tf.contrib.layers.fully_connected(outputs, num_outputs=self.vocab_size, activation_fn=None)

    mask = tf.sign(tf.abs(tf.cast(targets, dtype=tf.float32)))
    preds = tf.argmax(tf.nn.softmax(logits), axis=2)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                                            mask,
                                            average_across_timesteps=False,
                                            average_across_batch=True)
    # loss = tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=mask)
    return loss, preds

  def _sample_ops(self):
     self.input = tf.placeholder(dtype=tf.int32, shape=[None],
                                    name="sample_input")

