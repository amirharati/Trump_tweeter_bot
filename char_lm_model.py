"""
    A simple charecter model. This is not a conditional LM and is created for
    debugging.
"""
import tensorflow as tf
import numpy as np

class CharLmModel():
  def __init__(self, inputs, vocabs, model_size, gs):
    self.sequence = inputs['seq']
    self.lengths = inputs['length']
    self.model_size = model_size
    #self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
    self.vocabs = vocabs
    # need to add 1 to account for padding (or maybe just add a <PAD> to vocab?)
    self.vocab_size = len(vocabs) + 1  # without 1 it diverge!

    self.global_step = gs
    self.increment_gs = tf.assign(self.global_step, self.global_step + 1) #To increment during val


    with tf.variable_scope("main", initializer=tf.contrib.layers.xavier_initializer()):
      embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
      embed_seq = tf.nn.embedding_lookup(embedding, self.sequence)
      cell = tf.nn.rnn_cell.GRUCell(self.model_size)
      outputs, state = tf.nn.dynamic_rnn(cell, embed_seq, dtype=tf.float32, sequence_length=self.lengths)

      # compute the loss and also predictions
      loss, preds = self._loss(outputs)

      self.preds = preds
      self.loss = tf.reduce_mean(loss)
      tf.summary.scalar("loss", self.loss)
      #opt = tf.train.AdamOptimizer(0.001)
      #self.train_op = opt.minimize(self.loss, global_step=self.global_step)
      with tf.variable_scope("train_op"):
        #self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.initial_learning_rate = tf.constant(0.001)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                       self.global_step,
                                                        100, 0.9)
        tf.summary.scalar("learning_rate", learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Visualise gradients
        vis_grads = [0 if i is None else i for i in grads]
        for g in vis_grads:
          tf.summary.histogram("gradients_" + str(g), g)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

  def _loss(self, outputs):
    #self.oshape = tf.Print(outputs.shape, [outputs.shape], message="This is outputs: ")
    #self.OOO = tf.shape(outputs)
    targets = self.sequence[:, 1:]
    outputs = outputs[:, :-1, :]
    logits = tf.contrib.layers.fully_connected(outputs, num_outputs=self.vocab_size, activation_fn=None)
    self.OOO = tf.shape(logits)
    #mask = tf.sequence_mask(self.lengths - 1, tf.reduce_max(self.lengths) - 1)
    #with tf.variable_scope("softmax"):
    #  self.softmax_w = tf.get_variable("softmax_w",
    #                                    [self.model_size, self.vocab_size])
    #  self.softmax_b = tf.get_variable("softmax_b",
    #                                    [self.vocab_size])
      # [(batch_sizrxvocab_size)]
    #  logits = [tf.matmul(outputStep, self.softmax_w) + self.softmax_b for outputStep in outputs]

    # logits = tf.stack(logits, axis=1)

    mask = tf.sign(tf.abs(tf.cast(targets, dtype=tf.float32)))
    preds = tf.argmax(tf.nn.softmax(logits), axis=2)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                                            mask,
                                            average_across_timesteps=False,
                                            average_across_batch=True)
    #loss = tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=mask)
    return loss, preds
