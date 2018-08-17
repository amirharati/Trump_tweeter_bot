"""
    A simple charecter model. This is not a conditional LM and is created for
    debugging.
"""
import tensorflow as tf
import numpy as np


class WordLmModel():
  def __init__(self, inputs, vocabs, reverse_vocabs,
               num_layers, model_size, embedding_size):
    self.sequence = inputs['seq']
    self.lengths = inputs['length']
    self.model_size = model_size
    self.num_layers = num_layers
    self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
    self.vocabs = vocabs
    self.reverse_vocabs = reverse_vocabs
    self.vocab_size = len(vocabs)
    self.embedding_size = embedding_size

    with tf.variable_scope("main", initializer=tf.contrib.layers.xavier_initializer()):
      self.embedding = tf.get_variable("embedding",
         [self.vocab_size, self.embedding_size], dtype=tf.float32)
      embed_seq = tf.nn.embedding_lookup(self.embedding, self.sequence)
      cells = []
      for i in range(self.num_layers):
        c = tf.nn.rnn_cell.GRUCell(self.model_size)
        c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=self.keep_prob,
                                          output_keep_prob=self.keep_prob)
        cells.append(c)
      self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)
      # run the rnn
      outputs, state = tf.nn.dynamic_rnn(self.cell, embed_seq, dtype=tf.float32, sequence_length=self.lengths, scope="DRNN")

      # compute the loss and also predictions
      loss, preds = self._loss(outputs)

      self.preds = preds
      #self.loss = tf.reduce_sum(loss)
      self.loss = tf.reduce_mean(loss)
      tf.summary.scalar("loss", self.loss)
      with tf.variable_scope("train_op"):
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.initial_learning_rate = tf.constant(0.001)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                       self.global_step,
                                                        100, 0.99)
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
    with tf.variable_scope("softmax"):
      logits = tf.layers.dense(outputs, self.vocab_size, None, name="logits")

      mask = tf.sign(tf.abs(tf.cast(targets, dtype=tf.float32)))
      #mask = tf.sequence_mask(self.lengths - 1, tf.reduce_max(self.lengths) - 1)
      preds = tf.argmax(tf.nn.softmax(logits), axis=2)
      #loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
      #                                        mask,
      #                                        average_across_timesteps=False,
      #                                        average_across_batch=True)
    # alternative loss
    loss = tf.losses.sparse_softmax_cross_entropy(targets,
                                                  logits,
                                                   weights=mask)
    return loss, preds

  def _sample_ops(self):
    """
      Define a ops for sampling.
    """
    # create init state
    self.initial_states = self.cell.zero_state(1, dtype=tf.float32)

    self.current_states = list()
    for i in range(0, self.num_layers):
      self.current_states.append(tf.placeholder(tf.float32,                         shape=[1, self.model_size],
                                 name="gru_state_" + str(i)))

    self.current_states = tuple(self.current_states)
    # input for current time step
    self.input = tf.placeholder(dtype=tf.int32, shape=[None, 1],
                                    name="sample_input")

    embed_seq = tf.nn.embedding_lookup(self.embedding, self.input)
    outputs, self.state = tf.nn.dynamic_rnn(self.cell,
                                            embed_seq,
                                            dtype=tf.float32,
                                            initial_state=self.current_states,
                                            scope="DRNN")

    with tf.variable_scope("softmax", reuse=True):
      logits = tf.layers.dense(outputs, self.vocab_size, None,
                               reuse=True, name="logits")
      self.probs = tf.nn.softmax(logits)

  def sample(self, sess):
    """
      Do the actual sampling.
    """
    current_seq_ind = []
    iteration = 0

    initial_states = sess.run(self.initial_states)

    s = initial_states
    p = (1.0 / (self.vocab_size)) * np.ones(self.vocab_size)
    while iteration < 1000:
        # Now p contains probability of upcoming char, as estimated by model, and s the last RNN state
        ind_sample = np.random.choice(range(0, self.vocab_size), p=p.ravel())

        if self.reverse_vocabs[ind_sample] == "<EOS>":  # EOS token
            break
        if iteration == 0:
          ind_sample = self.vocabs["<START>"]
        else:
          current_seq_ind.append(ind_sample)

        # Create feed dict for states
        feed = dict()
        feed[self.keep_prob] = 1.0
        for i in range(0, self.num_layers):
            for c in range(0, len(s[i])):
                feed[self.current_states[i]] = s[i]
        tmp = np.array([ind_sample])
        tmp = np.reshape(tmp, [1, 1])
        feed[self.input] = tmp  # Add new input symbol to feed
        [p, s] = sess.run([self.probs, self.state], feed_dict=feed)
        iteration += 1
    # bug with data prep
    self.reverse_vocabs[3] = " "
    out_str = ""
    for c in current_seq_ind:
        out_str += self.reverse_vocabs[c] + " "
    print(out_str)

