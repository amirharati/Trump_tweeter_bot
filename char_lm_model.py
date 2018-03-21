"""
    A simple charecter model. This is not a conditional LM and is created for
    debugging.
"""
import tensorflow as tf
import numpy as np


class CharLmModel():
  def __init__(self, inputs, vocabs, reverse_vocabs, num_layers, model_size, gs):
    self.sequence = inputs['seq']
    self.lengths = inputs['length']
    self.model_size = model_size
    self.num_layers = num_layers
    self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
    self.vocabs = vocabs
    self.reverse_vocabs = reverse_vocabs
    self.vocab_size = len(vocabs)

    with tf.variable_scope("main", initializer=tf.contrib.layers.xavier_initializer()):
      # embadding for chars
      self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
      embed_seq = tf.nn.embedding_lookup(self.embedding, self.sequence)
      cells = []
      #for i in range(self.num_layers):
      #  c = tf.nn.rnn_cell.GRUCell(self.model_size)
      #  c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=self.keep_prob,
      #                                    output_keep_prob=self.keep_prob)
      # cells.append(c)
      #self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)
      self.cell = tf.nn.rnn_cell.GRUCell(self.model_size)
      # run the rnn
      outputs, state = tf.nn.dynamic_rnn(self.cell, embed_seq, dtype=tf.float32, sequence_length=self.lengths,scope="DRNN")

      # compute the loss and also predictions
      loss, preds = self._loss(outputs)

      self.preds = preds
      self.loss = tf.reduce_sum(loss)
      tf.summary.scalar("loss", self.loss)
      with tf.variable_scope("train_op"):
        #self.global_step = tf.train.get_or_create_global_step()
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.initial_learning_rate = tf.constant(0.01)
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
    #seq = tf.one_hot(self.seq, len(self.vocab))
    #self.create_rnn(seq)
    #self.logits = tf.layers.dense(self.output, len(self.vocab), None)
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1],
    #                                                    labels=seq[:, 1:])
    targets = self.sequence[:, 1:]
    #targets = tf.one_hot(targets, self.vocab_size)
    #targets2 = self.sequence[:, 1:]
    outputs = outputs[:, :-1, :]
    #n = tf.shape(outputs)[1]
    #outputs = tf.unstack(outputs, n.value, axis=1)
    #outputs = tf.reshape(outputs, [-1, self.model_size])
    self.aaa1 = tf.Print(targets, [targets], "targets ", summarize=100)
    #targets = tf.reshape(targets, [-1])

    with tf.variable_scope("softmax"):
      logits = tf.layers.dense(outputs, self.vocab_size, None, name="logits")
      #logits = tf.contrib.layers.fully_connected(outputs, num_outputs=self.vocab_size, activation_fn=None, reuse=None,
      #  scope="logits_fully_connected")
      # self.softmax_w = tf.get_variable("softmax_w",
      #                                 [self.model_size, self.vocab_size])
      # self.softmax_b = tf.get_variable("softmax_b",
      #                                 [self.vocab_size])
      # [(batch_sizrxvocab_size)]
      #batch_size = tf.shape(outputs)[0]

      #T = tf.shape(outputs)[1]
      #outputs = tf.reshape(outputs, [-1, self.model_size])
      #self.aaa = tf.Print(batch_size, [batch_size], "XXXXX")
      #logits = tf.matmul(outputs, self.softmax_w) + self.softmax_b
      #logits = tf.reshape(logits, [batch_size, -1, self.vocab_size])

      #targets = tf.reshape(targets, [batch_size, -1])
      #self.aaa2 = tf.Print(targets, [targets], "targets2 ", summarize=100)
      #self.aaa3 = tf.Print(T, [T], "TTT ", summarize=100)

      #self.aaa = tf.Print(tf.shape(logits), [tf.shape(logits)], "XXXXX")
      #logits = [tf.matmul(outputStep, self.softmax_w) + self.softmax_b for outputStep in outputs]

      #logits = tf.stack(logits, axis=1)
      #targets = tf.stack(targets, axis=1)

    mask = tf.sign(tf.abs(tf.cast(targets, dtype=tf.float32)))

    #mask = tf.sequence_mask(self.lengths - 1, tf.reduce_max(self.lengths) - 1)
    preds = tf.argmax(tf.nn.softmax(logits), axis=2)
    self.aaa3 = tf.Print(preds, [preds], "pred ", summarize=100)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                                            mask,
                                            average_across_timesteps=False,
                                            average_across_batch=True)
    #y_one_hot = tf.one_hot(targets, self.vocab_size)
    #targets = tf.reshape(y_one_hot, logits.get_shape())
    #loss = tf.losses.sparse_softmax_cross_entropy(targets, logits, weights=mask)
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1],
    #                                                    labels=targets[:, 1:])
    return loss, preds

  def _sample_ops(self):
    """
      Define a ops for sampling.
    """
    # input state for current time-step
    self.initial_states = self.cell.zero_state(1, dtype=tf.float32)

    #self.current_states = list()
    #for i in range(0, self.num_layers):
    #  self.current_states.append(tf.placeholder(tf.float32,
    #                                   shape=[1, self.model_size],
    #                                   name="gru_state_" + str(i)))
    self.current_states = tf.placeholder(tf.float32,
                                       shape=[1, self.model_size],
                                       name="gru_state_")

    #self.current_states = tuple(self.current_states)
    # input for current time step
    self.input = tf.placeholder(dtype=tf.int32, shape=[None, 1],
                                    name="sample_input")

    embed_seq = tf.nn.embedding_lookup(self.embedding, self.input)
    input_by_time = embed_seq #tf.reshape(embed_seq, [1,1,-1])# tf.reshape(embed_seq, [1, 1, -1])
    # run the rnn  how ti use dynmaic?
    # TODO: How to get rid of dropout ?
    outputs, self.state = tf.nn.dynamic_rnn(self.cell, input_by_time, dtype=tf.float32, initial_state=self.current_states, scope="DRNN")
    #outputs = tf.stack(outputs, axis=1)
    #outputs = tf.reshape(outputs, [-1, self.model_size])
    with tf.variable_scope("softmax", reuse=True):
      logits = tf.layers.dense(outputs, self.vocab_size, None, reuse=True, name="logits")
    #weights = tf.get_variable("logits/kernel")
      #self.bbb1 = tf.Print(tf.get_variable_scope().reuse, [tf.get_variable_scope().reuse], "reuse ", summarize=100)
      #print("****", tf.get_variable_scope().reuse)
      #logits = tf.contrib.layers.fully_connected(outputs, num_outputs=self.vocab_size, activation_fn=None, reuse=tf.AUTO_REUSE, # is this correct?
      # scope="logits_fully_connected")
      #self.softmax_w = tf.get_variable("softmax_w",
      #                                  [self.model_size, self.vocab_size])
      #self.softmax_b = tf.get_variable("softmax_b",
      #                                  [self.vocab_size])
      # [(batch_sizrxvocab_size)]
      #batch_size = tf.shape(outputs)[0]
      #outputs = tf.reshape(outputs, [-1, self.model_size])
      #self.aaa = tf.Print(batch_size, [batch_size], "XXXXX")
      #logits = tf.matmul(outputs, self.softmax_w) + self.softmax_b
      #logits = tf.reshape(logits, [batch_size, -1, self.vocab_size])
      #logits = tf.stack(logits, axis=1)
    self.probs = tf.nn.softmax(logits)

  def sample(self, sess):
    """
      Do the actual sampling.
    """
    current_seq_ind = []
    iteration = 0
    #initial_states = []
    #for i in range(0, self.num_layers):
    #    initial_states.append([np.zeros(shape=[1, self.model_size], dtype=np.float32)])
    initial_states = sess.run(self.initial_states)

    #initial_states = tuple(initial_states)
    print(np.shape(initial_states))
    s = initial_states
    p = (1.0 / (self.vocab_size)) * np.ones(self.vocab_size)
    print(self.reverse_vocabs)
    #self.is_training = False
    while iteration < 1000:
        # Now p contains probability of upcoming char, as estimated by model, and s the last RNN state
        #print(len(p.ravel()))
        #print(self.vocab_size)
        ind_sample = np.random.choice(range(0, self.vocab_size), p=p.ravel())
        if iteration == 0:
          ind_sample = self.vocabs["<START>"]
        if self.reverse_vocabs[ind_sample] == "<EOS>": # EOS token
            # print("Model decided to stop generating!")
            break

        current_seq_ind.append(ind_sample)

        # Create feed dict for states
        feed = dict()
        feed[self.keep_prob] = 1.0
        #for i in range(0, self.num_layers):
        #    for c in range(0, len(s[i])):
        #        feed[self.current_states[i]] = s[i]
        #       pass
        feed[self.current_states] = s
        tmp = np.array([ind_sample])
        tmp = np.reshape(tmp, [1,1])
        #print(tmp)
        feed[self.input] = tmp  # Add new input symbol to feed
        [p, s] = sess.run([self.probs, self.state], feed_dict=feed)

        iteration += 1
    self.reverse_vocabs[3] = " "
    out_str = ""
    for c in current_seq_ind:
        out_str += self.reverse_vocabs[c]
    print(out_str)

