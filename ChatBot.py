# File: ChatBot.py
# Amir Harati, Aug 2018
"""
    Trump Chatbot model using seq2seq model.
"""

import Seq2SeqDataPreppy as SDP
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
import logging
from tensorflow.contrib import layers



class ChatBot:
    def __init__(self, model_size, embedding_size, num_layers,
         keep_prob, vocabs, reverse_vocabs, batch_size, num_itr, train_tfrecords, eval_tfrecords,
         model_dir):
        self.params = dict()
        self.params["model_size"] = model_size
        self.params["embedding_size"] = embedding_size
        self.params["num_layers"] = num_layers
        self.params["keep_prob"] = keep_prob
        self.vocab_size = len(vocabs)
        self.params["vocab_size"] = self.vocab_size

        self.batch_size = batch_size
        self.num_itr = num_itr
        self.train_tfrecords = train_tfrecords
        self.eval_tfrecords = eval_tfrecords
        self.model_dir = model_dir
        self.vocabs = vocabs
        self.reverse_vocabs = reverse_vocabs

    def train(self):
        """
            train the custom estimator.
        """
        print_logs = tf.train.LoggingTensorHook(
        ['train_pred', "predictions"], every_n_iter=100,
             formatter=self.get_formatter(['train_pred', 'predictions'], self.vocabs))
        est = tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.model_dir,
            params=self.params)
        est.train(self._train_input_fn, hooks=[print_logs], steps=self.num_itr)
    
    def get_formatter(self, keys, vocab):

        def to_str(sequence):
            tokens = [self.reverse_vocabs[x] for x in sequence]
            return ' '.join(tokens)

        def format(values):
            res = []
            for key in keys:
                res.append("%s = %s" % (key, to_str(values[key]) ))
            return '\n'.join(res)
        return format
    """
    def generate(self):
     
        model_size = self.params["model_size"]
        
        num_layers = self.params["num_layers"]
        est = tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.model_dir,
            params=self.params,
            warm_start_from=self.model_dir)
        
        current_seq_ind = []
        # dummy seq.
        X = np.zeros((1, 1), dtype=np.int32)
        X[0, 0] = 1
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"seq": X},
            num_epochs=1,
            shuffle=False)
            
        result = est.predict(input_fn=predict_input_fn)
        #print(next(result))
        g = next(result)
        probs = g["probs"]
        syms = g["syms"]
        #print(len(probs))
        

        for p in probs:
            ind_sample = np.random.choice(range(0, self.vocab_size), p=p.ravel())
            #ind_sample = np.argmax(p.ravel())
            current_seq_ind.append(ind_sample)

        self.reverse_vocabs[3] = " "
        out_str = ""
        out_str2 = ""
        #print(syms)
        #print(current_seq_ind)
        for c in current_seq_ind:
            out_str += self.reverse_vocabs[c] + " " 
        for c in syms:
            out_str2 += self.reverse_vocabs[c] + " "
        print("argmax: ", out_str2)
        print("sampling: ", out_str)
    """

    def _train_input_fn(self):
        return SDP.Seq2SeqDataPreppy.make_dataset(self.train_tfrecords, "train", self.vocabs["<PAD>"], self.batch_size)

    def _eval_input_fn(self):
        return SDP.Seq2SeqDataPreppy.make_dataset(self.eval_tfrecords, "eval", self.vocabs["<PAD>"], self.batch_size)


    def _model(self, features, labels, mode, params):
        """
            main model.
        """
        question_sequence = features['question_seq']
        answer_sequence = features['answer_seq']

        batch_size = tf.shape(question_sequence)[0]
        start_token = tf.ones([1], tf.int32) 
    
        model_size = params["model_size"]
        num_layers = params["num_layers"]
        keep_prob = params["keep_prob"]
        vocab_size = params["vocab_size"]
        embedding_size = params["embedding_size"]


        question_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(question_sequence, self.vocabs["<PAD>"])), 1)
        answer_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(answer_sequence, self.vocabs["<PAD>"])), 1)

        question_embed = layers.embed_sequence(
            question_sequence, vocab_size=vocab_size, embed_dim=embedding_size , scope='embed')
        answer_embed = layers.embed_sequence(
            answer_sequence, vocab_size=vocab_size, embed_dim=embedding_size, scope='embed', reuse=True)
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')
        icells = []
        for i in range(num_layers):
            c = tf.nn.rnn_cell.GRUCell(model_size)
            icells.append(c)
        # I cant figure out how to use tuple version.    
        icell = tf.nn.rnn_cell.MultiRNNCell(icells)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(icell, question_embed, dtype=tf.float32)

        # helpers
        train_helper = tf.contrib.seq2seq.TrainingHelper(answer_embed, answer_lengths, time_major=False)
        start_tokens = tf.tile(tf.constant([self.vocabs['<START>']], dtype=tf.int32), [batch_size], name='start_tokens')
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, 
                    start_tokens=start_tokens, end_token=self.vocabs["<EOS>"])

        # rnn cell and dense layer
        #cell = tf.contrib.rnn.GRUCell(num_units=model_size)
        cells = []
        for i in range(num_layers):
            c = tf.nn.rnn_cell.GRUCell(model_size)
            c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=keep_prob,
                                            output_keep_prob=keep_prob)
            cells.append(c)
        # I cant figure out how to use tuple version.    
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        projection_layer = Dense(units=vocab_size,
         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

        # deocder in seq2seq model. For this case we don't have an encoder.
        def decode(helper, scope, output_max_length,reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=cell, helper=helper,
                    #initial_state=cell.zero_state(
                    #    dtype=tf.float32, batch_size=batch_size),
                        initial_state=encoder_final_state,
                        output_layer=projection_layer)
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=output_max_length
                )
            return outputs[0]
        train_outputs = decode(train_helper, 'decode', 3000)
        pred_outputs = decode(pred_helper, 'decode', 300, reuse=True)


        targets = answer_sequence[:, 1:]

        probs = tf.nn.softmax(pred_outputs.rnn_output, name="probs")
        # in case in prediction mode return
        if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"probs": probs, "syms": pred_outputs.sample_id})

        # mask the PADs    
        mask = tf.to_float(tf.not_equal(answer_sequence[:, :-1], self.vocabs["<PAD>"]))

        #tf.identity(mask[0], name='mask')
        #tf.identity(targets[0], name='targets')
        #tf.identity(train_outputs.rnn_output[0,output_lengths[0]-2:output_lengths[0],:], name='rnn_out')
        # Loss function
        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output[:,:-1,:],
            targets,
            mask)
        tf.summary.scalar("loss", loss)

        # Optimizer
        learning_rate = tf.Variable(0.0, trainable=False)
        initial_learning_rate = tf.constant(0.001)
        learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                    tf.train.get_global_step(), 100, 0.99)
        tf.summary.scalar("learning_rate", learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Visualise gradients
        vis_grads = [0 if i is None else i for i in grads]
        for g in vis_grads:
            tf.summary.histogram("gradients_" + str(g), g)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.train.get_global_step())
        
        tf.identity(train_outputs.sample_id[0], name='train_pred')
        tf.identity(pred_outputs.sample_id[0], name='predictions')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=loss,
            train_op=train_op
        )

def test():
    tf.logging.set_verbosity(logging.DEBUG)
    dpp = SDP.Seq2SeqDataPreppy("qa_word", "./data/qa_word2id.txt", "./data/qa_wordid_questions.txt", "./data/qa_wordid_answers.txt", "./data")
    m = ChatBot(model_size=512, embedding_size=100, num_layers=2,
         keep_prob=1.0, batch_size=32, num_itr=500, vocabs=dpp.vocabs, reverse_vocabs=dpp.reverse_vocabs,
         train_tfrecords='./data/qa_word-train.tfrecord',
         eval_tfrecords='./data/qa_word-val.tfrecord',
         model_dir="./checkpoints")
    print(dpp.vocabs)
    print(len(dpp.vocabs))
    m.train()
    #m.generate()

if __name__ == "__main__":
    test()