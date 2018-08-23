# File: ChatBot.py
# Amir Harati, Aug 2018
"""
    Trump Chatbot model using seq2seq model.

    TODO:
    1- add utilities for pbt <-> text, generator, and print.
    2- add randomizer for both train and inference so it generates diff. things with same input.(e.g. add random noise): add dropout in prediction does the trick
    but is there anything better to do?
    3- remvoe <START>/<EOS> for input string.
    4- make sure it actually works correctly (response seems a little bit random)
    5- use attention.:  Use too much mem. Check see if we can make it work : after more training looks better
    6- correct the input (add the dot in answers but not to question, comma either remove or make one word, remove single words cases and (), remove @xxxx: cases. )
    7- reduce the number of <UNK> both for questions (perhaps max one in a question but during training allow different places)
    8- add train example with  removed <UNK> all together
    9- use some normal  (but large) ENglish text to pretrain the models for a while. perhaps use dialoug like move subs? This help to improve
    generalization somewhat and model learn better English and will be able to go beyond trump tweets
    10- create a script for all steps of data prep.: sort of done two pipelines
    https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
    perhaps we only trian on some specfic charecters? so it become a hybrid trump bot (think of some funny/crazy charecters)
    11- perhaps we can allow the stop words just remove very rare words from input and make them <UNK>
    12- make encoder bidirect

   
    
    seq2seq/attention
    https://towardsdatascience.com/memory-attention-sequences-37456d271992
    https://tutorials.botsfloor.com/how-to-build-your-first-chatbot-c84495d4622d
    https://arxiv.org/pdf/1703.03906v2.pdf
    https://distill.pub/2016/augmented-rnns/
    https://arxiv.org/pdf/1610.06258.pdf

    serving:
    https://towardsdatascience.com/serving-tensorflow-models-serverless-6a39614094ff
    https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/
    https://medium.com/tooso/serving-tensorflow-predictions-with-python-and-aws-lambda-facb4ab87ddd
    use cloud ml engine?
    https://github.com/GoogleCloudPlatform/training-data-analyst  check the  training-data-analyst/courses/machine_learning/cloudmle


"""

import Seq2SeqDataPreppy as SDP
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
import logging
from tensorflow.contrib import layers



class TwitterBot:
    def __init__(self, model_size, embedding_size, num_layers,
         keep_prob, vocabs, reverse_vocabs, en_bpe_dict, batch_size, num_itr, train_tfrecords, eval_tfrecords,
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

        self.en_bpe_dict = {}
        lines = [line.strip() for line in open(en_bpe_dict)]
        for line in lines:
            parts = line.split()
            self.en_bpe_dict[parts[0]] = parts[1].split()

    def bpe_to_en(self, inp):
        return inp.replace("@@ ", "")

    def train(self):
        """
            train the custom estimator.
        """
        print_logs = tf.train.LoggingTensorHook(
        ['train_input', 'train_pred',  "predictions"], every_n_iter=100,
             formatter=self.get_formatter(['train_input', 'train_pred', 'predictions'], self.vocabs))
        est = tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.model_dir,
            params=self.params)
        est.train(self._train_input_fn, hooks=[print_logs], steps=self.num_itr)
    
    def get_formatter(self, keys, vocab):

        def to_str(sequence):
            tokens = [self.reverse_vocabs[x] for x in sequence]
            return self.bpe_to_en(' '.join(tokens))

        def format(values):
            res = []
            for key in keys:
                res.append("%s = %s" % (key, to_str(values[key]) ))
            return '\n'.join(res)
        return format
    
    def generate(self, question_str):
     
        question_str = question_str.lower()

        model_size = self.params["model_size"]
        
        num_layers = self.params["num_layers"]
        est = tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.model_dir,
            params=self.params,
            warm_start_from=self.model_dir)
        
        question_wordids = [self.vocabs["<START>"]]
        word_seq = question_str.split()
        for w in word_seq:
            if w in self.en_bpe_dict:
                bpet = self.en_bpe_dict[w]
            else:
                bpet = ["<UNK>"]
            for b in bpet:
                question_wordids.append(self.vocabs[b])
        question_wordids.append(self.vocabs["<EOS>"])

    
        current_seq_ind = []
        # dummy seq.
        X = np.zeros((1, len(question_wordids)), dtype=np.int32)
        X[0, :] = 1
        Y = np.zeros((1, len(question_wordids)), dtype=np.int32)
        Y[0, :] = question_wordids
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"answer_seq": X, "question_seq": Y},
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
        out_str = self.bpe_to_en(out_str)
        out_str2 = self.bpe_to_en(out_str2)

        print("argmax: ", out_str2)
        print("sampling: ", out_str)
    

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
        fcells = []
        for i in range(num_layers):
            c = tf.nn.rnn_cell.GRUCell(model_size)
            c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=keep_prob,
                                            output_keep_prob=keep_prob)
            fcells.append(c)
        # I cant figure out how to use tuple version.    
        fcell = tf.nn.rnn_cell.MultiRNNCell(fcells)

        #bcells = []
        #for i in range(num_layers):
        #    c = tf.nn.rnn_cell.GRUCell(model_size)
        #    c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=keep_prob,
        #                                    output_keep_prob=keep_prob)
        #    bcells.append(c)
        # I cant figure out how to use tuple version.    
        #bcell = tf.nn.rnn_cell.MultiRNNCell(bcells)

        bcell = tf.contrib.rnn.GRUCell(num_units=model_size)

        #icell = tf.contrib.rnn.GRUCell(num_units=model_size)
        encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(fcell, bcell, question_embed, sequence_length=question_lengths, dtype=tf.float32)



        # helpers
        train_helper = tf.contrib.seq2seq.TrainingHelper(answer_embed, answer_lengths, time_major=False)
        start_tokens = tf.tile(tf.constant([self.vocabs['<START>']], dtype=tf.int32), [batch_size], name='start_tokens')
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, 
                    start_tokens=start_tokens, end_token=self.vocabs["<EOS>"])

        # rnn cell and dense layer
        cell = tf.contrib.rnn.GRUCell(num_units=model_size)
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
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=model_size, memory=encoder_outputs[0],
                    memory_sequence_length=question_lengths)
                #cell = tf.contrib.rnn.GRUCell(num_units=model_size)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=model_size)
                #out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                #    attn_cell, vocab_size, reuse=reuse
                #)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=attn_cell, helper=helper,
                    initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                    #initial_state=encoder_final_state,
                    output_layer=projection_layer
                    )
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
        tf.identity(question_sequence[0], name="train_input")
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
    dpp = SDP.Seq2SeqDataPreppy("trump_corrnel_qa_word", "./data/trump_corrnel_words2ids.txt", "./data/trump_corrnel_wordid_questions.txt",
     "./data/trump_corrnel_wordid_answers.txt", "./data")
    m = TwitterBot(model_size=256, embedding_size=100, num_layers=2,
         keep_prob=.98, batch_size=32, num_itr=100, vocabs=dpp.vocabs, reverse_vocabs=dpp.reverse_vocabs,
         en_bpe_dict="./data/trump_corrnel_questions_en_bpe.dict", 
         train_tfrecords='./data/trump_tweet_qa_word-train.tfrecord',
         eval_tfrecords='./data/trump_tweet_qa_word-val.tfrecord',
         model_dir="./checkpoints")
    print(dpp.vocabs)
    print(len(dpp.vocabs))
    m.train()
    m.generate("iran deal")

if __name__ == "__main__":
    test()