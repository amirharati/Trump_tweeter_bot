"""
 Trainer script for char-lm model.
"""

import char_lm_model as clm
import datapreppy as dp
import tensorflow as tf

model_size = 256
num_layers = 1


def main():
  with tf.Graph().as_default():
    #gs = tf.train.get_or_create_global_step()
    dpp = dp.DataPreppy("char", "./data/chars2id.txt", "", "")
    next_element, training_init_op, _, _ = dpp.prepare_dataset_iterators("char", batch_size=16)

    train_writer = tf.summary.FileWriter("./logs/train")

    M = clm.CharLmModel(next_element, dpp.vocabs, dpp.reverse_vocabs, num_layers, model_size, '')
    summary_op = tf.summary.merge_all()
    #with tf.train.MonitoredTrainingSession(checkpoint_dir="./chkpoint",
    #                                      save_summaries_steps=None,
    #                                       save_summaries_secs=None, ) as sess:
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      for epoch in range(1000):
        # inititilize the iterator to consume data
        sess.run(training_init_op)
        # a = sess.run(M.aaa, feed_dict={M.keep_prob: 0.8})
        #print(a)
        while True:
          try:
            [res_loss, _, res_global_step, summary] = \
                sess.run([M.loss, M.train_op, M.global_step, summary_op],
                         feed_dict={M.keep_prob: 0.8})


            #print(a)
            if res_global_step % 100 == 0:
              print("loss: ", res_loss)
              """a1, a3 = sess.run([M.aaa1, M.aaa3], feed_dict={M.keep_prob: 0.8})
              tmp_str = ""
              for x in a1[0]:
                #print(x)
                tmp_str += dpp.reverse_vocabs[x]
              print("str1: ", tmp_str)
              tmp_str = ""
              for x in a3[0]:
                #print(x)
                tmp_str += dpp.reverse_vocabs[x]
              print("str2: ", tmp_str)
              """
              #sess.run(M.aaa2, feed_dict={M.keep_prob: 0.8})
              #sess.run(M.aaa3, feed_dict={M.keep_prob: 0.8})
            train_writer.add_summary(summary,
                                       global_step=int(res_global_step))

            # sample the model
            if res_global_step % 1000 == 0:
              M.sample(sess)

          except tf.errors.OutOfRangeError:
            print("all data consumed.")
            break

if __name__ == "__main__":
  main()
