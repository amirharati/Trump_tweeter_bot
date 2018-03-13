"""
 Trainer script for char-lm model.
"""

import char_lm_model as clm
import datapreppy as dp
import tensorflow as tf

model_size = 64
num_layers = 1


def main():
  with tf.Graph().as_default():
    gs = tf.train.get_or_create_global_step()
    dpp = dp.DataPreppy("char", "./data/chars2id.txt", "", "")
    next_element, training_init_op, _, _ = dpp.prepare_dataset_iterators("char", batch_size=128)

    train_writer = tf.summary.FileWriter("./logs/train")

    M = clm.CharLmModel(next_element, dpp.vocabs, num_layers, model_size, '')
    summary_op = tf.summary.merge_all()
    with tf.train.MonitoredTrainingSession(checkpoint_dir="./chkpoint",
                                          save_summaries_steps=None,
                                           save_summaries_secs=None, ) as sess:

      for epoch in range(1000):
        # inititilize the iterator to consume data
        sess.run(training_init_op)
        while True:
          try:
            [res_loss, _, res_global_step, summary] = \
                sess.run([M.loss, M.train_op, M.global_step, summary_op],
                         feed_dict={M.keep_prob: 0.8})
            if res_global_step % 100 == 0:
              print("loss: ", res_loss)
            train_writer.add_summary(summary,
                                       global_step=int(res_global_step))

          except tf.errors.OutOfRangeError:
            break

if __name__ == "__main__":
  main()
