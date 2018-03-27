"""
 Trainer script for char-lm model.
"""

import word_lm_model as wlm
import datapreppy as dp
import tensorflow as tf

model_size = 256
num_layers = 1
batch_size = 64
embedding_size = 10
checkpoints_dir = "./chkpoints"


def main():
  with tf.Graph().as_default():
    dpp = dp.DataPreppy("char", "./data/word2id.txt", "", "")
    next_element, training_init_op, _, _ = \
      dpp.prepare_dataset_iterators("word", batch_size=batch_size)

    train_writer = tf.summary.FileWriter("./logs/train")

    M = wlm.WordLmModel(next_element, dpp.vocabs, dpp.reverse_vocabs, num_layers, model_size, embedding_size)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      saver = tf.train.Saver(tf.global_variables(),
                             write_version=tf.train.SaverDef.V2)

      latestCheckpoint = tf.train.latest_checkpoint(checkpoints_dir)
      if latestCheckpoint is not None:
        restorer = tf.train.Saver(tf.global_variables(),
                                  write_version=tf.train.SaverDef.V2)
        restorer.restore(sess, latestCheckpoint)
        print('Pre-trained model restored')
      for epoch in range(1000):
        # inititilize the iterator to consume data
        sess.run(training_init_op)
        while True:
          try:
            [res_loss, _, res_global_step, summary] = \
                sess.run([M.loss, M.train_op, M.global_step, summary_op],
                         feed_dict={M.keep_prob: 1.0})

            if res_global_step % 100 == 0:
              print("loss: ", res_loss)
            train_writer.add_summary(summary,
                                     global_step=int(res_global_step))

            # sample the model
            if res_global_step % 100 == 0:
              print("Saving model...")
              saver.save(sess, checkpoints_dir + "/model",
                         global_step=int(res_global_step))
              M.sample(sess)

          except tf.errors.OutOfRangeError:
            print("all data consumed.")
            break

if __name__ == "__main__":
  main()
