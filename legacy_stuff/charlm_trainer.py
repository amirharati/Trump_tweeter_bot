"""
 Trainer script for char-lm model.
"""

import char_lm_model as clm
import datapreppy as dp
import tensorflow as tf

model_size = 512
num_layers = 1
batch_size = 128
checkpoints_dir = "./chkpoints"
embedding_size = 100

# CRASH REASON:  SOME LINES ARE tooo  long check line 41336

def main():
  #config = tf.ConfigProto(
  #      device_count = {'GPU': 0}
  #)
  run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
  with tf.Graph().as_default():
    dpp = dp.DataPreppy("echar", "./data/echars2id.txt", "", "")
    next_element, training_init_op, _, _ = \
      dpp.prepare_dataset_iterators("echar", batch_size=batch_size)

    train_writer = tf.summary.FileWriter("./logs/train")

    M = clm.CharLmModel(next_element, dpp.vocabs, dpp.reverse_vocabs, num_layers, model_size, embedding_size)
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
                         feed_dict={M.keep_prob: 1.0}, options=run_options)

            if res_global_step % 100 == 0:
              print("loss: ", res_loss)
            train_writer.add_summary(summary,
                                     global_step=int(res_global_step))

            # sample the model
            if res_global_step % 1000 == 0:
              print("Saving model...")
              saver.save(sess, checkpoints_dir + "/model",
                         global_step=int(res_global_step))
              M.sample(sess)

          except tf.errors.OutOfRangeError:
            print("all data consumed.")
            break

if __name__ == "__main__":
  main()
