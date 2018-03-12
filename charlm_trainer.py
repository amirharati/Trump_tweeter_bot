"""
 Trainer script for char-lm model.
"""

import char_lm_model as clm
import datapreppy as dp
import tensorflow as tf

model_size = 256


def main():
  with tf.Graph().as_default():
    gs = tf.train.get_or_create_global_step()
    dpp = dp.DataPreppy("char", "./data/chars2id.txt", "", "")
    next_element, training_init_op, _, _ = dpp.prepare_dataset_iterators("char", batch_size=128)

    train_writer = tf.summary.FileWriter("./logs/train")

    M = clm.CharLmModel(next_element, dpp.vocabs, model_size, gs)
    summary_op = tf.summary.merge_all()
    with tf.train.MonitoredTrainingSession(checkpoint_dir="./chkpoint",
                                          save_summaries_steps=None,
                                           save_summaries_secs=None, ) as sess:
    #with tf.Session() as sess:
      #init_op = tf.global_variables_initializer()
      #sess.run(init_op)

      for epoch in range(1000):
        # inititilize the iterator to consume data
        sess.run(training_init_op)
        while True:
          try:
            [res_loss, _, res_global_step, summary] = \
                sess.run([M.loss, M.train_op, M.global_step, summary_op])
            #res_loss, _, _, _ =sess.run([M.loss, M.train_op, M.global_step, summary_op])
            sess.run([M.increment_gs])
            x=sess.run(M.OOO)
            print("xx:",x)
            if res_global_step % 100 == 0:
              print("loss: ", res_loss)
              # print("prepL", np.exp(costs/lens))
            train_writer.add_summary(summary,
                                       global_step=int(res_global_step))

          except tf.errors.OutOfRangeError:
            break

if __name__ == "__main__":
  main()
