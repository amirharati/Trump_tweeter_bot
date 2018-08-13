# Seq2SeqDataPreppy.py
'''
  A class to convert data to dataset format for tensoflow.
  This is a version of calss for Seq2Seq (chatbot in this case) data.
  ref:
  https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
  https://cs230-stanford.github.io/tensorflow-input-data.html
  https://github.com/tensorflow/nmt/#data-input-pipeline
'''
import tensorflow as tf
from random import shuffle
import numpy as np


def expand(x):
    '''
    Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar  to a vector of length 1
    :param x:
    :return:
    '''
    x['question_length'] = tf.expand_dims(tf.convert_to_tensor(x['question_length']), 0)
    x['answer_length'] = tf.expand_dims(tf.convert_to_tensor(x['answer_length']), 0)
    return x


def deflate(x):
    '''
        Undo Hack. We undo the expansion we did in expand
    '''
    x['question_length'] = tf.squeeze(x['question_length'])
    x['answer_length'] = tf.squeeze(x['answer_length'])
    
    return x


class Seq2SeqDataPreppy():
  def __init__(self, prefix, input_voc_dict, input_data_questions, input_data_answers, outdir):
    self.vocabs = {}
    self.reverse_vocabs = {}
    self.prefix = prefix
    self.input_data_questions = input_data_questions
    self.input_data_answers = input_data_answers
    self.outdir = outdir
    lines = [line for line in open(input_voc_dict)]
    for line in lines:
      parts = line.split()
      # if voc is space
      if (len(parts) == 1):
        print(line)
        self.vocabs[" "] = int(parts[0])
        self.reverse_vocabs[int(parts[0])] = ""
      else:
        self.vocabs[parts[0]] = int(parts[1])
        self.reverse_vocabs[int(parts[1])] = parts[0]

  def sequence_to_tf_example(self, sequence):
    """convert a sequence (already in final token form) to tf.SequenceExample

    Arguments:
      sequence {Int} -- Input List of tokens

    Returns:
      tf.SequenceExample
    """
    #ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    question = [self.vocabs["<START>"]] + sequence[0] + [self.vocabs["<EOS>"]]
    answer = [self.vocabs["<START>"]] + sequence[1] + [self.vocabs["<EOS>"]]

    question_length = len(question)
    answer_length = len(answer)

    context = tf.train.Features(feature={
    'question_length': tf.train.Feature(int64_list=tf.train.Int64List(
        value=[question_length])),
    'answer_length': tf.train.Feature(int64_list=tf.train.Int64List(
        value=[answer_length])),
        })
    
    # Create sequence data
    question_features = []
    answer_features = []
   
    for  q in question:
        # Create each of the features, then add it to the
        # corresponding feature list
        q_feat = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[q]))
        question_features.append(q_feat)
    
    for  a in answer:
        # Create each of the features, then add it to the
        # corresponding feature list
        a_feat = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[a]))
        answer_features.append(a_feat)


    question_seq = tf.train.FeatureList(feature=question_features)
    answer_seq = tf.train.FeatureList(feature=answer_features)

    
    features = tf.train.FeatureLists(feature_list={
        'question_seq': question_seq,
        'answer_seq': answer_seq,
    })

  
    ex = tf.train.SequenceExample(context=context,
                                   feature_lists=features)
    #print(ex)
    return ex

  @staticmethod
  def parse(ex):
    """
      convert a serilized example to a tensor.
    """
    context_features = {
        "question_length": tf.FixedLenFeature([], dtype=tf.int64),
        "answer_length": tf.FixedLenFeature([], dtype=tf.int64),

    }
    sequence_features = {
        "question_seq": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "answer_seq": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return {"question_seq": sequence_parsed["question_seq"], "answer_seq": sequence_parsed["answer_seq"],
         "question_length": context_parsed["question_length"], "answer_length": context_parsed["answer_length"]}

  def save_to_tfrecord(self):
    # since our data is small we read it at once into memory.
    lines1 = [line for line in open(self.input_data_questions)]
    lines2 = [line for line in open(self.input_data_answers)]

    all_data = []
    for line1, line2 in zip(lines1, lines2):
      all_data.append((list(map(int, line1.split())), list(map(int, line2.split()))))

    # suffle
    shuffle(all_data)

    val = all_data[:1000]
    test = all_data[1000:2000]
    # use all data for train
    train = all_data
    print("trian size:", len(train))
    for (data, path) in [(val, './data/' + self.prefix + '-val.tfrecord'), (test, './data/' + self.prefix + '-test.tfrecord'), (train, './data/' + self.prefix + '-train.tfrecord')]:
      with open(path, 'w') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
      for example in data:
        # print(example)
        record = self.sequence_to_tf_example(sequence=example)
        writer.write(record.SerializeToString())
    

  @staticmethod
  def make_dataset(path, mode, pad_sym, batch_size=128):
    '''
      Makes  a Tensorflow dataset that is shuffled, batched and parsed according to BibPreppy.
      You can chain all the lines here, I split them into seperate calls so I could comment easily
      :param path: The path to a tf record file
      :param path: The size of our batch
      :return: a Dataset that shuffles and is padded
    '''
    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset([path])
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset = dataset.map(Seq2SeqDataPreppy.parse, num_parallel_calls=8)
    #Shuffle the dataset
    if mode == "trian":
        dataset = dataset.shuffle(buffer_size=10000)
    #In order the pad the dataset, I had to use this hack to expand scalars to vectors.
    dataset = dataset.map(expand)
    # Batch the dataset so that we get batch_size examples in each batch.
    # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "question_length": 1,
        "question_seq": tf.TensorShape([None]),
        "answer_length": 1,
        "answer_seq": tf.TensorShape([None])
    # but the seqeunce is variable length, we pass that information to TF
    },
    padding_values={
        "question_length": np.int64(0),
        "question_seq":  np.int64(pad_sym),
        "answer_length":  np.int64(0),
        "answer_seq":  np.int64(pad_sym)
    })

    if mode == "train":  
      # unlimited repeats
      dataset = dataset.repeat()

    #Finnaly, we need to undo that hack from the expand function
    dataset = dataset.map(deflate)
    #return dataset
    """
      here we can return the dataset; however in https://www.tensorflow.org/programmers_guide/datasets
      it is recommended to return iterator.
      ###return dataset
    """  
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features 


  """  
  @staticmethod
  def prepare_dataset_iterators(prefix, batch_size=128):
      # Make a dataset from the train data
      train_ds = DataPreppy.make_dataset('./data/' + prefix + '-train.tfrecord', batch_size=batch_size)
      # make a dataset from the valdiation data
      val_ds = DataPreppy.make_dataset('./data/' + prefix + '-val.tfrecord',batch_size=batch_size)
      test_ds = DataPreppy.make_dataset('./data/' + prefix + '-test.tfrecord',batch_size=batch_size)

      # Define an abstract iterator
      # Make an iterator object that has the shape and type of our datasets
      iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                 train_ds.output_shapes)

      # This is an op that gets the next element from the iterator
      next_element = iterator.get_next()
      # These ops let us switch and reinitialize every time we finish an epoch
      training_init_op = iterator.make_initializer(train_ds)
      validation_init_op = iterator.make_initializer(val_ds)
      test_init_op = iterator.make_initializer(test_ds)

      return next_element, training_init_op, validation_init_op, test_init_op
  """