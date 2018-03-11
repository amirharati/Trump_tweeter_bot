'''
  A class to convert data to dataset format for tensoflow.
'''


class DataPreppy():
  def __init__(self, input_voc_dict, input_data_file, outdir):
    self.vocabs = {}
    self.reverse_vocabs = {}
    self.input_data = input_data_file
    self.outdir = outdir
    lines = [line for line in open(input_voc_dict)]
    for line in lines:
      parts = line.split()
      # if voc is space
      if (len(parts) == 1):
        self.vocab[" "] = int(parts[0])
        self.reverse_vocabs[int(parts[0])] = ""
      else:
        self.vocab[parts[0]] = int(parts[1])
        self.reverse_vocabs[int(parts[1])] = parts[0]

  def sequence_to_tf_example(self, sequence):
    """convert a sequence (already in final token form) to tf.SequenceExample

    Arguments:
      sequence {Int} -- Input List of tokens

    Returns:
      tf.SequenceExample
    """
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    # +2 For start and end
    ex.context.feature["length"].int64_list.value.append(sequence_length + 2)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_tokens.feature.add().int64_list.value.append(self.vocab["<START>"])
    for token in sequence:
      fl_tokens.feature.add().int64_list.value.append(token)
    fl_tokens.feature.add().int64_list.value.append(self.vocab["<EOS>"])
    return ex

  @staticmethod
  def parse(ex):
    """
      convert a serilized example to a tensor.
    """
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return {"seq": sequence_parsed["tokens"], "length": context_parsed["length"]}

  def save_to_tfrecord(self):
    # since our data is small we read it at once into memory.
    lines = [line for line in open(self.input_data)]
    all_data = []
    for line in lines:
      all_data.append(list(map(int, line[1].split())))

    val = all_data[:1000]
    test = all_data[1000:5000]
    train = all_data[5000:]
    for (data, path) in [(val, './data/val.tfrecord'), (test, './data/test.tfrecord'), (train, './data/train.tfrecord')]:
      with open(path, 'w') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
      for example in data:
        record = BP.sequence_to_tf_example(sequence=example["text"])
        writer.write(record.SerializeToString())
    pass
