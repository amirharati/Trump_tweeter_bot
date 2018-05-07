"""
  Script to generate extended-char set.
  This script generate an extended set that includes all charecters and freqenet words.
  And treat these words as charecters.
  Amir Harati, May 2018
"""


import collections
from itertools import dropwhile


def main():
    inp_text = "data/text_data.txt"
    out_count = "data/trump_mix_count.txt"
    extended_set = "data/echars2id.txt"
    echar_data = "data/echarid_data.txt"

    min_count = 10  # min number of count to add to extended char set

    lines = [line.strip() for line in open(inp_text)]

    chars = set()
    words = collections.Counter()

    for line in lines:
        for c in line:
            chars.add(c)
        words.update(line.split())

    fo = open(out_count, "w")
    for k, v in words.most_common():
        fo.write(k + " " + str(v) + "\n")
    fo.close()

    for key, count in dropwhile(lambda key_count: key_count[1] >= min_count,
         words.most_common()):
            del words[key]
    echars = set()
    for k, v in words.most_common():
        echars.add(k)

    chars = ["<PAD>", "<START>", "<EOS>"] + sorted(list(chars)) + sorted(list(echars.difference(chars)))

    chars_to_ids = {w: id for id, w in enumerate(chars)}
    ids_to_chars = {chars_to_ids[x]: x for x in chars_to_ids}

    fo = open(extended_set, "w")
    for key, val in chars_to_ids.items():
        fo.write(key + "\t" + str(val) + "\n")
    fo.close()

    fo = open(echar_data, "w")

    lines = [line.strip() for line in open(inp_text)]
    for line in lines:
        parts = line.split()
        encoded_line = []
        for p in parts:
            if p in chars_to_ids:
                encoded_line.append(chars_to_ids[p])
            else:
                for c in p:
                    encoded_line.append(chars_to_ids[c])
            encoded_line.append(chars_to_ids[" "])
        for ec in encoded_line:
            fo.write(str(ec) + " ")
        fo.write("\n")
    fo.close()

if __name__ == "__main__":
    main()
