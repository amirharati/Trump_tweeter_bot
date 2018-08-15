# create_english_bpe_dict.py
# Amir Harati Aug 2018
"""
    Create a dictionary to convert words between bpe and normal English format.
    This use a converted file and generate a dict from it.
"""

import  re
import sys

def create_dict(inp):
    lines = [line.strip() for line in open(inp)]
    pat = re.compile("(.*)@@")
    dictionary = {}
    for line in lines:
        parts = line.split()
        partial = ""
        val = []
        for p in parts:
            r = pat.match(p)
            if r is not None:
                partial += r.group(1)
                val.append(p)
            else:
                if partial != "":
                    partial += p
                    val.append(p)
                    dictionary[partial] = " ".join(val)
                    partial = ""
                    val = []
                else:
                    dictionary[p] = p  # map to itself
    return dictionary


def main(input_file):
    d = create_dict(input_file)
    for k in d:
        print(k + " " + d[k])


if __name__ == "__main__":
    main(sys.argv[1])



