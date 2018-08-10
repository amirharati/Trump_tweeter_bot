# File: generate_qa_pairs.py
# Amir Harati, July 2018
"""
    Generate pairs to train seq2seq model.
    The idea is to have these pairs:
    1- identical: the same as tweet/utterance.
    2- One Hot word: Selected after removing stop words randomly from an input list of words.
    3- Two and Three hot words. (later)
    ** Notice for encoder we only use words with several examples. Everything else will be mapped to  <UNK>.
    ** we aso need to have several combination of <UNK>/hot words as input.
    ** For decoder we used extended echar that includes all words from encoder and part of words and chars.
    ** This hopefully allows to generate meaningful responses.
  
"""

from nltk.corpus import stopwords
import random
import re
import sys

class PairGen:
    def __init__(self, freq_terms_file, input_file, output_file1, output_file2):
        self.stop_words = set(stopwords.words('english'))
        self.freq_terms = []
        lines = [line.strip() for line in open(freq_terms_file)]
        for line in lines:
            parts = line.split()
            s = " ".join(parts[0:-1])
            self.freq_terms.append(s)

        self.input_data = [line.strip() for line in open(input_file)]
        self.output_file1 = output_file1
        self.output_file2 = output_file2
    
    def get_words(self, inp):
        """
            apply different filters and return remaining words.
            return two lists: 1- just filtered words, 2- filtered words + removed non-stopwords replaced with <UNK> 
        """
        inp_words = inp.split()
        # filter stop words
        nonstops = [w for w in inp_words if not w.lower() in self.stop_words]
        # filter not in list
        filtered_freq_words = [w for w in nonstops if w in self.freq_terms]
        # replaced with <UNK> if not in freq list
        filtered_freq_words_with_unk = []
        for w in nonstops:
            if w in self.freq_terms:
                filtered_freq_words_with_unk.append(w)
            else:
                filtered_freq_words_with_unk.append("<UNK>")

        return filtered_freq_words, filtered_freq_words_with_unk

    def generate_pairs(self):
        """
            generate pairs, optionally add inputs with <UNK> inlcuded.
        """
        # loop over all data points
        pairs = []
        for data in self.input_data:
            filtered, filtered_unk = self.get_words(data)
            
            if len(filtered) > 1:
                #print(filtered)
                onehot_1, onehot_2 = random.sample(filtered, 2)
            else:
                if len(filtered) == 1:
                    onehot_1 = random.sample(filtered, 1)
                else:
                    onehot_1 = None
                onehot_2 = None

            #if len(filterd) > 2:
            #    twohot = random.sample(filtered, 2)

            pairs.append((["<UNK>"], data))

            if onehot_1 is not None:
                pairs.append((onehot_1, data)) 
            
            if onehot_2 is not None:
                pairs.append((onehot_2, data))
            
            # also append with whole sentence after removing all stopwords and replacing non-freq words with <UNK>
            pairs.append((filtered_unk, data))

        
        # now loop over freq terms and add if term for n>1 is in the data
        # TODO: fix the error
        
        for data  in self.input_data:
            count = 0
            for term in self.freq_terms:
                
                # we cant have too many examples
                if count > 10:
                    break
        

                if (len(term.split())>1):
                    print(term)
                    p = re.compile(".*" + term.lower() + ".*")
                    if p.match(data.lower()) is not None:
                        fterm = []
                        ucount = 0
                        for x in term.lower().split():
                            if x in self.stop_words:
                                fterm.append("<UNK>")
                                ucount += 1
                            else:
                                fterm.append(x)
                        if ucount < 2:
                            count += 1
                            fterm_str = " ".join(fterm)
                            pairs.append((fterm_str, data))
                

        return pairs    


    
    def generate_dataset(self):
        """
            generate the full dataset.
        """
        pairs = self.generate_pairs()
        fo1 = open(self.output_file1, "w")
        fo2 = open(self.output_file2, "w")

        for p in pairs:
            p_str = "".join(p[0])
            fo1.write(p_str + "\n")
            fo2.write(p[1] + "\n")
        fo1.close()
        fo2.close()

def main(freq_terms_file, input_file, output_file1, output_file2):
    PG = PairGen(freq_terms_file, input_file, output_file1, output_file2)
    PG.generate_dataset()


if __name__ == "__main__":
    freq_terms_file = sys.argv[1]
    input_file = sys.argv[2]
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]
    main(freq_terms_file, input_file, output_file1, output_file2)
