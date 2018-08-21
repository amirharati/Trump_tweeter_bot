# corrnelmovies_preprocessing_gen_pairs.py
# Amir Harati, Aug 2018
"""
    Script to both run the preprocessing on this data and generate the question/answer pairs.
    We use this data to bootstarp the TweetBot model.
    1- First train for 1-2 epoch on corrnelmovies+trumptweets
    2- for few more epochs on Trumptweets.
    3- Use a shared voc. 
"""

import re
import unicodedata

pat = re.compile(".*\[(.*)\]")

def cleanup_pipeline(input_text):
    #temp = input_text.lower()
    #print(input_text)
    temp = str(unicodedata.normalize('NFKD', input_text).encode('ascii', 'ignore'))
    temp = temp.strip("\r\n")
    temp = temp.replace("\n", " ")
    temp = temp.replace("\\n", " ")
    temp = temp.replace("\\t", " ")
    temp = temp.replace("\t", " ")
    temp = re.sub(r'RT\s+.*\:\s+', r'', temp)
    temp = temp.replace("&amp", "")
    temp = temp.replace("&lt", "")
    temp = temp.replace("\"", "")
    temp = temp.replace("\'", "")
    temp = temp.replace("&gt", "")
    temp = temp.replace("-", "")
    temp = temp.replace(".", " ")
    temp = temp.replace("\\", "")
    #temp = temp.replace("    ", " ")
    #temp = temp.replace("   ", " ")
    #temp = temp.replace("  ", "")
    temp = ' '.join(temp.split())
    temp = temp.replace('?', ' ?')
    temp = temp.replace('!', ' !')
    temp = temp.lower()
    return temp[1:]

def create_pairs(lines_dict, conv_dict):
    questions = []
    answers = []
    for key in conv_dict:
        line_keys = conv_dict[key]
        lines = []
        for lk in line_keys:
            lines.append(lines_dict[lk])
        for l1, l2 in zip(lines[:-1], lines[1:]):
            #l2 = lines
            #for l2 in lines[1:]:
                #print(l1)
            if l1[0] != l2[0]:
                    
                    questions.append(l1[1])
                    answers.append(l2[1])
    return questions, answers


def main():
    inp_lines = "./raw_data/cornell_movie_dialogs_corpus/movie_lines.txt"
    inp_conversations = "./raw_data/cornell_movie_dialogs_corpus/movie_conversations.txt"

    lines = [str(line.strip()) for line in open(inp_lines, 'rb')]
    lines_dict = {}
    for line in lines:
        line = line.replace("\'", "")
        line = line.replace("\"", "")

        p = line.split("+++$+++")
        code = p[0].strip()
        
        #code = str(unicodedata.normalize('NFKD', code).encode('ascii', 'ignore'))
        sp1 = p[1].strip()
        text = cleanup_pipeline(p[-1])
        lines_dict[code[1:]] = (str(sp1), text.strip())
    
    lines = [str(line.strip()) for line in open(inp_conversations, 'rb')]
    conv_dict = {}
    count = 0
    for line in lines:
        line = line.replace("\'", "")
        line = line.replace("\"", "")
        r = pat.match(line)
        if r is not None:
            l = r.group(1)
            p = l.split(",")
            ll = []
            for x in p:
                x = x.replace("\'", "")
                ll.append(x.strip())
            conv_dict[count] = ll
            count += 1

    #print(lines_dict)
    q, a = create_pairs(lines_dict, conv_dict)
    
    with open("data/corrnel_movies_questions.txt", "w") as fo:
        for l in q:
            fo.write(l + '\n')

    with open("data/corrnel_movies_answers.txt", "w") as fo:
        for l in a:
            fo.write(l + '\n')


if __name__ == "__main__":
    main()