import sys
import math
model_filename = 'data/big-ws-model.txt' #'01hw_model_ja'
sep = u'\t' #u'|'
def load_model():
    model = {}
    with open(model_filename,'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            line = unicode(line.strip(), 'utf-8')
            t = line.split(sep)
            if len(t) == 2:
                model[t[0]] = float(t[1])
    return model
model = load_model()
V = 1000000
lambda1 = 0.95
lambda_unk = 1 - lambda1

def get_unigram_prob(w):
    P = lambda_unk/V
    if w in model:    
        P += lambda1*model[w]
    return P

#filename = 'test/04-input.txt'
filename = 'data/wiki-ja-test.txt'
with open(filename, 'r') as f:
    lines = f.read().split('\n')
best_edge = {}
best_score = {}

for line in lines:
    if len(line) == 0:
        continue
    line = unicode(line.strip(), 'utf-8')
    best_edge[0] = None
    best_score[0] = 0
    for word_end in range(1, len(line)+1):
        best_score[word_end] = 1e10
        for word_start in range(len(line)):
            word = line[word_start:word_end]
            if word in model or len(word) == 1:
                prob = get_unigram_prob(word)
                my_score = best_score[word_start] + (-math.log(prob, 2))
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_start, word_end)

    words=[]
    next_edge = best_edge[len(best_edge) - 1]
    while next_edge != None:
        word = line[next_edge[0]:next_edge[1]]
        words.append(word.encode('utf-8'))
        next_edge = best_edge[next_edge[0]]
    words.reverse()
    print ' '.join(words)


