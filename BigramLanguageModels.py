from collections import defaultdict
import math

class BigramLM(object):
    def __init__(self):
        self.counts=defaultdict(int)
        self.context_counts= defaultdict(int)
        self.probs = {}

    def train_model(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split()
            words.append('</s>')
            words.insert(0, '<s>')
            for i in range(1, len(words)):
                self.counts[' '.join([words[i-1], words[i]])] += 1
                self.context_counts[words[i-1]] += 1
                self.counts[words[i]] += 1
                self.context_counts[''] += 1

        for ngram, v in self.counts.iteritems():
            words = ngram.split()
            if len(words) > 1:
                context = words[:-1]
                context = ' '.join(context)
            else:
                context = ''

            self.probs[ngram] = float(self.counts[ngram]) / self.context_counts[context]

    def save_model(self, outfile):
        with open(outfile,'w') as f:
            for k in self.probs:
                f.write(k + '|' + str(self.probs[k]) + '\n')


    def load_model(self, filename):
        self.probs = {}
        with open(filename,'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                t = line.split('|')
                if len(t) == 2:
                    self.probs[t[0]] = float(t[1])

    def predict(self, input_data):
        with open(input_data, 'r') as f:
            lines = f.read().split('\n')
        lambda1 = 0.95
        lambda2 = 0.95
        V = 1000000
        H = 0
        W = 0
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split()
            words.append('</s>')
            words.insert(0, '<s>')
            for i in range(1, len(words)):
                P1 = (1-lambda1)/V
                ngram = words[i]
                if ngram in self.probs:
                    P1 += lambda1*self.probs[ngram]
                P2 = (1-lambda2)*P1
                ngram = ' '.join([words[i-1], words[i]])
                if ngram in self.probs:
                    P2 += lambda2*self.probs[ngram]
                W += 1
                H += -math.log(P2,2)

        print 'Entropy: ', H/W

if __name__ == '__main__':
    lm = BigramLM()
    lm.train_model('test/02-train-input.txt')
    lm.save_model('02hw_model_test')
    lm.load_model('02hw_model_test')
    lm.predict('test/02-train-answer.txt')

    # 'data/wiki-en-train.word',
    #'02hw_model',
    #'data/wiki-en-test.word'