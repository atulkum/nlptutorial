import math

class UnigramLM(object):
    def __init__(self):
        self.counts = {}
        self.total_count = 0
        self.probs = {}

    def train_model(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        for line in lines:
            line = line.strip()
            words = line.split()
            if len(words) ==0:
                continue
            words += ['</s>']
            for w in words:
                if w in self.counts:
                    self.counts[w] += 1
                else:
                    self.counts[w] = 1
                self.total_count += 1

        for k in self.counts:
            self.probs[k] = float(self.counts[k]) / self.total_count

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
        lambda_unk = 1 - lambda1
        V = 1000000
        H = 0
        W = 0
        unk = 0
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split()
            words += ['</s>']
            for w in words:
                W += 1
                P = lambda_unk / V
                if w in self.probs:
                    P += lambda1 * self.probs[w]
                else:
                    unk += 1

                H += -math.log(P, 2)

        print 'Entropy: ', H / W
        print 'Coverage ', (W - unk) * 1.0 / W


if __name__ == '__main__':
    lm = UnigramLM()
    lm.train_model('test/01-train-input.txt')
    lm.save_model('01hw_model_test')
    lm.load_model('01hw_model_test')
    lm.predict('test/01-test-input.txt')

    # filename = 'test/01-train-input.txt'
    # filename = 'data/wiki-en-train.word'
    # filename = 'data/wiki-ja-train.word'

    # outfile = '01hw_model_test'
    # outfile =  '01hw_model'
    # outfile = '01hw_model_ja'

    #filename = 'data/wiki-en-test.word'
    # filename = 'test/01-test-input.txt'