from collections import defaultdict
import sys
import math

def load_data(filename):
    data = []
    possible_tags = set()
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            wordtags = line.split()
            X, Y = [], []
            for wordtag in wordtags:
                t = wordtag.split('_')
                word, tag = t[0], t[1]
                X.append(word)
                Y.append(tag)
                possible_tags.add(tag)
            data.append((X, Y))
    possible_tags.add('<s>')
    return data, possible_tags

class HMM(object):
    def __init__(self, possible_tags=None):
        self.emission = defaultdict(float)
        self.transition = defaultdict(float)
        self.possible_tags = possible_tags
        self.N = 1000000
        self.lambda1 = 0.95

    def get_emission_prob(self, key):
        P = (1 - self.lambda1) / self.N
        if key in self.emission:
            P += self.lambda1 * self.emission[key]
        return P

    def hmm_viterbi(self, words):
        l = len(words)
        best_score = {}
        best_edge = {}

        best_score[(0, '<s>')] = 0
        best_edge[(0, '<s>')] = None

        for i in range(l):
            for prev_tag in self.possible_tags:
                for next_tag in self.possible_tags:
                    if (i, prev_tag) in best_score and (prev_tag, next_tag) in self.transition:
                        P = self.get_emission_prob((next_tag, words[i]))
                        score = best_score[(i, prev_tag)] + \
                                (-math.log(self.transition[(prev_tag, next_tag)], 2)) + \
                                (-math.log(P, 2))

                        if ((i + 1, next_tag) not in best_score) or (best_score[(i + 1, next_tag)] > score):
                            best_score[(i + 1, next_tag)] = score
                            best_edge[(i + 1, next_tag)] = (i, prev_tag)

        for prev_tag in self.possible_tags:
            next_tag = '</s>'
            if (l, prev_tag) in best_score and (prev_tag, next_tag) in self.transition:
                score = best_score[(l, prev_tag)] + \
                        (-math.log(self.transition[(prev_tag, next_tag)], 2))
                if (l + 1, next_tag) not in best_score or best_score[(l + 1, next_tag)] > score:
                    best_score[(l + 1, next_tag)] = score
                    best_edge[(l + 1, next_tag)] = (l, prev_tag)
        tags = []
        next_edge = best_edge[(l + 1, '</s>')]
        while next_edge != (0, '<s>'):
            position, tag = next_edge
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        return tags

    def train(self, data):
        tx_context = defaultdict(float)
        em_context = defaultdict(float)
        for X, Y in data:
            previous = '<s>'
            for word, tag in zip(X, Y):
                self.transition[(previous, tag)] += 1
                tx_context[previous] += 1
                self.emission[(tag, word)] += 1
                em_context[tag] += 1
                previous = tag

            self.transition[(previous, '</s>')] += 1
            tx_context[previous] += 1

        for key, value in self.transition.items():
            previous, tag = key
            self.transition[key] = value / tx_context[previous]
            #print 'T', previous, tag, self.transition[key]

        for key, value in self.emission.items():
            tag, word = key
            self.emission[key] = value / em_context[tag]
            #print 'E', tag, word, self.emission[key]

    def save_model(self, filename):
        with open(filename, 'w') as f:
            for key, value in self.transition.items():
                prev_token, next_token = key
                f.write('T %s %s %f\n'%(prev_token, next_token, value))
            for key, value in self.emission.items():
                prev_token, next_token = key
                f.write('E %s %s %f\n'%(prev_token, next_token, value))

    def load_model(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')
        possible_tags = set()
        for line in lines:
            if len(line) == 0:
                continue
            t = line.split()
            type, pre_token, next_token, prob = t[0], t[1], t[2], float(t[3])

            if type == 'T':
                possible_tags.add(pre_token)
                self.transition[(pre_token, next_token)] = prob
            else:
                self.emission[(pre_token, next_token)] = prob
        self.possible_tags = possible_tags

class MEM(object):
    def __init__(self, possible_tags=None):
        self.w = defaultdict(int)
        self.possible_tags = possible_tags

    def create_trans(self, first_tag, next_tag):
        phi = defaultdict(int)
        phi['T,%s,%s'%(first_tag, next_tag)] = 1
        return phi

    def create_emit(self, y, x):
        phi = defaultdict(int)
        phi['E,%s,%s'%(y, x)] = 1
        phi['CAPS,%s'%(x)] = 1 if x[0].isupper() else 0
        return phi

    def create_feature(self, X, Y):
        phi = defaultdict(int)
        for i in range(len(Y) + 1):
            first_tag = '<s>' if i == 0 else Y[i-1]
            next_tag = '</s>' if i == len(Y) else Y[i]
            tx = self.create_trans(first_tag, next_tag)
            for k, v in tx.items():
                phi[k] += v
        for i in range(len(Y)):
            em = self.create_emit(Y[i], X[i])
            for k, v in em.items():
                phi[k] += v

        return phi

    def hmm_viterbi(self, words):
        l = len(words)
        best_score = {}
        best_edge = {}
        
        best_score[(0, '<s>')] = 0
        best_edge[(0, '<s>')] = None

        for i in range(l):
            for prev_tag in self.possible_tags:
                for next_tag in self.possible_tags:
                    if (i, prev_tag) in best_score:
                        act = 0.0
                        tx = self.create_trans(prev_tag, next_tag)
                        for k, v in tx.items():
                            act += self.w[k]*v
                        em = self.create_emit(next_tag, words[i])
                        for k, v in em.items():
                            act += self.w[k]*v

                        score = best_score[(i, prev_tag)] + act
                        
                        if ((i+1, next_tag) not in best_score) or (best_score[(i+1, next_tag)] > score):
                            best_score[(i+1, next_tag)] = score
                            best_edge[(i+1, next_tag)] = (i, prev_tag)

        for prev_tag in self.possible_tags:
            next_tag = '</s>'
            if (l, prev_tag) in best_score:
                act = 0.0
                tx = self.create_trans(prev_tag, next_tag)
                for k, v in tx.items():
                    act += self.w[k] * v
                score = best_score[(l, prev_tag)] + act
                if (l+1, next_tag) not in best_score or best_score[(l+1, next_tag)] > score:
                    best_score[(l+1, next_tag)] = score
                    best_edge[(l+1, next_tag)] = (l, prev_tag)
        tags = []
        next_edge = best_edge[(l+1, '</s>')]
        while next_edge != (0, '<s>'):
            position, tag = next_edge
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        return tags

    def train(self, data, l):
        for i in range(l):
            for X, Y_prime in data:
                Y_hat = self.hmm_viterbi(X)
                phi_prime = self.create_feature(X, Y_prime)
                phi_hat = self.create_feature(X, Y_hat)

                for k, v in phi_prime.items():
                    self.w[k] += v
                for k, v in phi_hat.items():
                    self.w[k] -= v
                    self.w[k] += v

    def save_model(self, filename):
        with open(filename, 'w') as f:
            for key, value in self.w.items():
                f.write('%s %f\n'%(key, value))

        with open(filename + '.possible_tags', 'w') as f:
            for tag in self.possible_tags:
                f.write('%s\n'%(tag))

    def load_model(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            t = line.split()
            key, value = t[0], float(t[1])
            self.w[key] = value

        possible_tags = set()
        with open(filename + '.possible_tags', 'r') as f:
            lines = f.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            possible_tags.add(line.strip())
        self.possible_tags = possible_tags

if __name__ == '__main__':
    #train_filename = sys.argv[1]
    #test_filename = sys.argv[1]

    #train_filename = 'test/05-train-input.txt'
    #test_filename = 'test/05-test-input.txt'

    train_filename = 'data/wiki-en-train.norm_pos'
    test_filename = 'data/wiki-en-test.norm'

    data, possible_tags = load_data(train_filename)
    '''
    model = HMM(possible_tags)
    model.train(data)
    model_savefile = 'HMM.model.wiki'
    model.save_model(model_savefile)
    '''
    model = MEM(possible_tags)
    model.train(data, 10)
    model_savefile = 'MEM.model.wiki'
    model.save_model(model_savefile)

    with open(test_filename,'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            words = line.split()
            tags = model.hmm_viterbi(words)
            print ' '.join(tags)



#script/gradepos.pl data/wiki-en-test.pos my_answer.pos


