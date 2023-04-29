from collections import defaultdict as dd
from collections import deque
import copy

class Example(object):
    def __init__(self, id, word, pos, base, pos2):
        self.id = id
        self.word = word
        self.base = base
        self.pos = pos
        self.pos2 = pos2
        self.head = -1
        self.unproc = 0
        self.tokens = None

class ShiftReduce(object):
    def __init__(self):
        self.W = {}
        for c in ['s','l', 'r']:
            self.W[c] = dd(int)

    def load_data(self, filename):
        data = []
        with open(filename, 'r') as f:
            lines = f.read().split('\n')
        queue = deque([Example(0, 'ROOT', 'ROOT', 'ROOT', 'ROOT')])
        words = []
        for line in lines:
            if line:
                tokens = line.split('\t')
                id = int(tokens[0])
                word = tokens[1]
                base = tokens[2]
                pos = tokens[3]
                pos2 = tokens[4]
                head = tokens[6]
                ex = Example(id, word, pos, base, pos2)
                ex.head = int(head)
                ex.tokens = tokens
                queue.append(ex)
                words.append(word)
            elif len(queue) > 1:
                #fix unproc
                for i in range(1, len(queue)):
                    ex = queue[i]
                    if ex.head > 0:
                        queue[ex.head].unproc += 1
               
                data.append(('|'.join(words), queue))
                queue = deque([Example(0, 'ROOT', 'ROOT', 'ROOT', 'ROOT')])
                words = []
        return data

    def create_feats(self, stack, queue):
        phi = dd(int)
        feats = []
        if len(stack) >= 2:
            ex = stack[-2]
            prefix = 'stack-2'
            feats.append((ex, prefix))
        if len(stack) >= 1:
            ex = stack[-1]
            prefix = 'stack-1'
            feats.append((ex, prefix))
        if len(queue) >= 1:
            ex = queue[0]
            prefix = 'queue0'
            feats.append((ex, prefix))
        
        for ex, prefix in feats:
            #phi['%s:%s %s %s %s'%(prefix, ex.word, ex.pos, ex.base, ex.pos2)] += 1
            phi['%s:%s %s'%(prefix, ex.word, ex.pos)] += 1

        return phi

    def predict_action(self, feats, queue):
        scores = {}
        for c in ['s','l', 'r']:
            w = self.W[c]
            score = 0
            for name, value in feats.items():
                if name in w:
                    score += value*w[name]
            scores[c] = score

        argmax =  max(scores, key=scores.get)
        if argmax == 's' and len(queue) > 0:
            return 's'
        elif scores['l'] >= scores['r']:
            return 'l'
        else:
            return 'r'
    def proces_action(self, action, stack, queue, heads):
        if action == 's':
            stack.append(queue.popleft())
        elif action == 'l':
            heads[stack[-2].id] = stack[-1].id
            stack.pop(-2)
        else:
            heads[stack[-1].id] = stack[-2].id
            stack.pop(-1)
        
    def get_correct_action(self, stack):
        if len(stack) >= 2 and stack[-1].head == stack[-2].id and stack[-1].unproc == 0:
            stack[-2].unproc -= 1
            return 'r'
        elif len(stack) >= 2 and stack[-2].head == stack[-1].id and stack[-2].unproc == 0:
            stack[-1].unproc -= 1
            return 'l'
        else:
            return 's'

    def shift_reduce(self, queue, is_train=True):
        heads = [-1]*len(queue)
        stack = []
        #initialize stack with ROOT
        stack.append(queue.popleft())

        while len(queue) > 0 or len(stack) > 1:
            feats = self.create_feats(stack, queue)
            ans = self.predict_action(feats, queue)
            if is_train:
                corr = self.get_correct_action(stack)

                if ans != corr:
                    for name, value in feats.items():
                        self.W[ans][name] -= value
                        self.W[corr][name] += value
                action = corr
            else:
                action = ans

            self.proces_action(action, stack, queue, heads)

        return heads

if __name__ == '__main__':
    sr = ShiftReduce()
    train_data = sr.load_data('data/mstparser-en-train.dep')
    for sentence, q in train_data:
        queue = copy.copy(q)
        heads = sr.shift_reduce(queue)
        #print sentence, heads

    test_data = sr.load_data('data/mstparser-en-test.dep')
    for sentence, q in test_data:
        queue = copy.copy(q)
        heads = sr.shift_reduce(queue, False)
        for i in range(1, len(heads)):
            tokens = q[i].tokens
            tokens[6] = str(heads[i])
            print '\t'.join(tokens)
        print ' '
        
