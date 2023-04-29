import cPickle as pickle
from collections import defaultdict as dd
import math

class CKY(object):
    def read_grammer(self, input_file):
        with open(input_file,'r') as f:
            rules = f.read().split('\n')
        nonterm = []
        preterm = dd(list)
        self.all_lhs = []
        for rule in rules:
            if len(rule) == 0:
                continue
            r = rule.split('\t')
            lhs = r[0]
            rhs = r[1].split()
            prob = float(r[2])
            logprob = math.log(prob, 2)
            self.all_lhs.append(lhs)

            if len(rhs) == 1:
                preterm[rhs[0]].append((lhs, logprob))
            else:
                nonterm.append((lhs, rhs[0], rhs[1], logprob))

        self.nonterm = nonterm
        self.preterm = preterm

    def parse(self, line):
        words = line.split()
        self.init_parsing(words)
        self.genrate_tree_bottom_up(words)
        root = ('S', 0, len(words))
        print self.print_tree(root, words)
        #for k in self.best_score:
        #    for a in self.best_score[k]:
        #        print k, a
        #print self.best_edge

    def init_parsing(self, words):
        n = len(words)
        self.best_score = {} 
        self.best_edge = {}
        for lhs in self.all_lhs:
            self.best_score[lhs] = [[-1000]*(n+1)]*(n+1)
            self.best_edge[lhs] = [[None]*(n+1)]*(n+1)

        for i in range(n):
            word = words[i]
            if word in self.preterm:
                for lhs, log_prob in self.preterm[word]:
                    self.best_score[lhs][i][i+1] = log_prob
            else:
                print 'not found in non terminal', word

    def genrate_tree_bottom_up(self, words):
        n = len(words)
        for j in range(2, n+1):
            for i in reversed(range(j-2 +1)):
                #process span (i, j)
                for k in range(i+1, j):
                    for sym, lsym, rsym, logprob in self.nonterm:
                        my_log_prob = logprob + self.best_score[lsym][i][k] + self.best_score[rsym][k][j]
                        if my_log_prob > self.best_score[sym][i][j]:
                            self.best_score[sym][i][j] = my_log_prob
                            self.best_edge[sym][i][j] = ((lsym, i, k), (rsym, k, j))


    def print_tree(self, root, words):
        sym, i, j = root
        if self.best_edge[sym][i][j] is not None:
            left, right = self.best_edge[sym][i][j]
            return '(' + sym + ' ' + self.print_tree(left, words) + ' ' + self.print_tree(right, words) + ')'
        else:
            return '(' + sym + ' ' + words[i] + ')'

if __name__ == '__main__':
    #input_file, grammer_file = 'test/08-input.txt', 'test/08-grammar.txt'
    input_file, grammer_file = 'data/wiki-en-short.tok', 'data/wiki-en-test.grammar'
    cky = CKY()
    cky.read_grammer(grammer_file)

    with open(input_file,'r') as f:
        lines = f.read().split('\n')
    for line in lines:
        if line:
            cky.parse(line)



