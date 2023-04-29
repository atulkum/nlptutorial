import numpy as np
from collections import defaultdict

class TopicModeling(object):
    def __init__(self, num_topics, filename):
        self.num_topics = num_topics
        self.xcorpus = []
        self.ycorpus = []

        self.xcounts = defaultdict(int)
        self.ycounts = defaultdict(int)

        self.Nx = 0
        self.Ny = self.num_topics

        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        all_words = set()
        for line in lines:
            if not line:
                continue
            docid = len(self.xcorpus)
            words = line.split()
            topics = []
            for word in words:
                all_words.add(word)
                topic = np.random.uniform(0, self.num_topics)
                topics.append(topic)
                self.add_counts(word, topic, docid, 1)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)
        self.Nx = len(all_words)

    def sample_one(self, probs):
        z = sum(probs)
        remaining = np.random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i
        return len(probs) - 1

    def add_counts(self, word, topic, docid, c=1):
        self.xcounts[topic] += c
        self.xcounts[(word,topic)] += c
        
        self.ycounts[docid] += c
        self.ycounts[(word,docid)] += c

    def get_word_prob(self, wij, yij):
        alpha = 0.95
        denom = self.xcounts[yij] + alpha*self.Nx
        numerator = alpha
        if (wij, yij) in self.xcounts:
            numerator += self.xcounts[(wij, yij)]
        return numerator/denom

    def get_topic_prob(self, yij, y):
        beta = 0.95
        denom = self.xcounts[y] + beta*self.Ny
        numerator = beta
        if (yij, y) in self.xcounts:
            numerator += self.xcounts[(yij, y)]
        return numerator/denom
        

    def sampling(self):
        for itr in range(10):
            ll = 0
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, -1)
                    probs = []
                    for k in range(self.num_topics):
                        word_prob = self.get_word_prob(x, k)
                        topic_prob = self.get_topic_prob(k, y)
                        probs.append(word_prob*topic_prob)
                    new_y = self.sample_one(probs)
                    ll += np.log2(probs[new_y])
                    self.add_counts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y
            print ll
        #print self.xcounts, self.ycounts
        for i in range(len(self.xcorpus)):
            print self.xcorpus[i], self.ycorpus[i]


if __name__ == '__main__':
    #model = TopicModeling(2, 'test/07-train.txt')
    model = TopicModeling(20, 'data/wiki-en-documents.word')
    model.sampling()
