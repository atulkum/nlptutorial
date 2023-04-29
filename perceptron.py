import cPickle as pickle
from collections import defaultdict as dd

class Perceptron(object):
    def __init__(self):
        self.w = dd(int)

    def predict_all(self, input_file):
        with open(input_file,'r') as f:
            lines = f.read().split('\n')

            for line in lines:
                if len(line) == 0:
                    continue
                phi = self.create_features(line)
                y_hat = self.predict_one(phi)
                print y_hat

    def predict_one(self, phi):
        score = 0
        for name, value in phi.items():
            if name in self.w:
                score += value*self.w[name]
        if score >= 0:
            return 1
        else:
            return -1

    def create_features(self, x):
        phi = dd(int)
        words = x.split()
        for word in words:
            phi['UNI:%s'%word] += 1
        return phi

    def update_weights(self, phi, y):
        for name, value in phi.items():
            self.w[name] += (value * y)

    def train(self, train_file):
        with open(train_file,'r') as f:
            lines = f.read().split('\n')

        for i in range(10):
            for line in lines:
                row = line.split('\t')
                if len(row) < 2:
                    continue
                x = row[1]
                y = int(row[0])

                phi = self.create_features(x)
                y_hat = self.predict_one(phi)
                if y_hat != y:
                    self.update_weights(phi, y)

    def save_model(self, model_file):
        with open(model_file, 'w') as f:
            pickle.dump(self.w, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_file):
        with open(model_file, 'r') as f:
            self.w = pickle.load(f)



if __name__ == '__main__':
    model_file = '05hw_model'
    train_file = 'test/03-train-input.txt'
    train_file = 'data/titles-en-train.labeled'
    model = Perceptron()
    model.train(train_file)
    model.save_model(model_file)
    model.load_model(model_file)
    model.predict_all('data/titles-en-test.word')

