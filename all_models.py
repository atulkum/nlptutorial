import cPickle as pickle
import pandas as pd
from collections import defaultdict as dd

class ModelBase(object):
    def predict_all(self, model_file, input_file):
        with open(input_file,'r') as f:
            lines = f.read().split('\n')
        with open(model_file, 'r') as f:
            w = pickle.load(f)
            for line in lines:
                if len(line) == 0:
                    continue
                phi = create_features(line)
                y_hat = predict_one(w, phi)
                print y_hat
    

    def create_features(self, x):
        phi = dd(int)
        words = x.split()
        for word in words:
            phi['UNI:%s'%word] += 1
        return phi


    def train(self, train_file, model_file):
        with open(train_file,'r') as f:
            lines = f.read().split('\n')
        w = dd(float)
        for i in range(10):
            for line in lines:
                row = line.split('\t')
                if len(row) < 2:
                    continue
                x = row[1]
                y = int(row[0])

                phi = create_features(x)
                y_hat = predict_one(w, phi)
                update_weights(w, phi, y, y_hat)

        with open(model_file, 'w') as f:
            pickle.dump(w, f, pickle.HIGHEST_PROTOCOL)

    def update_weights(self, w, phi, y, y_hat):
        pass
    def predict_one(self, w, phi):
        pass

class OnlinePerceptron(ModelBase):
    def predict_one(self, w, phi):
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value*w[name]
        if score >= 0:
            return 1
        else:
            return -1

    def update_weights(self, w, phi, y, y_hat):
        if y_hat != y:
            for name, value in phi.items():
                w[name] += (value * y)

class L1RegularizedOnlPerc(OnlinePerceptron):
    def update_weights(self, w, phi, y, y_hat):
        c = 0.001
        sign = lambda x: (1, -1)[x < 0]
        if y_hat != y:
            for name, value in w:
                if abs(value) < c:
                    w[name] = 0
                else:
                    w[name] -= sign(value)*c

            for name, value in phi.items():
                w[name] += (value * y)

class LogisticRegression(ModelBase):
    def predict_one(self, w, phi):
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value*w[name]
        prob = np.exp(score)/(1+np.exp(score))
        if prob >= 0.5:
            return 1
        else:
            return -1 

    def update_weights(self, w, phi, y, y_hat):
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value*w[name]

        e_x = np.exp(score)
        gradient = e_x/((1+e_x)**2)
        alpha = 1
        for name, value in phi.items():
            w[name] += alpha*(y * value * gradient)

class MarginBased(LogisticRegression):

    def update_weights(self, w, phi, y, y_hat):
        alpha = 1
        margin = 1

        score = 0
        for name, value in phi.items():
            if name in w:
                score += value*w[name]
        val = y*score

        if val <= margin:
            e_x = np.exp(score)
            gradient = e_x/((1+e_x)**2)
            for name, value in phi.items():
                w[name] += alpha*(y * value * gradient)


if __name__ == '__main__':
    model_file = '06hw_model'
    #train_file = 'test/03-train-input.txt'
    train_file = 'data/titles-en-train.labeled'
    train(train_file, model_file)
        predict_all(model_file, 'data/titles-en-test.word')


