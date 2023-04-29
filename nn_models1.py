from collections import defaultdict as dd

ids = dd(lambda: len(ids))

def create_features(self, x):
    phi = dd(float)
    words = x.split()
    for word in words:
        phi[ids['UNI:%s'%word]] += 1
    return phi

w = np.zeros(len(ids))
# w = np.random.rand(len(ids)) – 0.5

with open(train_file,'r') as f:
    lines = f.read().split('\n')

for i in range(10):
    for line in lines:
        row = line.split('\t')
        if len(row) < 2:
            continue
        x = row[1]
        y = int(row[0])

        phi = create_features(x)
        y_hat = predict_one(w, phi)
        if y != y_hat:
            update_weights(w, phi, y)

with open(model_file, 'w') as f:
    pickle.dump(w, f, pickle.HIGHEST_PROTOCOL)
