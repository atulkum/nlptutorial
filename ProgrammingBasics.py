import sys
filename = sys.argv[1]
counts = {}
with open(filename, 'r') as f:
    lines = f.read().split('\n')

for line in lines:
    words = line.split()
    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1

print len(counts)
i=0
for k in counts:
    print k, counts[k]
    i += 1
    if i == 10:
        break

