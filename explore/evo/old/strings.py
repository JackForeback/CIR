import random
import sys
sys.setrecursionlimit(999999999)


target = ""
pop = []
scores = dict()

for i in range(20):
    target += str(random.randint(0,1))
    scores[i] = []

print("target:", target)

for i in range(1000):
    tmp = ""
    for i in range(20):
        tmp += str(random.randint(0,1))
    pop.append(tmp)

def mix(e1, e2):
    new = ""
    while len(e1) > 0:
        new += random.choice([e1[0], e2[0]])
        e1 = e1[1:]
        e2 = e2[1:]
    return new

def sub(n1, n2):
    track = 0
    for i in range(len(n1)):
        if n1[i] == n2[i]:
            track += 1
    return track

def evo(tg, pop, niter):
    temp = []
    npop = []
    for i in pop:
        if i == tg:
            return niter
        else:
            # calc num diff digits
            diff = sub(tg, i)
            # add element to corresponding diff in scores dict
            scores[diff].append(i)
    x = scores.values()
    for value in x:
        # append values in order
        for entry in value:
            temp.append(entry)

    x = temp[len(temp)-1]
    print(x,sub(tg, x), niter)
    temp = temp[(len(temp)//2):]
    for i in range(1000):
        c1 = random.randint(0,49)
        c2 = random.randint(0,49)
        npop.append(mix(str(temp[c1]),str(temp[c2])))
    
    niter += 1
    
    return evo(tg, npop, niter)



num = evo(target, pop, 0)
print(num)
