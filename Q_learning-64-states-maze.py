import numpy as np
import random
import matplotlib.pyplot as plt
import math


def main(allitrates, epsilon,i, Q):
  print("\nSetting up maze in memory")
  R=np.array(
          [[-1,-1,-1,-1],
          [-1,-1,-1,-1],
          [-20,-1,-1,-1],
          [-1,-1,-1,-1],
          [-1,-1,-1,-1],
          [-20,-1,-1,-1],
          [-1,-1,-1,-1],
          [-1,-1,-1,-1],

          [-1,-1,-1,-1],
          [-1,-1,-1,-20],
          [-20,-1,-1,-1],
          [-1,-1,-20,-1],
          [-1,-1,-1,-20],
          [-20,-1,-1,-1],
          [-1,-1,-20,-1],
          [-1,-1,-1,-1],

          [-1,-1,-1,-1],
          [-1,-1,-1,-20],
          [-20,-20,-1,-1],
          [-1,-1,-20,-1],
          [-1,-1,-1,-20],
          [-1,-20,-1,-1],
          [-1,-1,-20,-1],
          [-1,-1,-1,-1],

          [-1,-1,-1,-1],
          [-20,-1,-1,-20],
          [-20,-20,-1,-1],
          [-1,-1,-20,-1],
          [-20,-1,-1,-1],
          [-20,-20,-1,-1],
          [-20,-1,-1,-1],
          [-20,-1,-1,-1],

          [-1,-1,-1,-20],
          [-1,-1,-1,-20],
          [-1,-20,-20,-1],
          [-1,-1,-20,-20],
          [-1,-1,-1,-20],
          [-1,-1,-20,-20],
          [-1,-1,-20,-20],
          [-1,-1,-20,-1],

          [-20,-1,-1,-1],
          [-20,-20,-1,-1],
          [-1,-20,-1,-1],
          [-20,-1,-1,-1],
          [-1,-20,-1,-1],
          [-20,-20,-1,-1],
          [-20,-20,-1,-1],
          [-1,-20,-1,-1],

          [-1,-1,-1,-20],
          [-1,-1,-20,-1],
          [-1,-1,-20,-20],
          [-20,-1,-1,-1],
          [-1,-1,-20,-20],
          [-1,-1,-1,-20],
          [-1,-1,-20,-1],
          [+100,-1,-20,-1],

          [-1,-20,-1,-1],
          [-1,-20,-1,-1],
          [-1,-1,-1,-20],
          [-20,-20,-1,-1],
          [-1,-1,-20,-1],
          [-1,-20,-1,-1],
          [-1,-20,-1,+100],
          [+100,-1,-1,+100]])
  actions=[0,1,2,3]
  tm=np.array(
          [[8,0,0,1],
          [9,1,0,2],
          [10,2,1,3],
          [11,3,2,4],
          [12,4,3,5],
          [13,5,4,6],
          [14,6,5,7],
          [15,7,6,7],

          [16,0,8,9],
          [17,1,8,10],
          [18,2,9,11],
          [19,3,10,12],
          [20,4,11,13],
          [21,5,12,14],
          [22,6,13,15],
          [23,7,14,15],

          [24,8,16,17],
          [25,9,16,18],
          [26,10,17,19],
          [27,11,18,20],
          [28,12,19,21],
          [29,13,20,22],
          [30,14,21,23],
          [31,15,22,23],

          [32,16,24,25],
          [33,17,24,26],
          [34,18,25,27],
          [35,19,26,28],
          [36,20,27,29],
          [37,21,28,30],
          [38,22,29,31],
          [39,23,30,31],

          [40,24,32,33],
          [41,25,32,34],
          [42,26,33,35],
          [43,27,34,36],
          [44,28,35,37],
          [45,29,36,38],
          [46,30,37,39],
          [47,31,38,39],

          [48,32,40,41],
          [49,33,40,42],
          [50,34,41,43],
          [51,35,42,44],
          [52,36,43,45],
          [53,37,44,46],
          [54,38,45,47],
          [55,39,46,47],

          [56,40,40,41],
          [57,41,40,42],
          [58,42,41,43],
          [59,43,42,44],
          [60,44,43,45],
          [61,45,44,46],
          [62,46,45,47],
          [63,47,46,48],

          [56,48,56,57],
          [57,49,56,58],
          [58,50,57,59],
          [59,51,58,60],
          [60,52,59,61],
          [61,53,60,62],
          [62,54,61,63],
          [63,55,62,63],

          ])
#--------------------------------------
  print "Analyzing maze with RL Q-learning"
  goal = 64
  ns = 64  # number of states
  gamma = 0.3
  lrn_rate = 1.
  max_epochs = 50
  allitrts, itrts, Qt=train(R, Q, gamma, lrn_rate, goal, ns, max_epochs, tm, actions, allitrates, epsilon)
  return allitrts, itrts, Qt

def train ( R, Q, gamma, lrn_rate, goal, ns, max_epochs, tm, actions, allitrates, epsilon):
  curr_s=0
  itrates=1
  while(True):
    itrates+=1
    numberofsteps=allitrates+itrates
    lr_diminishing=lrn_rate/math.log10(numberofsteps)
    action=random.choice(actions)
    if epsilon/(1+math.sqrt(numberofsteps)) < random.random():
      action=np.argmax(Q[curr_s])
    next_s=get_rnd_next_state(actions, curr_s, tm, action)

    Q[curr_s][action] = ((1 - lr_diminishing) * Q[curr_s][action]) + (lr_diminishing * (R[curr_s][action] + (gamma * np.max(Q[next_s]))))
    curr_s =next_s
    if R[curr_s][action]==100 :
      allitrates+=itrates
      break
  print "It converged at Epoch number%i "%itrates
  return allitrates, itrates, Q 

def get_rnd_next_state (actions,curr_s,tm, action):
  return tm[curr_s][action] 

allitrates=0
epsilon=10
itratestrack=[]
Q=np.zeros((64,4), dtype=np.float32)
for i in range(1000):
  alitr,itr,Qt=main(allitrates, epsilon,i, Q)
  Q=Qt
  itratestrack.append(itr)
  allitrates=alitr
print allitrates
print itratestrack

