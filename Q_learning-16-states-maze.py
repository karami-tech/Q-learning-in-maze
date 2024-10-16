import numpy as np
import random
print("MAMad")
def main():
    print("\nSetting up maze in memory")
    R=np.array([[-1,-1,-1,-1],
            [-20,-1,-1,-20],
            [-1,-20,-1,-1],
            [-1,-1,-20,-1],
            [-1,-1,-1,-20],
            [-1,-1,-1,-1],
            [-1,-20,-20,-1],
            [-100,-1,-1,-1],
            [-1,-1,-1,-1],
            [-20,-20,-1,-1],
            [-1,-1,-1,-100],
            [100,-1,-1,-100],
            [-1,-1,-1,-20],
            [-20,-1,-1,-1],
            [-1,-1,-20,100],
            [100,-100,-1,100]])
    Q=np.zeros((16,4), dtype=np.float32)
    actions=[0,1,2,3]
    tm=np.array([[4,0,0,1],
            [5,1,0,2],
            [6,2,1,3],
            [7,3,2,3],
            [8,0,4,5],
            [9,1,4,6],
            [10,2,5,7],
            [11,3,6,7],
            [12,4,8,9],
            [13,5,8,10],
            [14,6,9,11],
            [15,7,10,11],
            [12,8,12,13],
            [13,9,12,14],
            [14,10,13,15],
            [15,11,14,15]])
#--------------------------------------
    print "Analyzing maze with RL Q-learning"
    goal = 15
    ns = 16  # number of states
    gamma = 0.99
    lrn_rate = 0.1
    max_epochs = 100
    train(R, Q, gamma, lrn_rate, goal, ns, max_epochs, tm, actions)
    print "Done"
    print "The Q matrix is: \n ", Q
    for i in range(16):
        if np.argmax(Q[i])==0:
            print " Best action in state", i+1, "is taking up"
        elif np.argmax(Q[i])==1:
            print " Best action in state", i+1, "is taking down"
        elif np.argmax(Q[i])==2:
            print " Best action in state", i+1, "is taking left"
        elif np.argmax(Q[i])==3:
            print " Best action in state", i+1, "is taking right"

def get_rnd_next_state (actions,curr_s,tm, action):
    return tm[curr_s][action]

def train ( R, Q, gamma, lrn_rate, goal, ns, max_epochs, tm, actions):
  # compute the Q matrix
  itrates=0
  for i in range(0,max_epochs):
    for j in range(ns):
      curr_s=j
      while(True):
        itrates+=1
        action=random.choice(actions)
        next_s=get_rnd_next_state(actions, curr_s, tm, action)
        Q[curr_s][action] = ((1 - lrn_rate) * Q[curr_s][action]) + (lrn_rate * (R[curr_s][action] + (gamma * np.max(Q[next_s]))))
        curr_s =next_s
        if next_s==goal:
          print "goal state reached"
          break
  print itrates
  
if __name__ == "__main__":
  main()


