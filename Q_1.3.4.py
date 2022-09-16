import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random

# initialisation

beta = 4.95691434e-02
gamma = 9.24291593e-06
max_time = 900

t = np.linspace(0, max_time, max_time)

S0 = 0.90
I0 = 0.10
R0 = 0
N = 500

I0, R0 = math.floor(N*(0.10798003802281400)), math.floor(N*(0.0020386203150461700))

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

num_run = 50 # number of simulation for the stochastic method

count_sz = [[0 for x in range(max_time)] for y in range(num_run)] 
count_iz = [[0 for x in range(max_time)] for y in range(num_run)] 
count_rz = [[0 for x in range(max_time)] for y in range(num_run)] 

count_s= [0 for x in range(max_time)]
count_i= [0 for x in range(max_time)]
count_r= [0 for x in range(max_time)]



a = [1 for x in range(N+1)]
for p in range(I0):
    num = random.randint(0, N)
    a[num]=0

    for zi in range(num_run):

  print(zi)
  
  a = [1 for x in range(N+10)]

  for p in range(I0):
    num = random.randint(0, N)
    a[num]=0

  for iter in range(max_time):
    
    cs = 0
    ci = 0
    cr = 0

    for i in a:
      if(i == 1):
        cs = cs+1
      elif(i == 0):
        ci = ci+1
      else:
        cr = cr+1 

    count_s[iter] = cs/N
    count_i[iter] = ci/N
    count_r[iter] = cr/N
    
    G = nx.Graph()

    for u in range(N): # iterate over all nodes
      #print(u)
      
      if(a[u]==1):
          #nb = nx.all_neighbors(g,u)
          #G.add_node(0)
          #nx.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")
          
          for i in range(N):
            num = random.randint(0, N-1)
            #print('num',num)
            if (i == num): 
              if(a[i]==0):
                r = round(random.uniform(0, 1),10)
                if(r<beta):
                  a[u] = 0        

      elif(a[u]==0):

            r = round(random.uniform(0, 1),10)
            if(r<gamma):
                a[u] = -1      
      
      G.add_node(u)
      nx.set_node_attributes(G, a[u],'i')

  count_sz[zi][:] = count_s
  count_iz[zi][:] = count_i
  count_rz[zi][:] = count_r

# calucate the average of all the simulations

nms= np.mean(count_sz, axis=0)
nmi= np.mean(count_iz, axis=0)
nmr= np.mean(count_rz, axis=0)

# Make the graph with the nodes which we have already added and connect them with edges

for k in G.nodes:
  for y in G.nodes:
    if(k!=i):
      G.add_edge(k, y)

# SIR ODE for comparing the results.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
#N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 0.10798003802281400, 0.0020386203150461700
# Everyone else, S0, is susceptible to infection initially.
S0 = 1 - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 4.95694772e-02, 9.24288283e-06
# A grid of time points (in days)
t = np.linspace(0, 900, 900)

# The SIR model differential equations.
def deriv(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I 
    dIdt = beta * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=( beta, gamma))
S, I, R = ret.T

# Final results.

# Plotting both the curves simultaneously

from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)

plt.plot(t, nms, color='b', label='S_SIR network')
plt.plot(t, nmi, color='r', label='I_SIR network')
plt.plot(t, nmr, color='g', label='R_SIR network')
plt.plot(t, S, color='b', linestyle='dashed', label='S_SIR ode')
plt.plot(t, I, color='r', linestyle='dashed', label='I_SIR ode')
plt.plot(t, R, color='g', linestyle='dashed', label='R_SIR ode')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time (days)")
plt.ylabel(" Fraction of S/I/R ")
plt.title("Comparison of network (N=500) and ode model")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

# end