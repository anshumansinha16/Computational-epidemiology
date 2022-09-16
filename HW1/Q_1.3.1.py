import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


beta = float(input("Enter your value for beta: "))
gamma = float(input("Enter your value for gama: "))
S0 = float(input("Enter your value for S0 : "))
I0 = float(input("Enter your value for I0 : "))
R0 = float(input("Enter your value for R0 : "))
max_time = int(input("Enter your value for max_time: "))

t = np.linspace(0, max_time, max_time)

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
Output = t, S, I, R

print(Output)


from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)

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