#Q 1.3.3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
#N = 100
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 0.10798003802281400, 0.0020386203150461700
# Everyone else, S0, is susceptible to infection initially.
S0 = 1 - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#beta, gamma = 0.2, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 146, 146)

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
ret = odeint(deriv, y0, t, args=(beta, gamma))
S, I, R = ret.T

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

filename = '/content/COVID19_GA.csv'
df = pd.read_csv(filename)
df.shape

x = df.to_numpy()

def fn(x):
    # parameters unwrapped
    beta, gamma = x;
    
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 0.10798003802281400, 0.0020386203150461700
    # Everyone else, S0, is susceptible to infection initially.
    S0 = 1 - I0 - R0
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over time t.
    ret = odeint(deriv, y0, t_d, args=(beta, gamma))
    S, I, R = ret.T

    # absolute error to avoid negatives
    error = sum((R - R_d)*(R-R_d))
    
    return error

# initialise with current best guess
init_x = [0.05, 0.1]

# calculate result
res = minimize(fn, init_x, method='Nelder-Mead', tol=1e-8)

# calculate final error
fn(res.x)



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
t = np.linspace(0, 160, 160)

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

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='w', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Fractions')
#ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

#plt.title(r'Daily cases - SIR fit: N=42 000, $\beta=0.148$, $\gamma=0.05$')
plt.xlabel('Days')
plt.ylabel('Cummalative deaths')
plt.plot(t_d,R_d, ".")
plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
plt.show()