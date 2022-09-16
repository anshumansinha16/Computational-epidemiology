import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


alpha = float(input("Enter your value for alpha: "))
delta = float(input("Enter your value for delta: "))
beta = float(input("Enter your value for beta: "))
gamma = float(input("Enter your value for gama: "))
x0 = float(input("Enter your value for x0 : "))
y0 = float(input("Enter your value for y0 : "))
max_time = int(input("Enter your value for max_time: "))

t = np.linspace(0, max_time, max_time)

# The SIR model differential equations.
def deriv(z, t, alpha, delta, beta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y 
    dydt = gamma * x * y  - delta * y
    return dxdt, dydt


# Initial conditions vector
z0 = x0, y0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, z0, t, args=( alpha, delta, beta, gamma))

x, y = ret.T
Output = t, x, y

#print(Output)

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, x, 'b', alpha=0.5, lw=2, label='x')
ax.plot(t, y, 'r', alpha=0.5, lw=2, label='y')

ax.set_xlabel('Time')
ax.set_ylabel('x,y')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()