import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_random_curves(x, sigma=0.2, knot=3):
    xx = np.arange(0,len(x), (len(x)-1)/(knot+1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=knot+2)

    x_range = np.arange(len(x))
    cs = CubicSpline(xx, yy)

    return np.array(cs(x_range))

def distort_timesteps(x, sigma=0.2):
    tt = generate_random_curves(x, sigma, knot=7) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = (len(x)-1)/tt_cum[-1]
    tt_cum = tt_cum*t_scale
    return tt_cum

## Random curves around 1.0
my_x = np.zeros(250)
fig = plt.figure(figsize=(16,4))
ax1 = plt.subplot(121)
for ii in range(3):
    ax1.plot(generate_random_curves(my_x, 0.2
    ))
    ax1.hold(True)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([0,2])

ax2 = plt.subplot(122)
for ii in range(3):
    ax2.plot(distort_timesteps(my_x, 0.4))
    ax2.hold(True)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.draw()
plt.waitforbuttonpress()
plt.close()
