import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# These constants are arbitrary. Play with them.
a = 1
b = 1
c = 1
d = 1
delta = 0.01
x= 2                       # Initial rabbit population
y = 1                      # Initial wolf population
time_period = 1600         # Number of time periods (iterations) including t0


# Now for the actual calculation
tup = (0,x,y)              # The triple at time zero
data=[tup]                 # Create a list containining that triple
for i in range(1,time_period):
    prevx = data[i-1][1]
    prevy = data[i-1][2]
    new_tup = (i,prevx+(delta*(a*prevx - b*prevx*prevy)),prevy+(delta*(c*prevx*prevy - d*prevy)))
    data.append(new_tup)   # Append your newly calculated triple

rabbit = [data[i][1] for i in range(time_period)] # Extract rabbit data
wolf = [data[i][2] for i in range(time_period)]   # Extract wolf data

# Time to plot our results:
def plot_pop():            # In a function because it's easier to deal with
    plt.plot(rabbit, linewidth=2.0, label=r"Rabbit")
    plt.plot(wolf, linewidth=2.0, label=r"Wolf")
    plt.title(r"a=%s, b=%s, c=%s, d=%s, x$_o$=%s, y$_o$=%s, delta=%s" % (a,b,c,d, rabbit[0],wolf[0],delta))
    plt.ylabel(r"Rabbit and Wolf Population")
    plt.xlabel(r"Time Periods (Iterations)")
    plt.legend()
    plt.show()
#plot_pop()

####################

# Now let's take a look at the ab error function, and minimize it
def error_ab_fun(s,t):
    error_ab = 0
    for i in range(1,time_period):
        error_ab += (((rabbit[i]-rabbit[i-1])/delta) - (s*rabbit[i] - t*rabbit[i]*wolf[i]))**2
    return error_ab

def plot_ab_error():
    a_range = [0.01*i for i in range(201)]
    b_range = [0.01*i for i in range(201)]
    error = plt.figure()
    ax = error.gca(projection="3d")
    x = a_range
    y = b_range
    x, y = np.meshgrid(x,y)
    z = error_ab_fun(x,y)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel("a variable")
    plt.ylabel("b variable")
    plt.show()
#plot_ab_error()
    
def ab_gradient_descent(a_current, b_current):
    a_gradient = 0
    b_gradient = 0
    step_size = 0.000001
    for i in range(1,time_period):
        a_gradient += -2*rabbit[i]*(((rabbit[i]-rabbit[i-1])/delta) - (a_current*rabbit[i] - b_current*rabbit[i]*wolf[i]))
        b_gradient += 2*rabbit[i]*wolf[i]*(((rabbit[i]-rabbit[i-1])/delta) - (a_current*rabbit[i] - b_current*rabbit[i]*wolf[i]))
    new_a = a_current - (step_size*a_gradient)
    new_b = b_current - (step_size*b_gradient)
    return new_a, new_b

def ab_gradient_iterate():
    a_current = 1
    b_current = 1
    error_monitor = []
    for i in range(20000):
        a_current, b_current = ab_gradient_descent(a_current, b_current)
        error_monitor.append(error_ab_fun(a_current,b_current))
    print("Minimum error = %s, and occurs at a = %s, b=%s." % (error_ab_fun(a_current,b_current),a_current,b_current))
    plt.plot(error_monitor, linewidth=2, label="Error")
    plt.title(r"a,b Error function during gradient descent")
    plt.ylabel(r"Error")
    plt.xlabel(r"Iterations")
    plt.show()
#ab_gradient_iterate()

####################

# Now let's do the same for the cd error function
def error_cd_fun(s,t):
    error_cd = 0
    for i in range(1,time_period):
        error_cd += (((wolf[i]-wolf[i-1])/delta) - (s*rabbit[i]*wolf[i] - t*wolf[i]))**2
    return error_cd

def plot_cd_error():
    c_range = [0.01*i for i in range(201)]
    d_range = [0.01*i for i in range(201)]
    error = plt.figure()
    ax = error.gca(projection="3d")
    x = c_range
    y = d_range
    x, y = np.meshgrid(x,y)
    z = error_cd_fun(x,y)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel("c variable")
    plt.ylabel("d variable")
    plt.show()
#plot_cd_error()

def cd_gradient_descent(c_current, d_current):
    c_gradient = 0
    d_gradient = 0
    step_size = 0.000001
    for i in range(1,time_period):
        c_gradient += -2*rabbit[i]*wolf[i]*(((wolf[i]-wolf[i-1])/delta) - (c_current*rabbit[i]*wolf[i] - d_current*wolf[i]))
        d_gradient += 2*wolf[i]*(((wolf[i]-wolf[i-1])/delta) - (c_current*rabbit[i]*wolf[i] - d_current*wolf[i]))
    new_c = c_current - (step_size*c_gradient)
    new_d = d_current - (step_size*d_gradient)
    return new_c, new_d

def cd_gradient_iterate():
    c_current = 1
    d_current = 1
    error_monitor = []
    for i in range(20000):
        c_current, d_current = cd_gradient_descent(c_current, d_current)
        error_monitor.append(error_cd_fun(c_current,d_current))
    print("Minimum error = %s, and occurs at c = %s, d = %s" % (error_cd_fun(c_current,d_current),c_current,d_current))
    plt.plot(error_monitor, linewidth=2, label="Error")
    plt.title(r"c,d error function during gradient descent")
    plt.ylabel(r"Error")
    plt.xlabel(r"Iterations")
    plt.show()
#cd_gradient_iterate()
