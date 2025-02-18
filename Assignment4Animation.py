import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Define symbolic variables for lengths and masses
r1, r2, r3 = 1, 1, 1  # Lengths of the rods
u1, u2, u3 = 3, 2, 1  # Masses
g = 9.8  # Acceleration due to gravity

# Define time variable
t = sp.symbols('t')

# Define angles as functions of time
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta3 = sp.Function('theta3')(t)

# Define the time derivatives of the angles
theta1_dot = sp.diff(theta1, t)
theta2_dot = sp.diff(theta2, t)
theta3_dot = sp.diff(theta3, t)

# Define position vectors X and Y for velocities
X = sp.Matrix([r1 * theta1_dot, r2 * theta2_dot, r3 * theta3_dot])
Y = sp.Matrix([r1 * sp.cos(theta1), r2 * sp.cos(theta2), r3 * sp.cos(theta3)])

# Define the mass matrix
M = sp.Matrix([[u1, u2 * sp.cos(theta2 - theta1), u3 * sp.cos(theta3 - theta1)],
               [u2 * sp.cos(theta2 - theta1), u2, u3 * sp.cos(theta3 - theta2)],
               [u3 * sp.cos(theta3 - theta1), u3 * sp.cos(theta3 - theta2), u3]])

# Compute the kinetic term
kinetic_term = (1/2) * (X.T * M * X)[0]

# Compute the potential term
potential_term = g * (Y.T * sp.Matrix([u1, u2, u3]))[0]

# Define the Lagrangian L
L = sp.simplify(kinetic_term + potential_term)

# Define the generalized coordinates
q = [theta1, theta2, theta3]
q_dot = [theta1_dot, theta2_dot, theta3_dot]

# Initialize an empty list to hold equations of motion
equations_of_motion = []

# Loop through each generalized coordinate to find the equations of motion
for i in range(len(q)):
    # Compute the partial derivatives
    dL_dq = sp.diff(L, q[i])
    dL_dq_dot = sp.diff(L, q_dot[i])
    
    # Apply the Euler-Lagrange equation
    equation = sp.diff(dL_dq_dot, t) - dL_dq
    equations_of_motion.append(sp.simplify(equation))

# Define the second derivatives
theta1_ddot = sp.Function('theta1_ddot')(t)
theta2_ddot = sp.Function('theta2_ddot')(t)
theta3_ddot = sp.Function('theta3_ddot')(t)

# Substitute the second derivatives into the equations
eqs_substituted = [eq.subs({sp.diff(theta1, t, 2): theta1_ddot,
                            sp.diff(theta2, t, 2): theta2_ddot,
                            sp.diff(theta3, t, 2): theta3_ddot}) 
                   for eq in equations_of_motion]

# Solve for the second derivatives
solved_accelerations = sp.solve(eqs_substituted, (theta1_ddot, theta2_ddot, theta3_ddot))

# Extract the solutions
theta1_ddot_eq = solved_accelerations[theta1_ddot]
theta2_ddot_eq = solved_accelerations[theta2_ddot]
theta3_ddot_eq = solved_accelerations[theta3_ddot]

# Extract numerical functions for ODE solver
def equations_of_motion_func(t, y):
    theta1_val, theta2_val, theta3_val, theta1_dot_val, theta2_dot_val, theta3_dot_val = y
    
    # Create a dictionary for substitutions
    subs_dict = {
        theta1: theta1_val,
        theta2: theta2_val,
        theta3: theta3_val,
        theta1_dot: theta1_dot_val,
        theta2_dot: theta2_dot_val,
        theta3_dot: theta3_dot_val
    }
    
    # Evaluate the acceleration equations
    theta1_ddot_val = theta1_ddot_eq.subs(subs_dict).evalf()
    theta2_ddot_val = theta2_ddot_eq.subs(subs_dict).evalf()
    theta3_ddot_val = theta3_ddot_eq.subs(subs_dict).evalf()
    
    return [theta1_dot_val, theta2_dot_val, theta3_dot_val, 
            float(theta1_ddot_val), float(theta2_ddot_val), float(theta3_ddot_val)]

# Set the initial conditions
initial_conditions = [np.pi/3, np.pi/3, np.pi/3, 0, 0, 0]

# Define the time span for the simulation
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE
solution = solve_ivp(equations_of_motion_func, t_span, initial_conditions, t_eval=t_eval)

# Extract the angles from the solution
theta1_sol = solution.y[0]
theta2_sol = solution.y[1]
theta3_sol = solution.y[2]

# Calculate positions of the masses
x1 = r1 * np.sin(theta1_sol)
y1 = -r1 * np.cos(theta1_sol)

x2 = x1 + r2 * np.sin(theta2_sol)
y2 = y1 - r2 * np.cos(theta2_sol)

x3 = x2 + r3 * np.sin(theta3_sol)
y3 = y2 - r3 * np.cos(theta3_sol)

fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 1)

line1, = ax.plot([], [], 'k-', lw=2, label='Rod 1')
line2, = ax.plot([], [], 'k-', lw=2, label='Rod 2')
line3, = ax.plot([], [], 'k-', lw=2, label='Rod 3')

mass1, = ax.plot([], [], 'ro', markersize=8, label='Mass 1')
mass2, = ax.plot([], [], 'go', markersize=8, label='Mass 2')
mass3, = ax.plot([], [], 'bo', markersize=8, label='Mass 3')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    mass1.set_data([], [])
    mass2.set_data([], [])
    mass3.set_data([], [])
    return line1, line2, line3, mass1, mass2, mass3

def update(frame):
    # Update rods
    line1.set_data([0, x1[frame]], [0, y1[frame]])
    line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    line3.set_data([x2[frame], x3[frame]], [y2[frame], y3[frame]])

    # Update masses (using tuple to fix the deprecation warning)
    mass1.set_data((x1[frame],), (y1[frame],))
    mass2.set_data((x2[frame],), (y2[frame],))
    mass3.set_data((x3[frame],), (y3[frame],))

    return line1, line2, line3, mass1, mass2, mass3

# Adjust the interval to control the speed of the animation (lower value -> faster animation)
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=30)

plt.title('3-Link Pendulum Animation')
plt.grid()
plt.show()