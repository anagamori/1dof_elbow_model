"""
model_1dof_wMuscle.py
Author: Akira Nagamori
Last update: 7/19/20
Descriptions: a kinematic model of a 1 DoF elbow joint with a pair of agonist and antagonist muscles 
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import muscle_model_functions as muscle_fns
import parameter_muscle_1 as m1
import parameter_muscle_2 as m2
plt.rcParams['pdf.fonttype'] = 42

# limb segment modeled as a cylinder with the length of 45 cm and the diameter of 5 cm
# the proximal segment attaches 5 cm from the end of the distal segment   
l = 0.45 # segment length [m]
r = 0.05/2 # segment radius [m] 
d = 0.05 # distance from the end to the attachment site [m]
d_c = 1/2 - d/l # distance from the center of the distal segment to the joint center
M = 0.1 # segment mass [kg]
I = 1/12*l**2*M + (d_c*l)**2*M + 1/4*M*r**2 # segment inertia
b = 0 # external viscosity
k = 0 # external stiffness

Fs = 10000 # sampling frequency 
step = 1/float(Fs) # time step
time_sim = np.arange(0,3,step) # time vector

r_m1 = 0.03 # moment arm of muscle 1 [m]
r_m2 = -0.03 # moment arm of muscle 2 [m]

# Define input to muscle 1 and 2
U_1 = np.zeros(len(time_sim),dtype = "float") # pulse input to muscle 1
U_1[int(0.5*Fs):int(3*Fs)] = 1
U_2 = np.zeros(len(time_sim),dtype = "float") # pulse input to muscle 2
U_2[int(0.7*Fs):int(1*Fs)] = 0.0

## Initialization 
# Kinematic model
theta = 0 # joint angle 
theta_dot = 0 # joint angular velocity

# vectors for data storage
theta_vec = np.zeros(len(time_sim),dtype = "float") 
theta_dot_vec = np.zeros(len(time_sim),dtype = "float")
theta_ddot_vec = np.zeros(len(time_sim),dtype = "float") # joint angular acceleration
T_vec = np.zeros(len(time_sim),dtype = "float") # total torque from two muscles

# Muscle model
# muscle 1
(Lce_1,Lse_1,Lmt_1,Lmax_1) =  muscle_fns.InitialLength(m1.L0, m1.alpha,m1.L0T,m1.Lce_initial)
U_eff_1 = 0;
f_int_slow_1 = 0;
f_eff_slow_1 = 0;
f_eff_slow_dot_1 = 0;
f_int_fast_1 = 0;
f_eff_fast_1 = 0;
f_eff_fast_dot_1 = 0;
Af_slow_1 = 0;
Af_fast_1 = 0;
Y_1 = 0;
S_1 = 0;    
Vce_1 = 0;
Force_tendon_1 = muscle_fns.F_se_function(Lse_1)*m1.F0; # initial tendon force
muscle_length_1 = Lce_1*m1.L0/float(100) # non-normalized muscle length 
muscle_velocity_1 = 0 # non-normalized muscle velocity

# vectors for data storage
U_eff_1_vec = np.zeros(len(time_sim))
Force_tendon_1_vec = np.zeros(len(time_sim))
muscle_length_1_vec = np.zeros(len(time_sim))
Lse_1_vec = np.zeros(len(time_sim))
Lmt_1_vec = np.zeros(len(time_sim))

# muscle 2
(Lce_2,Lse_2,Lmt_2,Lmax_2) =  muscle_fns.InitialLength(m2.L0, m2.alpha,m2.L0T,m2.Lce_initial)
U_eff_2 = 0;
f_int_slow_2 = 0;
f_eff_slow_2 = 0;
f_eff_slow_dot_2 = 0;
f_int_fast_2 = 0;
f_eff_fast_2 = 0;
f_eff_fast_dot_2 = 0;
Af_slow_2 = 0;
Af_fast_2 = 0;
Y_2 = 0;
S_2 = 0;    
Vce_2 = 0;
Force_tendon_2 = muscle_fns.F_se_function(Lse_2)*m2.F0 # initial tendon force
muscle_length_2 = Lce_2*m2.L0/float(100) # non-normalized muscle length 
muscle_velocity_2 = 0 # non-normalized muscle velocity

# vector for data storage
U_eff_2_vec = np.zeros(len(time_sim))
Force_tendon_2_vec = np.zeros(len(time_sim))
muscle_length_2_vec = np.zeros(len(time_sim))
Lse_2_vec = np.zeros(len(time_sim))
Lmt_2_vec = np.zeros(len(time_sim))


# Start for-loop for simulation
start_time = time.time()
for t in range(len(time_sim)):
    # Muscle 1
    U_eff_1 = muscle_fns.U_function(U_1[t],U_eff_1,step)      
    
    (W_slow_1,W_fast_1) = muscle_fns.weighting_function(U_eff_1,m1.U_slow_th,m1.U_fast_th)
    
    f_env_slow_1 = muscle_fns.U2f_slow_function(U_eff_1,m1.U_slow_th)        
    (f_int_slow_1,f_int_slow_dot_1) = muscle_fns.f_slow_function(f_int_slow_1,f_env_slow_1,f_env_slow_1,f_eff_slow_dot_1,Af_slow_1,Lce_1,step);
    (f_eff_slow_1,f_eff_slow_dot_1) = muscle_fns.f_slow_function(f_eff_slow_1,f_int_slow_1,f_env_slow_1,f_eff_slow_dot_1,Af_slow_1,Lce_1,step);
    
    f_env_fast_1 = muscle_fns.U2f_fast_function(U_eff_1,m1.U_fast_th)
    (f_int_fast_1,f_int_fast_dot_1) = muscle_fns.f_fast_function(f_int_fast_1,f_env_fast_1,f_env_fast_1,f_eff_fast_dot_1,Af_fast_1,Lce_1,step);
    (f_eff_fast_1,f_eff_fast_dot_1) = muscle_fns.f_fast_function(f_eff_fast_1,f_int_fast_1,f_env_fast_1,f_eff_fast_dot_1,Af_fast_1,Lce_1,step);
            
    Y_1 = muscle_fns.yield_function(Y_1,Vce_1,step)
    S_1 = muscle_fns.sag_function(S_1,f_eff_fast_1,step)    
    
    # Integration to get muscle length and velocity
    k_0_de = step*muscle_velocity_1;
    l_0_de = step*muscle_fns.contraction_dynamics(Lse_1,Lce_1,Vce_1,U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,step,m1.Lmax,m1.F0,m1.L0,m1.mass,m1.alpha);
    k_1_de = step*(muscle_velocity_1+l_0_de/2);
    l_1_de = step*muscle_fns.contraction_dynamics((Lmt_1 - (Lce_1+k_0_de/m1.L0)*m1.L0*np.cos(m1.alpha))/m1.L0T,Lce_1+k_0_de/m1.L0,Vce_1+l_0_de/m1.L0,U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,step,m1.Lmax,m1.F0,m1.L0,m1.mass,m1.alpha);
    k_2_de = step*(muscle_velocity_1+l_1_de/2);
    l_2_de = step*muscle_fns.contraction_dynamics((Lmt_1 - (Lce_1+k_1_de/m1.L0)*m1.L0*np.cos(m1.alpha))/m1.L0T,Lce_1+k_1_de/m1.L0,Vce_1+l_1_de/m1.L0,U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,step,m1.Lmax,m1.F0,m1.L0,m1.mass,m1.alpha);
    k_3_de = step*(muscle_velocity_1+l_2_de);
    l_3_de = step*muscle_fns.contraction_dynamics((Lmt_1 - (Lce_1+k_2_de/m1.L0)*m1.L0*np.cos(m1.alpha))/m1.L0T,Lce_1+k_2_de/m1.L0,Vce_1+l_2_de/m1.L0,U_eff_1,f_eff_slow_1,f_eff_fast_1,W_slow_1,W_fast_1,Y_1,S_1,step,m1.Lmax,m1.F0,m1.L0,m1.mass,m1.alpha);
    
    muscle_length_1 = muscle_length_1 + 1/6*(k_0_de+2*k_1_de+2*k_2_de+k_3_de);
    muscle_velocity_1 = muscle_velocity_1 + 1/6*(l_0_de+2*l_1_de+2*l_2_de+l_3_de);
    Vce_1 = muscle_velocity_1/float(m1.L0/100)
    Lce_1 = muscle_length_1/float(m1.L0/100)
    Lse_1 = (Lmt_1 - Lce_1*m1.L0*np.cos(m1.alpha))/float(m1.L0T) # Eq. 16
    
    Force_tendon_1 = muscle_fns.F_se_function(Lse_1)*m1.F0;    
    T_1 = Force_tendon_1*r_m1 # torque from muscle 1
    
    # Muscle 2
    U_eff_2 = muscle_fns.U_function(U_2[t],U_eff_2,step)  
    
    (W_slow_2,W_fast_2) = muscle_fns.weighting_function(U_eff_2,m2.U_slow_th,m2.U_fast_th)
    
    f_env_slow_2 = muscle_fns.U2f_slow_function(U_eff_2,m2.U_slow_th)        
    (f_int_slow_2,f_int_slow_dot_2) = muscle_fns.f_slow_function(f_int_slow_2,f_env_slow_2,f_env_slow_2,f_eff_slow_dot_2,Af_slow_2,Lce_2,step);
    (f_eff_slow_2,f_eff_slow_dot_2) = muscle_fns.f_slow_function(f_eff_slow_2,f_int_slow_2,f_env_slow_2,f_eff_slow_dot_2,Af_slow_2,Lce_2,step);
    
    f_env_fast_2 = muscle_fns.U2f_fast_function(U_eff_2,m2.U_fast_th)
    (f_int_fast_2,f_int_fast_dot_2) = muscle_fns.f_fast_function(f_int_fast_2,f_env_fast_2,f_env_fast_2,f_eff_fast_dot_2,Af_fast_2,Lce_2,step);
    (f_eff_fast_2,f_eff_fast_dot_2) = muscle_fns.f_fast_function(f_eff_fast_2,f_int_fast_2,f_env_fast_2,f_eff_fast_dot_2,Af_fast_2,Lce_2,step);
            
    Y_2 = muscle_fns.yield_function(Y_2,Vce_2,step)
    S_2 = muscle_fns.sag_function(S_2,f_eff_fast_2,step)    
    
    # Integration to get muscle length and velocity
    k_0_de = step*muscle_velocity_2;
    l_0_de = step*muscle_fns.contraction_dynamics(Lse_2,Lce_2,Vce_2,U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,step,m2.Lmax,m2.F0,m2.L0,m2.mass,m2.alpha);
    k_1_de = step*(muscle_velocity_2+l_0_de/2);
    l_1_de = step*muscle_fns.contraction_dynamics((Lmt_2 - (Lce_2+k_0_de/m2.L0)*m2.L0*np.cos(m2.alpha))/m2.L0T,Lce_2+k_0_de/m2.L0,Vce_2+l_0_de/m2.L0,U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,step,m2.Lmax,m2.F0,m2.L0,m2.mass,m2.alpha);
    k_2_de = step*(muscle_velocity_2+l_1_de/2);
    l_2_de = step*muscle_fns.contraction_dynamics((Lmt_2 - (Lce_2+k_1_de/m2.L0)*m2.L0*np.cos(m2.alpha))/m2.L0T,Lce_2+k_1_de/m2.L0,Vce_2+l_1_de/m2.L0,U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,step,m2.Lmax,m2.F0,m2.L0,m2.mass,m2.alpha);
    k_3_de = step*(muscle_velocity_2+l_2_de);
    l_3_de = step*muscle_fns.contraction_dynamics((Lmt_2 - (Lce_2+k_2_de/m2.L0)*m2.L0*np.cos(m2.alpha))/m2.L0T,Lce_2+k_2_de/m2.L0,Vce_2+l_2_de/m2.L0,U_eff_2,f_eff_slow_2,f_eff_fast_2,W_slow_2,W_fast_2,Y_2,S_2,step,m2.Lmax,m2.F0,m2.L0,m2.mass,m2.alpha);
    muscle_length_2 = muscle_length_2 + 1/6*(k_0_de+2*k_1_de+2*k_2_de+k_3_de);
    muscle_velocity_2 = muscle_velocity_2 + 1/6*(l_0_de+2*l_1_de+2*l_2_de+l_3_de);
    Vce_2 = muscle_velocity_2/float(m2.L0/100)
    Lce_2 = muscle_length_2/float(m2.L0/100)    
    Lse_2 = (Lmt_2 - Lce_2*m2.L0*np.cos(m2.alpha))/float(m2.L0T) # Eq. 16
    
    Force_tendon_2 = muscle_fns.F_se_function(Lse_2)*m2.F0;  
    T_2 = Force_tendon_2*r_m2 # torque from muscle 2
    
    # Kinematic model
    T = T_1 + T_2
    theta_ddot = (T - b*theta_dot - k*theta)/I
    theta_dot = theta_ddot*step + theta_dot
    theta = theta_dot*step + theta
    
    
    # Update the musculotendon length
    if t > 1:
        Lmt_1 = Lmt_1 - r_m1*100*(theta-theta_vec[t-1])
        Lmt_2 = Lmt_2 - r_m2*100*(theta-theta_vec[t-1])
    
    # save variables 
    T_vec[t] = T
    theta_vec[t] = theta
    theta_dot_vec[t] = theta_dot
    theta_ddot_vec[t] = theta_ddot
    
    U_eff_1_vec[t] = U_eff_1
    U_eff_2_vec[t] = U_eff_2
    
    Force_tendon_1_vec[t] = Force_tendon_1;
    muscle_length_1_vec[t] = muscle_length_1 
    Force_tendon_2_vec[t] = Force_tendon_2;
    muscle_length_2_vec[t] = muscle_length_2
    Lmt_1_vec[t] = Lmt_1
    Lmt_2_vec[t] = Lmt_2    
    Lse_1_vec[t] = Lse_1
    Lse_2_vec[t] = Lse_2    

end_time = time.time()
print(end_time - start_time)

# Plot data
fig = plt.figure()
ax1 = plt.subplot(3,1,1)
ax1.plot(time_sim,U_1,time_sim,U_2)
ax1.set_ylabel('Input')
ax1.legend(['m1','m2'])
ax2 = plt.subplot(3,1,2)
ax2.plot(time_sim,T_vec)
ax2.set_xlim([0,np.max(time_sim)])
ax2.set_ylabel('Joint Troque\n(N)')
ax3 = plt.subplot(3,1,3)
ax3.plot(time_sim,np.degrees(theta_vec))
ax3.set_xlim([0,np.max(time_sim)])
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('Joint Angle\n(degrees)')
plt.show()
fig.savefig("Output.pdf", bbox_inches='tight',transparent=True)

fig = plt.figure()
ax4 = plt.subplot(3,1,1)
ax4.plot(time_sim,muscle_length_1_vec*100/m1.L0,time_sim,muscle_length_2_vec*100/m1.L0)
ax4.set_xlim([0,np.max(time_sim)])
ax4.legend(['m1','m2'])
ax4.set_xlabel('Time (sec)')
ax4.set_ylabel('Muscle Length \n(L0)')
ax5 = plt.subplot(3,1,2)
ax5.plot(time_sim,Lse_1_vec,time_sim,Lse_2_vec)
ax5.set_xlim([0,np.max(time_sim)])
ax5.legend(['m1','m2'])
ax5.set_xlabel('Time (sec)')
ax5.set_ylabel('Tendon Length \n(L0T)')
ax6 = plt.subplot(3,1,3)
ax6.plot(time_sim,Force_tendon_1_vec,time_sim,Force_tendon_2_vec)
ax6.set_xlim([0,np.max(time_sim)])
ax6.set_xlabel('Time (sec)')
ax6.set_ylabel('Tendon Force \n(N)')

 