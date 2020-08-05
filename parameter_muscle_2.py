"""
parameter_muscle_2.py
@author: Akira Nagamori
Last update: 8/5/20
Descriptions: 
    Defines model parameters associated with muscle 2
"""
import numpy as np
import muscle_model_functions as muscle_fns

L0 = 21.5 # muscle optimal length [cm]
alpha = 0*np.pi/float(180) # pennation angle [rad]
L0T = 15.5*1.05; # optimal length of a series-elastic element [cm]
Lce_initial = L0; # initial muscle length [cm]

# compute the initial normlized length of muscle, the series elastic element and a musculotendon unit 
# as well as the maximum muscle length 
(Lce,Lse,Lmt,Lmax) =  muscle_fns.InitialLength(L0, alpha, L0T, Lce_initial)
  
F0 = 434 # Maximum force output [N]
sigma = 31.8 # muscle specific tension [N/cm^2]
PCSA = F0/sigma  # physiological cross-sectional area [cm^2]
density = 1.06 # muscle densitiy [g/cm^3]
mass = PCSA*density*L0/1000 # muscle mass [kg]

Ur = 0.8 # activation level at which all units are recruited 
F_pcsa_slow = 0.5 # fraction of PCSA of slow-twitch unit
U_slow_th = 0.001 # recruitment threshold of slow-twitch unit 
U_fast_th = Ur*F_pcsa_slow # recruitment threshold of fast-twitch unit
