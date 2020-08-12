"""
simple_muscle_model_functions.py
@author: Akira Nagamori
Last update: 8/12/20
Descriptions: 
    Defines functions associated with muscle model
"""

import numpy as np

# Determine the normalized initial muscle length, series-elastic element length, musculotendon unit length and the maixmum muscle length
def InitialLength(L0,alpha,L0T,Lce_length_initial):
    #Lce_length_initial in cm
    cT = 27.8;
    kT = 0.0047;
    LrT = 0.964;
    c1 = 23;
    k1 = 0.046;
    Lr1 = 1.17;
    
    PE1_force_max = c1 * k1 * np.log(np.exp((1 - Lr1)/float(k1))+1)*np.cos(alpha); #passive force from PE1 when the muscle length is at Lmax
    normalized_SE_Length_max = kT*np.log(np.exp(PE1_force_max/float(cT)/float(kT))-1)+LrT; # tendon length when passive force from PE1 at Lce = Lmax is applied. 
    Lmt_max = L0*np.cos(alpha)+ L0T; #the maximum length of musculotenodn unit
    SE_Length =  L0T * normalized_SE_Length_max; #tendon length in cm when the muscle is maximally stretched (i.e., Lce = Lmax)
    muscle_length_max = (Lmt_max - SE_Length); #muscle length in cm at the maximum length of musculotendon unit
    Lmax = muscle_length_max/float(L0)/float(np.cos(alpha)); #normalized maximum muscle fascicle length 
    
    PE1_force_initial = c1 * k1 * np.log(np.exp((Lce_length_initial/L0/Lmax - Lr1)/float(k1))+1)*np.cos(alpha); #PE1 force at muscle length = Lce_initial defined by the user
    normalized_SE_Length_initial = kT*np.log(np.exp(PE1_force_initial/float(cT)/float(kT))-1)+LrT #Normalized tendon length at PE1_force_initial
    Lse_initial = normalized_SE_Length_initial #initial tendon length
    Lmt_initial = Lce_length_initial*np.cos(alpha)+Lse_initial*L0T
    Lce_initial = Lce_length_initial/float(L0)

    return (Lce_initial,Lse_initial,Lmt_initial,Lmax)


# Two first-order dynamics from neural activation to muscle activation (Eq. 1&2)
def U2Act_function(U,A_int,A,step):
    if U >= A:
        T_U = 0.0343
    else:
        T_U = 0.047    
    A_int_dot = (U-A_int)/float(T_U)
    A_int = A_int_dot*step + A_int
    A_dot = (A_int - A)/float(T_U)
    A = A_dot*step + A    
    return (A_int,A)

# Force-length relationship (Eq. 3)
def FL_function(L):
    beta = 2.3;
    omega = 1.12;
    rho = 1.62;
    
    FL = np.exp(-np.power(abs((np.power(L,beta) - 1)/float(omega)),rho));
    return FL

# Concentric (i.e. shortening) force-velocity relationship (Eq. 4)
def FV_con_function(L,V):
    Vmax = -7.88;
    cv0 = 5.88;
    cv1 = 0;
    
    FVcon = (Vmax - V)/float(Vmax + (cv0 + cv1*L)*V);
    return FVcon
# Eccentric (i.e. lengthening) force-velocity relationship (Eq. 4)
def FV_ecc_function(L,V):
    av0 = -4.7;
    av1 = 8.41;
    av2 = -5.34;
    bv = 0.35;
    FVecc = (bv - (av0 + av1*L + av2*np.power(L,2))*V)/float(bv+V);

    return FVecc


# Passive force from passive element 1 (Eq. 6)
def F_pe_1_function(L,V):
    c1_pe1 = 23;
    k1_pe1 = 0.046;
    Lr1_pe1 = 1.17;
    eta = 0.01;
    
    Fpe1 = c1_pe1 * k1_pe1 * np.log(np.exp((L - Lr1_pe1)/float(k1_pe1))+1) + eta*V;

    return Fpe1

# Passive force from passive element 2 (Eq. 7)
def F_pe_2_function(Lce):
    c2_pe2 = -0.02;
    k2_pe2 = -21;
    Lr2_pe2 = 0.70;
    
    Fpe2 = c2_pe2*np.exp((k2_pe2*(Lce-Lr2_pe2))-1);
    return Fpe2

# Force-length relationship of series-elastic element (Eq. 8)
def F_se_function(Lse):
    cT_se = 27.8; 
    kT_se = 0.0047;
    LrT_se = 0.964;
    
    Fse = cT_se * kT_se * np.log(np.exp((Lse - LrT_se)/float(kT_se))+1);
    return Fse

# Equation for contraction dyamics (Eq. 5&9)
def contraction_dynamics(Lse,Lce,Vce,A,step,Lmax,F0,L0,mass,alpha):
  
    FL = FL_function(Lce);
    if Vce <= 0:
        FV = FV_con_function(Lce,Vce);
       
    else:
        FV = FV_ecc_function(Lce,Vce);
      
        
    FP1 = F_pe_1_function(Lce/float(Lmax),Vce);
    FP2 = F_pe_2_function(Lce);
    if FP2 > 0:
        FP2 = 0;     
        
    Fce = A*(FL*FV+FP2);
    if Fce < 0:
        Fce = 0;
    elif Fce > 1:
        Fce = 1;
        
    Fce = Fce + FP1;
    Fse = F_se_function(Lse);
    
    Force_muscle = Fce*F0;
    Force_tendon = Fse*F0;
    
    ddx = (Force_tendon*np.cos(alpha) - Force_muscle*np.power(np.cos(alpha),2))/float(mass) + (np.power(Vce*L0/float(100),2)*np.power(np.tan(alpha),2)/float(Lce*L0/float(100)))
    return ddx





