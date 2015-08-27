'''
config_qg3D.py

This driver will specify the spins.conf file required by qg3D.x.
This is part of SPINSqg, a subset of SPINS created by C. Subich.

This will be located on github at the following address:

...

Created by F.J.Poulin and B.A. Storer, August 18, 2015

'''
 
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# Nondimensional parameters
Bu = 0.1               # Burger number
Ro = -0.2              # Rossby number
al = 20.               # Rato of frequencies

'''
type_z:

 Vertical Geometry type
 'FOURIER' for periodic 
 'REAL' for top and bottom

method:

There are three different physical models that can be solved:

 -) nonlinear
 -) linear
 -) nonlinear_pert

This is set in the method parameter.

'''

# Spatial Resolution
Nx, Ny, Nz = 128,128,128

# Time parameters
mins  = 60.0
days  = 3600*24.0 
t0    = 0
tf    = 400*days
tplot = 1.0*days
nct   = np.ceil(tf/tplot)         

# Filter parameters
f_strength = 20.0
f_cutoff   = 0.6
f_order    = 2.0

# Inital Conditions can be set to be BT_Vortex, Meddy, ...
type_z = 'REAL'
method = 'linear'
IC     = 'BT_Vortex'
case   = 'data_'
write_vels = 0
write_psi = 0

# Lengths of vortex: 
Lh = 32.e3
Lv = 450.

# Physical Parameters: FJP: sqrt?
N0   = 3.2*1e-3
beta = 0e-11

if IC == 'BT_Vortex':

    # Lengths of Domain
    Lx, Ly, Lz = 300.e3, 300.e3, 15.e3 

    # Grid
    x = (Lx/Nx)*np.arange(0.5,Nx) - Lx/2
    y = (Ly/Ny)*np.arange(0.5,Ny) - Ly/2
    z = (Lz/Nz)*np.arange(0.5,Nz) - Lz 
    x,y,z = np.meshgrid(x,y,z,indexing='ij')

    f0 = N0/al
    U0 = f0*Lh*Ro
    
    x0,y0,z0 = (0,0,-Lz/2)
    x, y, z = (x - x0)/Lh, (y - y0)/Lh, (z - z0)/Lv
    gaus = np.exp( - x**2 - y**2 )
    
    qb =  U0/Lh*(x**2 + y**2 - 1.)*gaus

    # FJP: things to try
    # 1) change linear to NL, does it grow normally?
    # 2) try nl_pert
    # 3) ... 
    
    if method == 'linear' or method == 'nonlinear_pert':
        ub =  0.5*U0*y*gaus
        vb = -0.5*U0*x*gaus
        qxb=  2.0*U0/Lh**2*x*(2. - x**2 - y**2)*gaus
        qyb=  2.0*U0/Lh**2*y*(2. - x**2 - y**2)*gaus

        print U0, Lh
        print Lx/Nx, Ly/Ny
        print np.amax(abs(ub.ravel()))
        print np.amax(abs(vb.ravel()))
    
        
elif IC == 'Meddy':

    # Lengths of Domain
    Lx, Ly, Lz = 200.e3, 200.e3, 2.4e3

    # Grid
    x = (Lx/Nx)*np.arange(0.5,Nx) - Lx/2
    y = (Ly/Ny)*np.arange(0.5,Ny) - Ly/2
    z = (Lz/Nz)*np.arange(0.5,Nz) - Lz 
    x,y,z = np.meshgrid(x,y,z,indexing='ij')


    f0 = N0*Lv/Lh/np.sqrt(Bu)
    U0 = f0*Lh*Ro
    
    x0,y0,z0 = (0,0,-Lz/2)    
    x, y, z = (x - x0)/Lh, (y - y0)/Lh, (z - z0)/Lv
    gaus = np.exp( - x**2 - y**2 - z**2)
    
    qb = U0/Lh*(x**2 + y**2 + z**2/Bu - 1. - 0.5/Bu)*gaus

    if method == 'linear' or method == 'nonlinear_pert':
        ub =  0.5*U0*y*gaus
        vb = -0.5*U0*x*gaus
        qxb = 2.*U0/Lh**2*x*(2. - x**2 - y**2 - z**2/Bu + 0.5/Bu)*gaus
        qyb = 2.*U0/Lh**2*y*(2. - x**2 - y**2 - z**2/Bu + 0.5/Bu)*gaus

else:
    print "No ICs are specified."
    print "Maybe you want to write your own?"
    sys.exit()

# Perturb PV: FJP: make amplitude a parameter?
q = 1e-6*np.random.randn(Nx,Ny,Nz)*gaus
if method == 'nonlinear':
    q += qb 

# Write background state
qb.tofile(case+'qb')
q.tofile(case+'q')

# Write other fields if necessary
if method == 'linear' or method == 'nonlinear_pert':
    ub.tofile(case+'ub')
    vb.tofile(case+'vb')
    qxb.tofile(case+'qxb')
    qyb.tofile(case+'qyb')

# Write variables to a file
text_file = open("spins.conf", "w")

text_file.write("type_z = %s \n" % type_z)

text_file.write("method = %s \n" % method)

text_file.write("Nx         = {0:d} \n".format(Nx))
text_file.write("Ny         = {0:d} \n".format(Ny))
text_file.write("Nz         = {0:d} \n".format(Nz))

text_file.write("Lx         = {0:.12g} \n".format(Lx)) 
text_file.write("Ly         = {0:.12g} \n".format(Ly))
text_file.write("Lz         = {0:.12g} \n".format(Lz))

text_file.write("N0         = {0:.12g} \n".format(N0))
text_file.write("f0         = {0:.12g} \n".format(f0))
text_file.write("beta       = {0:.12g} \n".format(beta))
text_file.write("U0         = {0:.12g} \n".format(U0))

text_file.write("Lh         = {0:.12g} \n".format(Lh))
try:
    text_file.write("Lv         = {0:.12g}\n".format(Lv))
except:
    pass
text_file.write("Bu         = {0:.12g} \n".format(Bu))
text_file.write("Ro         = {0:.12g} \n".format(Ro))

text_file.write("t0         = {0:.12g} \n".format(t0))
text_file.write("tf         = {0:.12g} \n".format(tf))
text_file.write("tplot      = {0:.12g} \n".format(tplot))
text_file.write("nct        = {0:.12g} \n".format(nct))

text_file.write("f_strength = {0:.12g} \n".format(f_strength))
text_file.write("f_cutoff   = {0:.12g} \n".format(f_cutoff))
text_file.write("f_order    = {0:.12g} \n".format(f_order))

text_file.write("xgrid      = {0:s}\n".format('xgrid'))
text_file.write("ygrid      = {0:s}\n".format('ygrid'))
text_file.write("zgrid      = {0:s}\n".format('zgrid'))

text_file.write("q_file     = {0:s}\n".format(case+'q'))
text_file.write("qb_file    = {0:s}\n".format(case+'qb'))

text_file.write("write_vels = {0:d} \n".format(write_vels))
text_file.write("write_psi  = {0:d} \n".format(write_psi))

if method == 'linear' or method == 'nonlinear_pert':

    text_file.write("ub_file  = {0:s}\n".format(case+'ub'))
    text_file.write("vb_file  = {0:s}\n".format(case+'vb'))
    text_file.write("qxb_file = {0:s}\n".format(case+'qxb'))
    text_file.write("qyb_file = {0:s}\n".format(case+'qyb'))

text_file.write("restart = false\n")
text_file.write("restart_sequence = 0\n")

text_file.close()
