import matplotlib
matplotlib.use('Agg')
import matpy as mp
import numpy as np
import spinspy as spy
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys, os, shutil, tempfile
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

try: # Try using mpi
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
except:
    rank = 0
    num_procs = 1

# Load some information
dat = spy.get_params()
Nx = dat.Nx
Ny = dat.Ny
Nz = dat.Nz
dt = dat.tplot/(24.*3600) 
ct = len(glob.glob('q.*')) # Number of files of form q.*
ts = np.arange(0,(ct+1)*dt,dt)

if dat.method == 'nonlinear':
    qb = np.fromfile(dat.qb_file,'<d').reshape((Nx,Ny,Nz))

norms = np.zeros(ct+1)

cont = True
ii = rank
while cont:
    try:
        print('Processor {0:d} accessing q.{1:d}'.format(rank,ii))

        var_3d = np.fromfile('q.{0:d}'.format(ii),'<d').reshape((Nx,Ny,Nz))
        if dat.method == 'nonlinear':
            var_3d -= qb
        norms[ii] = np.linalg.norm(var_3d.ravel())

        ii += num_procs # Parallel, so skip a `bunch'
    except:
        ii += -num_procs 
        cont = False


# Determine how many outputs there were
final = np.zeros(num_procs)
max_is = np.zeros(num_procs)
final[rank] = ii
comm.Reduce([final, MPI.DOUBLE], [max_is,MPI.DOUBLE], op=MPI.MAX, root=0)
max_ii = np.max(max_is)

the_norms = np.zeros(ct+1)
comm.Reduce([norms, MPI.DOUBLE], [the_norms,MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:

    the_norms = the_norms[:max_ii]
    ts = ts[:max_ii]
    Dt = mp.FiniteDiff(ts,4)

    fig = plt.figure()
    ax  = plt.gca()
    axt = ax.twinx()
    ax.plot(ts,the_norms,'b',linewidth=2)
    ax.set_xlabel('t (days)')
    ax.set_ylabel('norm(q) (blue)')
    ax.set_yscale('log')

    deriv = Dt.dot(np.log(the_norms))
    deriv[np.isinf(deriv)] = np.nan
    axt.plot(ts,deriv,'r',linewidth=2)
    axt.set_ylim((0,0.1))
    axt.set_ylabel('d/dt(log(norm)) (red)')

    plt.draw()
    fig.set_tight_layout(True)

    np.savez('Diagnostics/perturbation_norms',ts=ts,norms=the_norms)
    plt.savefig('Diagnostics/perturbation_norms.png')
