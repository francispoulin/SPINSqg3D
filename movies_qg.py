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

try: # Try using mpi
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
except:
    rank = 0
    num_procs = 1

## USER CHANGE THIS SECTION
out_direct = os.getcwd() + '/Videos'  # Where to put the movies
the_name = 'QG'                       # What to call the movies
out_suffix = 'mp4'                    # Movie type
mov_fps = 10                          # Framerate for movie
plt_var = ['q','psi']                     # Which variables to plot
cmap = 'ocean'
##

# If the the out_directory doesn't exist, create it
if (rank == 0) and not(os.path.exists(out_direct)):
    os.makedirs(out_direct)

# Load some information
dat = spy.get_params()
spy.local_data.disc_order = 'xyz'
Nx = dat.Nx
Ny = dat.Ny
Nz = dat.Nz
tplot = dat.tplot/(24.*3600)

x,y,z = spy.get_grid()
x = x/1e3
y = y/1e3
z = z/1e3

dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]

gridx = np.zeros(Nx+1)
gridx[1:] = x+dx/2
gridx[0] = x[0] - dx/2

gridy = np.zeros(Ny+1)
gridy[1:] = y+dy/2
gridy[0] = y[0] - dy/2

gridz = np.zeros(Nz+1)
gridz[1:] = z+dz/2
gridz[0] = z[0] - dz/2

# Prepare directories
if rank == 0:
    print('Video files will be saved in {0:s}'.format(out_direct))
    tmp_dir = tempfile.mkdtemp(dir=out_direct)
    fig_prefix = tmp_dir + '/' + the_name # path for saving frames
    out_prefix = out_direct + '/' + the_name # path for saving videos
    for proc in range(1,num_procs):
        comm.send(tmp_dir,dest=proc,tag=1)
        comm.send(fig_prefix,dest=proc,tag=2)
        comm.send(out_prefix,dest=proc,tag=3)
else:
    tmp_dir = comm.recv(source=0,tag=1)
    fig_prefix = comm.recv(source=0,tag=2)
    out_prefix = comm.recv(source=0,tag=3)

# Initialize the meshes
if not(dat.method == 'linear'):
    fig_xy = plt.figure(figsize=(6,5))
    fig_xy_ttl = fig_xy.suptitle('')
    QM_xy = plt.pcolormesh(gridx,gridy,np.zeros((Ny,Nx)),cmap=cmap)
    plt.axis('tight')
    cbar = plt.colorbar()

    fig_xz = plt.figure(figsize=(6,5))
    fig_xz_ttl = fig_xz.suptitle('')
    QM_xz = plt.pcolormesh(gridx,gridz,np.zeros((Nz,Nx)),cmap=cmap)
    plt.axis('tight')
    cbar = plt.colorbar()

fig_xy_p = plt.figure(figsize=(6,5))
fig_xy_p_ttl = fig_xy_p.suptitle('')
QM_xy_p = plt.pcolormesh(gridx,gridy,np.zeros((Ny,Nx)),cmap=cmap)
plt.axis('tight')
cbar = plt.colorbar()

fig_xz_p = plt.figure(figsize=(6,5))
fig_xz_p_ttl = fig_xz_p.suptitle('')
QM_xz_p = plt.pcolormesh(gridx,gridz,np.zeros((Nz,Nx)),cmap=cmap)
plt.axis('tight')
cbar = plt.colorbar()

for var in plt_var:

    ii = rank # parallel, so start where necessary
    cont = True

    # Load background state
    if dat.method == 'linear':
        if var == 'q':
            bg_xy = spy.reader(dat.qb_file,0,[0,-1],[0,-1],dat.Nz/2,force_name=True)
            bg_xz = spy.reader(dat.qb_file,0,[0,-1],dat.Ny/2,[0,-1],force_name=True)
        elif var == 'psi':
            bg_xy = spy.reader(dat.psib_file,0,[0,-1],[0,-1],dat.Nz/2,force_name=True)
            bg_xz = spy.reader(dat.psib_file,0,[0,-1],dat.Ny/2,[0,-1],force_name=True)

    while cont:
        try:
            var_3d_xy = spy.reader(var,ii,[0,-1],[0,-1],dat.Nz/2)
            var_3d_xz = spy.reader(var,ii,[0,-1],dat.Ny/2,[0,-1])
            print('Processor {0:d} accessing {1:s}.{2:d}'.format(rank,var,ii))

            if dat.method == 'linear':
                QM_xy_p.set_array(var_3d_xy.T.ravel())
                QM_xz_p.set_array(var_3d_xz.T.ravel())

                cv = np.max(abs(var_3d_xy.ravel()))
                QM_xy_p.set_clim((-cv,cv))
                cv = np.max(abs(var_3d_xz.ravel()))
                QM_xz_p.set_clim((-cv,cv))

            elif dat.method == 'nonlinear':
                QM_xy.set_array(var_3d_xy.T.ravel())
                QM_xz.set_array(var_3d_xz.T.ravel())
                QM_xy_p.set_array((var_3d_xy-bg_xy).T.ravel())
                QM_xz_p.set_array((var_3d_xz-bg_xz).T.ravel())

                cv = np.max(abs(var_3d_xy.ravel()))
                QM_xy.set_clim((-cv,cv))
                cv = np.max(abs(var_3d_xz.ravel()))
                QM_xz.set_clim((-cv,cv))
                cv = np.max(abs((var_3d_xy-bg_xy).ravel()))
                QM_xy_p.set_clim((-cv,cv))
                cv = np.max(abs((var_3d_xz-bg_xz).ravel()))
                QM_xz_p.set_clim((-cv,cv))

            if not(dat.method == 'linear'):
                QM_xy.changed()
                QM_xz.changed()
                fig_xy_ttl.set_text('{0:s} (full) : t = {1:.3g} days'.format(var,ii*tplot))
                fig_xz_ttl.set_text('{0:s} (full) : t = {1:.3g} days'.format(var,ii*tplot))

            QM_xy_p.changed()
            QM_xz_p.changed()
            fig_xy_p_ttl.set_text('{0:s} (pert) : t = {1:.3g} days'.format(var,ii*tplot))
            fig_xz_p_ttl.set_text('{0:s} (pert) : t = {1:.3g} days'.format(var,ii*tplot))

            plt.draw()

            if not(dat.method == 'linear'):
                fig_xy.savefig('{0:s}{1:s}-{2:05d}_xy.png'.format(fig_prefix,var,ii))
                fig_xz.savefig('{0:s}{1:s}-{2:05d}_xz.png'.format(fig_prefix,var,ii))
            fig_xy_p.savefig('{0:s}{1:s}-{2:05d}_xy_p.png'.format(fig_prefix,var,ii))
            fig_xz_p.savefig('{0:s}{1:s}-{2:05d}_xz_p.png'.format(fig_prefix,var,ii))
        
            ii += num_procs # Parallel, so skip a `bunch'
        except:
            cont = False

    # Have processor 0 wait for the others
    if num_procs > 1:
        if rank > 0:
            isdone = True
            comm.send(isdone, dest=0, tag=rank)
            print('Processor {0:d} done.'.format(rank))
        elif rank == 0:
            isdone = False
            for proc in range(1,num_procs):
                isdone = comm.recv(source=proc,tag=proc)

    # Now that the individual files have been written, we need to parse them into a movie.
    if rank == 0:

        # Make the videos
        if dat.method == 'linear':
            plot_types = ['xy_p','xz_p']
        else:
            plot_types = ['xy','xz','xy_p','xz_p']

        for plot_type in plot_types:
            in_name = '{0:s}{1:s}-%05d_{2:s}.png'.format(fig_prefix,var,plot_type)
            out_name = '{0:s}_{1:s}_{2:s}.{3:s}'.format(out_prefix,var,plot_type,out_suffix)
            cmd = ['ffmpeg', '-framerate', str(mov_fps), '-r', str(mov_fps),
                '-i', in_name, '-y', '-q', '1', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)

if rank == 0:
    print('--------')
    print('Deleting directory of intermediate frames.')
    shutil.rmtree(tmp_dir)
    print('Video creation complete.')
    print('Processor {0:d} done.'.format(rank))

