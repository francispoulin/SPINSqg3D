#include "../TArray.hpp"
#include "../T_util.hpp"
#include "../Par_util.hpp"
#include "../Options.hpp"
#include "../Splits.hpp"
#include <blitz/array.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <random/normal.h>

using std::string;

string xgrid_filename, ygrid_filename, zgrid_filename, method_filename;
string q_filename, ub_filename, vb_filename, qxb_filename, qyb_filename;

int          Nx, Ny, Nz;                       // Number of points in x, y, z
double       Lx, Ly, Lz;                       // Grid lengths of x, y, z
double       N0, f0, beta, U0;                 // Buoyancy frequency, Coriolis parameters
double       t0, tf, tplot;                    // temporal parameters
double       fstrength, fcutoff, forder;       // filter parameters
int          plot_count;                       // track output number
int          kx, ky;                           // wavenumbers in x and y directions
double       Norm_x, Norm_y, Norm_z;           // Normalization factors
double       next_plot_time;                   // Time for next output
int          write_vels, write_psi;            // 0 if no

using namespace TArrayn;
using namespace Transformer;

using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;

using ranlib::Normal;

using namespace std;

// Normalization factors
#define Norm_x (2*M_PI/Lx)
#define Norm_y (2*M_PI/Ly)

// Structure to specify how to compute derivatives
struct Transgeom {
  
  TArrayn::S_EXP type;
  TArrayn::S_EXP q;

};

struct solnclass {

  // Nonlinear Fields
  vector<DTArray *> q;
  vector<DTArray *> psi;
  vector<DTArray *> u;
  vector<DTArray *> v;
  vector<DTArray *> qx;
  vector<DTArray *> qy;

  // Background State
  vector<DTArray *> ub;
  vector<DTArray *> vb;
  vector<DTArray *> qxb;
  vector<DTArray *> qyb;

  // Parameters for timestep
  double dx, dy, t, next_plot_time;
  double w0,w1,w2;
  double dt0;
  double dt1;
  double dt2;
  bool do_plot;

  double U0;
  double V0;

};

// Define flux function
struct methodclass {

  void (*flux)(solnclass &, DTArray &,
	       Trans1D &, Trans1D &, TransWrapper &, 
	       CTArray &, CTArray &, CTArray &, CTArray &,
	       Array<double,1> &);
  
};

/* Initialize Transform types */
Transgeom Sz;

// Blitz index placeholders
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

bool restarting = false;
double restart_time = 0;
int restart_sequence = -1;

// FJP: join Trans1D and TransWrapper into one class?
// FJP: specify filter parameters in config, like in SW
// FJP: options: 1) NL, 2) Lin, 3) NL pert
// FJP: grids are necessary for sponges and forcing, put in forcing?

void compute_dt(solnclass & soln);

void compute_weights2(solnclass & soln);

void compute_weights3(solnclass & soln);

void write_grids(DTArray & tmp, double Lx, double Ly, double Lz,
                                int Nx, int Ny, int Nz);

void nonlinear(solnclass & soln, DTArray & flux,
	       Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
	       CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	       Array<double,1> & ygrid); 

void linear(solnclass & soln, DTArray & flux,
	    Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
	    CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	    Array<double,1> & ygrid); 

void nonlinear_pert(solnclass & soln, DTArray & flux,
		    Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
		    CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
		    Array<double,1> & ygrid); 

void step_euler(solnclass & soln, solnclass & flux, methodclass & method,
		Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform,
		CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
		Array<double,1> & xgrid, Array<double,1> & ygrid);

void step_ab2(solnclass & soln, solnclass & flux, methodclass & method,
	      Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform,
	      CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	      Array<double,1> & xgrid, Array<double,1> & ygrid);

void step_ab3(solnclass & soln, solnclass & flux, methodclass & method,
	      Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform,
	      CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	      Array<double,1> & xgrid, Array<double,1> & ygrid);

int main(int argc, char ** argv) {

   // Initialize MPI
   MPI_Init(&argc, &argv);

   // To properly handle the variety of options, set up the boost
   // program_options library using the abbreviated interface in
   // ../Options.hpp

   options_init(); // Initialize options

   option_category("Grid Options");

   // Grid size
   add_option("Nx",&Nx,"Number of points in X");
   add_option("Ny",&Ny,"Number of points in Y");
   add_option("Nz",&Nz,"Number of points in Z");

   // Geometry: Periodic or Neumann in vertical
   string zgrid_type;
   add_option("type_z",&zgrid_type,
         "Grid type in Z.  Valid values are:\n"
         "   FOURIER: Periodic\n"
         "   REAL: Cosine expansion\n"
         "   CHEB: Chebyshev expansion");
   
   // Grid lengths
   add_option("Lx",&Lx,"X-length"); 
   add_option("Ly",&Ly,"Y-length");
   add_option("Lz",&Lz,"Z-length");


   // Defines for physical parameters
   option_category("Physical parameters");
   add_option("N0",&N0,"Buoyancy frequency");
   add_option("f0",&f0,"Coriolis parameter");
   add_option("beta",&beta,"beta parameter");
   add_option("U0",&U0,"Vortex velocity");

   // Timestep parameters
   option_category("Running options");
   add_option("write_vels",&write_vels,0,"If write u/v");
   add_option("write_psi",&write_psi,0,"If write psi");

   add_option("t0",&t0,"t0-initial time");
   add_option("tf",&tf,"tf-final time");
   add_option("tplot",&tplot,"tplot-frequency of output");

   // Flux Method to use 
   add_option("method",&method_filename,"Flux Method to use");

   // Filenames for Initial Conditions
   add_option("q_file",&q_filename,"Potential Vorticity filename");

   add_option("ub_file",&ub_filename,"u background filename");
   add_option("vb_file",&vb_filename,"v background filename");
   add_option("qxb_file",&qxb_filename,"qxb background filename");
   add_option("qyb_file",&qyb_filename,"qyb background filename");

   // Restarting
   option_category("Restart options"); 
   add_option("restart",&restarting,"Restart flag");
   add_option("restart_sequence",&restart_sequence,"Restart Sequence");
   
   option_category("Filtering options");
   add_option("f_strength",&fstrength,"filter strength");
   add_option("f_cutoff",&fcutoff,"filter cutoff");
   add_option("f_order",&forder,"fiter order");

   // Parse the options from the command line and config file
   options_parse(argc,argv);

   if (restarting) restart_time = tplot*restart_sequence;

   if (master()) printf("Parameters used in qg3D.cpp:\n");
   if (master()) printf("---------------------------\n");
   if (master()) printf("Nx    = %d, Ny = %d and Nz = %d \n",Nx,Ny,Nz);
   if (master()) printf("Lx    = %g, Ly = %g and Lz = %g\n",Lx,Ly,Lz);
   if (master()) printf("N0    = %6.2e\n",N0);
   if (master()) printf("f0    = %6.2e\n",f0);
   if (master()) printf("beta  = %6.2e\n",beta);
   if (master()) printf("t0    = %g\n",t0);
   if (master()) printf("tf    = %g\n",tf);
   if (master()) printf("tplot = %g\n",tplot);
   if (master()) printf("method = %s\n",method_filename.c_str());
   if (restarting) {
     if (master()) printf("restarting at t = %g\n",restart_time);
     if (master()) printf("restarting at index = %d\n",restart_sequence);
   }
   if (master()) printf("---------------------------\n");
   if (master()) printf(" \n");

   // Define Norm Factors and derivative functions
   if (zgrid_type == "FOURIER") {
     Sz.type = FOURIER;
     Sz.q = FOURIER;
     Norm_z = (2.*M_PI/Lz);
   } else if (zgrid_type == "REAL") {
     Sz.type = COSINE;
     Sz.q = COSINE;
     Norm_z = (1.*M_PI/Lz);
   } else if (zgrid_type == "CHEB") {
     Sz.type = CHEBY;
     Sz.q = CHEBY;
     Norm_z = (2./Lz);
   } else {
      if (master())
         fprintf(stderr,"Invalid option %s received for type_z\n",zgrid_type.c_str());
      MPI_Finalize(); exit(1);
   }

   // Iteration count
   int iterct;
   double mq;

   methodclass method;
   
   // Specify which flux to use
   if (method_filename == "nonlinear") {
     method.flux = & nonlinear;
   }
   else if (method_filename == "linear") {
     method.flux = & linear;
   }
   else if (method_filename == "nonlinear_pert") {
     method.flux = & nonlinear_pert;
   }

   TinyVector<int,3> local_lbound, local_extent;
   GeneralArrayStorage<3> local_storage;

   // Get parameters for local array storage
   local_lbound = alloc_lbound(Nx,Nz,Ny);
   local_extent = alloc_extent(Nx,Nz,Ny);
   local_storage = alloc_storage(Nx,Nz,Ny);

   /* Initialize solution and fluxes */
   solnclass soln, flux;
   
   soln.q.resize(1);
   soln.psi.resize(1);
   soln.u.resize(1);
   soln.v.resize(1);
   soln.qx.resize(1);
   soln.qy.resize(1);

   flux.q.resize(3);
   
   // Allocate the arrays used in the above
   soln.q[0] = new DTArray(local_lbound,local_extent,local_storage);
   soln.psi[0] = new DTArray(local_lbound,local_extent,local_storage);
   soln.u[0] = new DTArray(local_lbound,local_extent,local_storage);
   soln.v[0] = new DTArray(local_lbound,local_extent,local_storage);
   soln.qx[0] = new DTArray(local_lbound,local_extent,local_storage);
   soln.qy[0] = new DTArray(local_lbound,local_extent,local_storage);

   flux.q[0] = new DTArray(local_lbound,local_extent,local_storage);
   flux.q[1] = new DTArray(local_lbound,local_extent,local_storage);
   flux.q[2] = new DTArray(local_lbound,local_extent,local_storage);

   // Grid in x, y, z
   Array<double,1> xgrid(split_range(Nx)), ygrid(Ny), zgrid(Nz);
   xgrid = (ii+0.5)/Nx*Lx - Lx/2;
   ygrid = (ii+0.5)/Ny*Ly - Ly/2;
   zgrid = (ii+0.5)/Nz*Lz       ; // FJP: Allow for cheb?
   write_grids(*soln.q[0],Lx,Ly,Lz,Nx,Ny,Nz);

   // Read Initial Conditions and handle restarting
   
   // Initialize dt
   soln.dt0 = 1.0;
   soln.dt1 = 1.0;
   soln.dt2 = 1.0;
   soln.do_plot = false;

   if (restarting) {
     char filename[100];

     iterct = 0;
     plot_count = 1 + restart_sequence;
     soln.t = restart_time;
     soln.next_plot_time = tplot*plot_count;

     snprintf(filename,100,"q.%d",restart_sequence);

     if (master()) 
       fprintf(stdout,"Reading q from %s\n", filename);
     read_array(*soln.q[0],filename,Nx,Nz,Ny);
   }
   else {

     iterct = 0;
     plot_count = 1;
     soln.t = 0.0;
     soln.next_plot_time = tplot;

     if (master()) 
       fprintf(stdout,"Reading q from %s\n", q_filename.c_str());
     read_array(*soln.q[0],q_filename.c_str(),Nx,Nz,Ny);
   }

   // Read Basic State
   if (!(method_filename == "nonlinear")) {

     soln.ub.resize(1);
     soln.vb.resize(1);
     soln.qxb.resize(1);
     soln.qyb.resize(1);
   
     soln.ub[0] = new DTArray(local_lbound,local_extent,local_storage);
     soln.vb[0] = new DTArray(local_lbound,local_extent,local_storage);
     soln.qxb[0] = new DTArray(local_lbound,local_extent,local_storage);
     soln.qyb[0] = new DTArray(local_lbound,local_extent,local_storage);
     
     if (master()) 
        fprintf(stdout,"Reading ub from %s\n", ub_filename.c_str());
     read_array(*soln.ub[0],ub_filename.c_str(),Nx,Nz,Ny);
     if (master()) 
        fprintf(stdout,"Reading vb from %s\n", vb_filename.c_str());
     read_array(*soln.vb[0],vb_filename.c_str(),Nx,Nz,Ny);
     if (master()) 
        fprintf(stdout,"Reading qxb from %s\n", qxb_filename.c_str());
     read_array(*soln.qxb[0],qxb_filename.c_str(),Nx,Nz,Ny);
     if (master()) 
        fprintf(stdout,"Reading qby from %s\n", qyb_filename.c_str());
     read_array(*soln.qyb[0],qyb_filename.c_str(),Nx,Nz,Ny);
   }

   double tmp1, tmp2;
   
   if(!(method_filename == "nonlinear")) {
     tmp1 =  pvmax(*soln.u[0]);
     tmp2 = -pvmin(*soln.u[0]);
     if (tmp1 > tmp2) soln.U0 = tmp1;
     else soln.U0 = tmp2;

     tmp1 =  pvmax(*soln.v[0]);
     tmp2 = -pvmin(*soln.v[0]);
     if (tmp1 > tmp2) soln.V0 = tmp1;
    else soln.V0 = tmp2;
    }
    else {
      soln.U0 = 0.0;
      soln.V0 = 0.0;
    }
    
   // Necessary 1D FFT Transformers
   Trans1D X_xform(Nx,Nz,Ny,firstDim,FOURIER),
           Y_xform(Nx,Nz,Ny,thirdDim,FOURIER);

   // Necessary 3D FFT Transformers
   TransWrapper XYZ_xform(Nx,Nz,Ny,FOURIER,Sz.type,FOURIER);

   // Normalization factor for the 3D transform
   double norm_3d = XYZ_xform.norm_factor();

   // Spectral temporaries for the RHS
   GeneralArrayStorage<3> spec_ordering;
   spec_ordering.ordering() = XYZ_xform.get_complex_temp()->ordering();

   TinyVector<int,3> spec_lbound, spec_extent;
   spec_lbound = XYZ_xform.get_complex_temp()->lbound();
   spec_extent = XYZ_xform.get_complex_temp()->extent();

   // Allocating for the wavenumber matrices
   CTArray qh(spec_lbound,spec_extent,spec_ordering);
   CTArray K2(spec_lbound,spec_extent,spec_ordering);
   CTArray ik(spec_lbound,spec_extent,spec_ordering);
   CTArray il(spec_lbound,spec_extent,spec_ordering);


   // Store grid spacings 
   // BAS: Assumes uniform grid
   soln.dx = Lx/Nx;
   soln.dy = Ly/Ny;

   // K, L vectors
   Array<double,1> kvec(XYZ_xform.wavenums(firstDim)), 
     lvec(XYZ_xform.wavenums(thirdDim)),
     mvec(XYZ_xform.wavenums(secondDim));

   // Scale wavenumbers appropriately
   kvec = kvec*Norm_x;
   lvec = lvec*Norm_y;
   mvec = mvec*Norm_z*f0/N0;

   // Define wavenumbers
   K2 = -1.0/(pow(kvec(ii),2) + pow(lvec(kk),2) + pow(mvec(jj),2))/norm_3d;
   ik = complex<double>(0,1.0)*kvec(ii)/norm_3d;
   il = complex<double>(0,1.0)*lvec(kk)/norm_3d;

   // Change wavenumber at (0,0,0) from Inf to zero: Assumes mean of zero
   if (K2.lbound(firstDim) == 0 && K2.lbound(secondDim) == 0 && K2.lbound(thirdDim) == 0)
     {
       K2(0,0,0) = 0.0;
     }

   // Write to a file
   if (!restarting) {
     write_array(*soln.q[0],"q",0);
     if (!(write_psi == 0)) {
       write_array(*soln.psi[0],"psi",0);
     }
     if (!(write_vels == 0)) {
       write_array(*soln.u[0],"u",0);
       write_array(*soln.v[0],"v",0);
     }
   }
   mq = pvmax(*soln.q[0]);
   if (master()) printf("Wrote time %g (iter %d) with Max q = %g\n",soln.t,iterct,mq);

   // Advance Euler Step
   step_euler(soln, flux, method, X_xform, Y_xform, XYZ_xform,
	      K2, ik, il, qh, xgrid, ygrid);

   iterct++;
   soln.t = soln.t + soln.dt0;

   // Write maximum values
   mq = pvmax(*soln.q[0]);
   if (master()) printf("t = %g (%d outputs), max(q) = %g and dt = %g\n",soln.t,plot_count,mq,soln.dt0);
   
   //Advance AB2 Step
   step_ab2(soln, flux, method, X_xform, Y_xform, XYZ_xform,
	    K2, ik, il, qh, xgrid, ygrid); 

   //Advance time and increment
   iterct++;
   soln.t = soln.t + soln.dt0;

   // Write maximum values
   mq = pvmax(*soln.q[0]);
   if (master()) printf("t = %g (%d outputs), max(q) = %g and dt = %g\n",soln.t,plot_count,mq,soln.dt0);

   while (soln.t < tf) {

     //Advance Solution: AB3
     step_ab3(soln, flux, method, X_xform, Y_xform, XYZ_xform,
	      K2, ik, il, qh, xgrid, ygrid); 

     //Advance time and increment
     iterct++;
     soln.t = soln.t + soln.dt0;

     if (soln.do_plot) {
       write_array(*soln.q[0],"q",plot_count);
       if (!(write_psi == 0)) {
           write_array(*soln.psi[0],"psi",plot_count);
       }
       if (!(write_vels == 0)) {
           write_array(*soln.u[0],"u",plot_count);
           write_array(*soln.v[0],"v",plot_count);
       }
       soln.next_plot_time += tplot;
       soln.do_plot = false;
       plot_count++;
     }

     // Write maximum values
     mq = pvmax(*soln.q[0]);
     if (master()) printf("t = %g (%d outputs), max(q) = %g and dt = %g\n",soln.t,plot_count,mq,soln.dt0);

   }
   
   if (master()) printf("Finished at time %g!\n",soln.t);  

   MPI_Finalize();
   return 0;
}

void nonlinear(solnclass & soln, DTArray & flux,
	       Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
	       CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	       Array<double,1> & ygrid)
{

  // Make nice references for arrays used
  DTArray &q = *soln.q[0], &psi = *soln.psi[0];
  DTArray &u = *soln.u[0], &v = *soln.v[0];
  DTArray &qx = *soln.qx[0], &qy = *soln.qy[0];
  double norm_3d = XYZ_xform.norm_factor();

  // Compute 3D FFT of PV
  XYZ_xform.forward_transform(&q,FOURIER,Sz.q,FOURIER);        // compute q_hat    
  qh = *(XYZ_xform.get_complex_temp());

  // Gradient of PV
  *(XYZ_xform.get_complex_temp()) =  ik*qh;                    // calc qx
  XYZ_xform.back_transform(&qx,FOURIER,Sz.q,FOURIER);

  *(XYZ_xform.get_complex_temp()) =  il*qh;                    // calc qy
  XYZ_xform.back_transform(&qy,FOURIER,Sz.q,FOURIER);

  // Streamfunction
  qh = qh*K2;                                                  // calc psi_hat
  *(XYZ_xform.get_complex_temp()) = qh;
  XYZ_xform.back_transform(&psi,FOURIER,Sz.q,FOURIER);

  // Geostrophic Velocity
  *(XYZ_xform.get_complex_temp()) = -il*qh*norm_3d;            // calc u
  XYZ_xform.back_transform(&u,FOURIER,Sz.q,FOURIER);

  *(XYZ_xform.get_complex_temp()) =  ik*qh*norm_3d;            // calc v
  XYZ_xform.back_transform(&v,FOURIER,Sz.q,FOURIER);

  // Compute flux
  flux = u*qx + v*(qy + beta);                                 // u*qx + v*(qy + beta)

}

void linear(solnclass & soln, DTArray & flux,
	    Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
	    CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	    Array<double,1> & ygrid)
{

  // Make nice references for arrays used
  DTArray &q = *soln.q[0], &psi = *soln.psi[0];
  DTArray &u = *soln.u[0], &v = *soln.v[0];
  DTArray &qx = *soln.qx[0], &qy = *soln.qy[0];
  DTArray &ub = *soln.ub[0], &vb = *soln.vb[0];
  DTArray &qxb = *soln.qxb[0], &qyb = *soln.qyb[0];
  double norm_3d = XYZ_xform.norm_factor();

  // Compute 3D FFT of PV
  XYZ_xform.forward_transform(&q,FOURIER,Sz.q,FOURIER);        // compute q_hat    
  qh = *(XYZ_xform.get_complex_temp());

  // Gradient of PV
  *(XYZ_xform.get_complex_temp()) =  ik*qh;                    // calc qx
  XYZ_xform.back_transform(&qx,FOURIER,Sz.q,FOURIER);

  *(XYZ_xform.get_complex_temp()) =  il*qh;                    // calc qy
  XYZ_xform.back_transform(&qy,FOURIER,Sz.q,FOURIER);

  // Streamfunction
  qh = qh*K2;                                                  // calc psi_hat
  *(XYZ_xform.get_complex_temp()) = qh;
  XYZ_xform.back_transform(&psi,FOURIER,Sz.q,FOURIER);

  // Geostrophic Velocity
  *(XYZ_xform.get_complex_temp()) = -il*qh*norm_3d;            // calc u
  XYZ_xform.back_transform(&u,FOURIER,Sz.q,FOURIER);

  *(XYZ_xform.get_complex_temp()) =  ik*qh*norm_3d;            // calc v
  XYZ_xform.back_transform(&v,FOURIER,Sz.q,FOURIER);

  // Compute flux
  flux = ub*qx + vb*qy + u*qxb + v*(qyb + beta);               

}

void nonlinear_pert(solnclass & soln, DTArray & flux,
		    Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
		    CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
		    Array<double,1> & ygrid)
{

  // Make nice references for arrays used
  DTArray &q = *soln.q[0], &psi = *soln.psi[0];
  DTArray &u = *soln.u[0], &v = *soln.v[0];
  DTArray &qx = *soln.qx[0], &qy = *soln.qy[0];
  DTArray &ub = *soln.ub[0], &vb = *soln.vb[0];
  DTArray &qxb = *soln.qxb[0], &qyb = *soln.qyb[0];
  double norm_3d = XYZ_xform.norm_factor();

  // Compute 3D FFT of PV
  XYZ_xform.forward_transform(&q,FOURIER,Sz.q,FOURIER);        // compute q_hat    
  qh = *(XYZ_xform.get_complex_temp());

  // Gradient of PV
  *(XYZ_xform.get_complex_temp()) =  ik*qh;                    // calc qx
  XYZ_xform.back_transform(&qx,FOURIER,Sz.q,FOURIER);

  *(XYZ_xform.get_complex_temp()) =  il*qh;                    // calc qy
  XYZ_xform.back_transform(&qy,FOURIER,Sz.q,FOURIER);

  // Streamfunction
  qh = qh*K2;                                                  // calc psi_hat
  *(XYZ_xform.get_complex_temp()) = qh;
  XYZ_xform.back_transform(&psi,FOURIER,Sz.q,FOURIER);

  // Geostrophic Velocity
  *(XYZ_xform.get_complex_temp()) = -il*qh*norm_3d;            // calc u
  XYZ_xform.back_transform(&u,FOURIER,Sz.q,FOURIER);

  *(XYZ_xform.get_complex_temp()) =  ik*qh*norm_3d;            // calc v
  XYZ_xform.back_transform(&v,FOURIER,Sz.q,FOURIER);

  // Compute flux:
  flux = ub*qx + vb*qy + u*(qx + qxb) + v*(qy + qyb + beta); 

}

void step_euler(solnclass & soln, solnclass & flux, methodclass & method,
		Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform,
		CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
                Array<double,1> & xgrid, Array<double,1> & ygrid) {
  
  // Make nice references for arrays
  DTArray &q = *soln.q[0], &u = *soln.u[0], &v = *soln.v[0], &fluxq = *flux.q[0];

  // Compute flux
  method.flux(soln,fluxq,X_xform,Y_xform,XYZ_xform,K2,ik,il,qh,ygrid);

  // Compute new timestep, decrease for euler
  compute_dt(soln);
  soln.dt0 *= 0.1;

  // Advance solution
  q = q - soln.dt0*fluxq;
      
  // Exponential Filter
  filter3(q,XYZ_xform,FOURIER,Sz.q,FOURIER,fcutoff,forder,fstrength); 

}

void step_ab2(solnclass & soln, solnclass & flux, methodclass & method,
	      Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
	      CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	      Array<double,1> & xgrid, Array<double,1> & ygrid) {
  
  // Make nice references for arrays
  DTArray &q = *soln.q[0], &u = *soln.u[0], &v = *soln.v[0], &fluxq = *flux.q[1];

  // Compute flux
  method.flux(soln,fluxq,X_xform,Y_xform,XYZ_xform,K2,ik,il,qh,ygrid);

  // Compute new timestep, decrease for ab2
  compute_dt(soln);
  soln.dt0 *= 0.1;

  // Advance solution
  compute_weights2(soln);
  q = q - soln.w0*fluxq - soln.w1*(*flux.q[0]);
  //q = q - 0.5*soln.dt*(3*fluxq - *flux.q[0]);
      
  // Exponential Filter 
  filter3(q,XYZ_xform,FOURIER,Sz.q,FOURIER,fcutoff,forder,fstrength); 

}

void step_ab3(solnclass & soln, solnclass & flux, methodclass & method,
	      Trans1D & X_xform, Trans1D & Y_xform, TransWrapper & XYZ_xform, 
	      CTArray & K2, CTArray & ik, CTArray & il, CTArray & qh,
	      Array<double,1> & xgrid, Array<double,1> & ygrid) {
  
  // Make nice references for arrays
  DTArray &q = *soln.q[0], &u = *soln.u[0], &v = *soln.v[0], &fluxq = *flux.q[2];

  // Compute flux
  method.flux(soln,fluxq,X_xform,Y_xform,XYZ_xform,K2,ik,il,qh,ygrid);

  // Compute new timestep, full dt for ab3
  compute_dt(soln);

  // Advance solution
  compute_weights3(soln);
  q = q - soln.w0*fluxq - soln.w1*(*flux.q[1]) - soln.w2*(*flux.q[0]);
  //q = q - soln.dt/12*(23*fluxq - *flux.q[1]*16 + *flux.q[0]*5);
      
  // Exponential Filter
  filter3(q,XYZ_xform,FOURIER,Sz.q,FOURIER,fcutoff,forder,fstrength); 

  *flux.q[0] = *flux.q[1];
  *flux.q[1] = fluxq; 
}

void compute_dt(solnclass & soln) {  

    // Compute restrictions
    double tmp1, tmp2;
    double max_u, max_v;
    double dt;

    tmp1 =  pvmax(*soln.u[0]);
    tmp2 = -pvmin(*soln.u[0]);
    if (tmp1 > tmp2) max_u = tmp1 + soln.U0;
    else max_u = tmp2 + soln.U0;

    tmp1 =  pvmax(*soln.v[0]);
    tmp2 = -pvmin(*soln.v[0]);
    if (tmp1 > tmp2) max_v = tmp1 + soln.V0;
    else max_v = tmp2 + soln.V0;

    if (soln.dx/max_u < soln.dy/max_v) {
        dt = soln.dx/max_u/5;
    }
    else {
        dt = soln.dy/max_v/5;
    }

    if (dt < 20.0) {
      if (master())
	fprintf(stderr,"Time step is %g seconds, too small to continue. Force exit!\n",dt);
      MPI_Finalize(); exit(1);
    }
    else if(dt > 15000.0) {
        dt = 15000.0;
    }

    if (dt + soln.t >= soln.next_plot_time) {
        dt  = soln.next_plot_time - soln.t;
        soln.do_plot = true;
    }

    soln.dt2 = soln.dt1;
    soln.dt1 = soln.dt0;
    soln.dt0 = dt;

    return;
}

void compute_weights2(solnclass & soln) {

    double a,w,ts;
    double tnp = soln.t + soln.dt0;
    double tn  = soln.t;

    a  = soln.t - soln.dt1;
    ts = soln.t;
    w  = (  0.5*(tnp*tnp - tn*tn) 
            - a*(tnp-tn) )
          /(ts-a);
    soln.w0 = w;

    a  = soln.t;
    ts = soln.t - soln.dt1;
    w  = (  0.5*(tnp*tnp - tn*tn) 
            - a*(tnp-tn) )
          /(ts-a);
    soln.w1 = w;

    return;

}

void compute_weights3(solnclass & soln) {

    double a,b,w,ts;
    double tnp = soln.t + soln.dt0;
    double tn = soln.t;

    a  = soln.t - soln.dt1;
    b  = soln.t - soln.dt1 - soln.dt2;
    ts = soln.t;
    w  = (    (1./3)*(tnp*tnp*tnp - tn*tn*tn) 
            -    0.5*(a+b)*(tnp*tnp - tn*tn) 
                   + (a*b*(tnp-tn)) )
          /((ts-a)*(ts-b));
    soln.w0 = w;

    a  = soln.t;
    b  = soln.t - soln.dt1 - soln.dt2;
    ts = soln.t - soln.dt1;
    w  = (    (1./3)*(tnp*tnp*tnp - tn*tn*tn) 
            -    0.5*(a+b)*(tnp*tnp - tn*tn) 
                   + (a*b*(tnp-tn)) )
          /((ts-a)*(ts-b));
    soln.w1 = w;

    a  = soln.t;
    b  = soln.t - soln.dt1;
    ts = soln.t - soln.dt1 - soln.dt2;
    w  = (    (1./3)*(tnp*tnp*tnp - tn*tn*tn) 
            -    0.5*(a+b)*(tnp*tnp - tn*tn) 
                   + (a*b*(tnp-tn)) )
          /((ts-a)*(ts-b));
    soln.w2 = w;


    return;

}

void write_grids(DTArray & tmp, double Lx, double Ly, double Lz,
                                int Nx, int Ny, int Nz) {

   tmp = (ii + 0.5)/Nx*Lx + 0*jj + 0*kk - Lx/2;
   write_array(tmp,"xgrid");

   tmp = 0*ii + (kk + 0.5)/Ny*Ly + 0*jj - Ly/2;
   write_array(tmp,"ygrid");

   tmp = 0*ii + 0*kk + (jj + 0.5)/Nz*Lz;
   write_array(tmp,"zgrid");

}
