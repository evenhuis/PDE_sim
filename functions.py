# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def initial_cond( lanes, N=601, smooth=None ):
	'''
	lanes   : number of lanes
	N	   : number of steps
	smooth  : smoothing the edges
	'''
	
	import numpy as np
	u0=np.zeros(N)
	if( max(lanes)==0):return u0
	
	u0[:N//3]	  =lanes[0]
	u0[N//3:2*N//3]=lanes[1]
	u0[2*N//3:]	=lanes[2]
	u0=u0/sum(u0)
	
	if( smooth):
		from scipy.ndimage import convolve
		stencil = np.exp(-(np.mgrid[-smooth:smooth+1]/(0.5*smooth))**2)
		stencil = stencil/sum(stencil)
		u0=convolve(u0,stencil,mode='reflect')
	return u0




def solve_step( u0, s, v, dt,dx ):
	'''
	u0 : solution at this timestep
	s  : diffussion coefficent
	v  : advection term
	nx : number of x-steps
	'''
	import numpy as np

	# time / space conversion factor
	c1 = dt/dx**2
	c2 = dt/dx/2
	ab  = np.zeros([3,len(u0)])
	ab[0] =-s*c1 -v*c2
	ab[1] = 1+2*s*c1   
	ab[2] =-s*c1  +v*c2

	# boundary condition
	if( isinstance(v, (list, tuple, np.ndarray)) ):
		ab[1,0]  = 1+(s*c1-v[ 0]*c2)
		ab[1,-1] = 1+(s*c1+v[-1]*c2)
	else:
		ab[1,0]  = 1+(s*c1-v*c2)
		ab[1,-1] = 1+(s*c1+v*c2)
		
	from scipy.linalg import solve_banded	
	u1= solve_banded( (1,1),ab,u0)
	return u1

def modulate(x,ymax):
	import numpy as np
	ax = np.abs(x)
	return ymax/(ymax+ax)

def KS_rg( theta, init, tgrid, ygrid, tmax=0.1 ):
	''' theta : parameters for the simulation
					D_att
					D_cells
					chemo_sens
					DT difference in time between starts of obs (t=0 in the observations)
					   and when the flow stopped

		 tgrid : time steps in the observations
		 ygrid : y grid for the oberations
	'''
	import numpy as np
	nt = len(tgrid)	
	ny = len(ygrid)
	dy = (ygrid[-1]-ygrid[0])/(ny-1)



	#print(ygrid[0],ygrid[1])
	nY = int(600/dy)+1
	iy0 = int(ygrid[ 0]/dy)
	iy1 = int(ygrid[-1]/dy)+1
	#print("dy",dy)
	#print("parts of y to take",iy0,iy1)
	#print("raw , out ",nY,ny)



	ca        = initial_cond(init[0:3] ,N=nY,smooth=3)   ## IC's of chemo-attractant
	ca_out    = np.zeros( [nt,ny] )


	#print(sum(ca))

	u         = initial_cond(init[3:6], N=nY,smooth=3) ## IC's of cells
	u_out     = np.zeros( [nt,ny] )


	t0 = 0
	t1 = tgrid[1]


	t0 = 0
	for j in range(0,nt):
		t1 = tgrid[j] 
		# work out the number of steps to take to get from t0 to t1
		nstep = int(np.floor(((t1-t0)/tmax)))+1	# this is the number of steps to get to next time point
		dt = (t1-t0)/nstep

		for k in range(nstep):
			ca = solve_step( ca, theta[0], 0., dt,dy )

			# calculate the derivatiee of the attractant
			dca     = (np.roll(ca,1)-np.roll(ca,-1))/(2*dy)
			dca[ 0] = (ca[1] -ca[ 0])/dy
			dca[-1] = (ca[-1]-ca[-2])/dy

			# solving for cells
			cs = -theta[2]*dca
			u = solve_step(u ,theta[1],cs*modulate(cs,125.),dt,dy )

		ca_out[j] = ca[iy0:iy1]
		u_out [j] =  u[iy0:iy1]

		t0 = t1


	return ca_out, u_out

	#i =0 
	#for j in range(nstep):
	#		ca = 
		




def KS(theta, nx = 100, nt = 1000, X = 600, T = 30):

	'''
	Solve the 1-D Keller Segel minimal model FDM, backward time, central space, no-flux BC's. Returns U, V, tgrid, ygrid
	theta : parameters of equation
	nx	: number of x-steps
	nt	: number of t-steps
	X	 : size of spatial domain
	T	 : size of time domain
	'''
	
	import numpy as np
	u_init  = initial_cond([0, 0,1], N=nx,smooth=3)	## IC's of cells
	ca_init = initial_cond([.5,0,1] ,N=nx,smooth=3)   ## IC's of chemo-attractant
	
	y = np.linspace(0 , X, nx+1)	  ## y-grid
	t = np.linspace(0 , T, nt+1)	  ## t-grid
	
	
	
	c_sol   = np.zeros([nt,nx])	  ## big cell matrix
	c_sol[0]=u_init
	u0 = u_init
	
	ca_sol   = np.zeros([nt,nx])	## big chemo-attactant matrix
	ca_sol[0]=ca_init
	ca0 = ca_init

	dx = y[1]-y[0]
	
	for i in range(1,nt):
		dt = t[i]-t[i-1]	

		#dx = 1
		#dt = 1
		# solving for chemo-attrantant
		ca1 = solve_step(ca0,theta[0],0,dt,dx )
		ca_sol[i]=ca1
		
		# calculate the derivatiee of the attractant
		dca1     = (np.roll(ca1,1)-np.roll(ca1,-1))/(dx)
		dca1[ 0] = (ca1[1] -ca1[ 0])/dx
		dca1[-1] = (ca1[-1]-ca1[-2])/dx
		
		# solving for cells
		u1= solve_step(u0,theta[1],-theta[2]*dca1,dt,dx )
		c_sol[i]=u1
	
		u0=u1
		ca0=ca1
		
	return c_sol, ca_sol, t, y





def regrid(U,t_model,y_model, theta,  t_obs, y_obs, X= 600, T = 30):
	
	'''
	Regrids PDE onto new (N x M) array
	U		  : Solution of U term (array)
	t_model	: time-grid array of U term	  e.g. (0,T)
	y_model	: spatial grid array of U term   e.g. (0, X)
	theta	  : list of time and spatial points to cut off
	t_obs	  : observed time-grid array
	y_obs	  : observed spatial grid array
	X		  : size of spatial domain
	T		  : size of time domain   
	'''
		## Regridding ##

	dt = t_model[1]-t_model[0]
	dy = y_model[1]-y_model[0]
	
	import numpy as np
	y_interp = np.arange(dy/2, X, dy )
	t_interp = np.arange(dt/2, T, dt )


	dT = t_obs[-1] - t_obs[0]  # total time observed
	
	from scipy import interpolate
	f = interpolate.interp2d(y_interp, t_interp, U, kind='cubic')

	tnew = np.linspace(theta[4],  theta[4]+dT, t_obs.shape[0]-1)	## Matching t-grid
	ynew = np.linspace(theta[5], X-theta[6], y_obs.shape[0]-1)	  ## Matching y-grid
	
	return f(ynew, tnew)
