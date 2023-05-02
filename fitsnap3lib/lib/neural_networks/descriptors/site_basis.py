import numpy as np
import torch
import math
import torch.nn.functional as F

pi = 3.1415926

def spherical_harmonic(l,m,theta,phi):
    Y = {mi:None for mi in range(-l,l+1)}
    if l == 0:
        Y[0] =  (1/2) * torch.sqrt(1/pi) * torch.ones(torch.shape(theta)[0])

    elif l == 1:
        Y = {mi:None for mi in range(-l,l+1)}
        Y[0] = (1/2) * torch.sqrt(3/pi) * torch.cos(theta)
        Y[1] = (-1/2) * torch.sqrt(3/(2*pi)) * torch.sin(theta) * torch.exp( (1j) * phi)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])

    elif l == 2:
        Y[0] = (1/4)*torch.sqrt(5/pi) * ( (3 * ((torch.cos(theta))**2)  )  - 1  )
        Y[1] = (-1/2)*torch.sqrt(15/(2*pi)) * torch.exp( (1j) * phi) * torch.sin(theta) * torch.cos(theta)
        Y[2] = (1/4) *torch.sqrt(15/(2*pi)) * torch.exp( (2j) * phi) * ((torch.sin(theta))**2)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])
        Y[-2] = ( (-1)**2 ) * torch.conj(Y[2])

    elif l == 3:
        Y[0] = (1/4)*torch.sqrt(7/pi) * ( (5 * ((torch.cos(theta))**3) ) - ( 3* torch.cos(theta)) )
        Y[1] = (-1/8)*torch.sqrt(21/pi) *  torch.exp( (1j) * phi) * torch.sin(theta) * ((5 * ((torch.cos(theta))**2)) -1)
        Y[2] = (1/4)*torch.sqrt(105/(2*pi)) * torch.exp( (2j) * phi) * ((torch.sin(theta))**2)*torch.cos(theta)
        Y[3] = (-1/8)*torch.sqrt(35/pi) * torch.exp( (3j) * phi) * ((torch.sin(theta))**3)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])
        Y[-2] = ( (-1)**2 ) * torch.conj(Y[2])
        Y[-3] = ( (-1)**3 ) * torch.conj(Y[3])

    elif l == 4:
        Y[0] = (3/16)*torch.sqrt(1/pi)* ( (35 * ((torch.cos(theta))**4) ) - (30 * ((torch.cos(theta))**2) ) + 3)
        Y[1] = (-3/8)*torch.sqrt(5/(pi)) * torch.exp( (1j) * phi) * torch.sin(theta) * ((7 * ((torch.cos(theta))**3) ) - (3*torch.cos(theta)))
        Y[2] = (3/8)*torch.sqrt(5/(2*pi)) * torch.exp( (2j) * phi) * ((torch.sin(theta))**2) *((7 * ((torch.cos(theta))**2) ) - 1)
        Y[3] = (-3/8)*torch.sqrt(35/pi) * torch.exp( (3j) * phi) * ((torch.sin(theta))**3) * torch.cos(theta)
        Y[4] = (3/16)*torch.sqrt(35/(2*pi)) * torch.exp( (4j) * phi) * ((torch.sin(theta))**4)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])
        Y[-2] = ( (-1)**2 ) * torch.conj(Y[2])
        Y[-3] = ( (-1)**3 ) * torch.conj(Y[3])
        Y[-4] = ( (-1)**4 ) * torch.conj(Y[4])

    return Y[m]

class RadialBasis:
    """
    Class to calculate radial basis functions for a specific n, l, etc.

    Attributes:
	x (torch tensor): atomic positions 
        rc (float): Cutoff distance.
        nradbase (int): Max k index.
        nradmax (int): Max n index in Rnl.
        lmax (int): Max l index in Rnl.
        lmbda (float): Exponential factor in g(x).
    """
    def __init__(self, x, rc, nradbase, nradmax, lmax, lmbda):
	self.x = x
        self.rc = rc
        self.nradbase = nradbase
        self.nradmax = nradmax
        self.lmax = lmax
        self.lmbda = lmbda

    def set_species(self,specs=None):
        self.chemflag = True
	if specs != None:
	    assert len(specs) >= len(self.r_arr), "you must provide a species for each neighbor atom"
	    self.species = specs
	elif specs == None:
	    specs = torch.zeros(self.r_arr.shape[0])
	self.species = specs

    def g(self,n):
        return self.gks[n]

    def r_scale(self):
        r_arr = torch.norm(self.x, dim=1)
        self.r_arr = r_arr
        r_scale = r_arr/self.rc
        self.r_scale = r_scale
        numerator = torch.exp(-self.lmbda *(r_scale -1)) - 1
        denominator = torch.exp(self.lmbda)-1
        rscale = 1 - (2 *(numerator/denominator))
        return rscale
        
    def cut_func(self,func):
        rscale_msk = torch.zeros(array_a.shape[0])
        mask = (self.r_scale > 1)
        cutf = torch.clone(func)
        cutf[mask] = rscale_msk[mask]
	self.cutf = cutf
        return cutf

    def G(self,k):
        r_scale = self.r_scale
	pi_rscale = torch.mul(self.r_scale,pi)
        if k == 0:
            func = torch.ones(self.r_scale.shape[0])

        if k == 1:
            func = torch.add(torch.cos(pi_rscale),1)
	    func = torch.mul(func,0.5)

        if k > 1:
            cheb1 = torch.special.chebyshev_polynomial_t(k-1,self.r_scale) #if this doesnt work, then I need to implement chebychev polynomials of the first kind
            #func = (1/4)*(1 -cheb1)*torch.add(torch.cos(pi_rscale),1)
	    funca = torch.add(torch.cos(pi_rscale),1)
	    funcb = torch.mul( torch.subtract(1,cheb1),0.25)
	    func = torch.mul(funca,funcb)

        return self.cut_func(func)

    def R_nl(self,n,l,crad):
        rnl = torch.zeros(self.r_scale.shape[0])
        gs = { i: None for i in range(self.nradbase +1)[1:] }
        for i in range(self.nradbase +1)[1:]:
            f = self.g(i)
            gs[i] = f
            c_nlk = crad[n-1][l][i-1]
            rnl = np.add(rnl, f*c_nlk)
        return rnl

    def init_R_nl(self,**kwargs):
	r_nls = {n:{l: None for l in range(self.lmax +1)} for n in range(1,self.nradmax+1) }
	crad = torch.zeros((self.nradbase,self.lmax+1,self.nradbase))
	for nind in range(self.nradbase):
	    for lind in range(self.lmax+1):
		crad[nind][lind] = np.array([1. if k ==nind else 0. for k in range(self.nradbase)])
	for n in range(1,self.nradmax+1):
	    for l in range(self.lmax+1):
		r_nls[n][l] = self.R_nl(n,l,crad)
        self.r_nls = r_nls

    def set_basis(self,basis='ChebExpCos'):
	# dummy function until we add in other radial basis options
	self.basis = basis
	self.init_R_nl()

    def set_species(self,specs):
        self.chemflag = True
        if specs.shape[0] != self.r_scale.shape[0]:
            raise IndexError("size mismatch for chemical species")
        assert self.specs.shape[0] == self.r_scale.shape[0], "you must provide a species for each neighbor atom"
        self.species = specs
	return None

class angular_basis:
    def __init__(self,
                  x, # (torch tensor): atomic positions
                  lmax): #int-max l index in Ylm
	self.x = x
	self.r_arr = torch.norm(self.x, dim=1)	
	self.get_spherical_polar()
	self.lmax = lmax

    def get_spherical_polar(self):
	rhats = torch.div(self.x,self.r_arr)
	carts = rhats[:, :2]
	last_rhat = rhats[:, 2]
	cartnorm = torch.norm(carts,dim=1)
	polars = torch.atan2(cartnorm,last_rhat)
	asmuths = torch.atan2(rhats[:,1],rhats[1:,0])
	self.rhats = rhats
	self.polars = polars
	self.asmuths = asmuths

    def Ylm(self):
	lm_matrix = { l: {m:None for m in range(-l,l+1)} for l in range(self.lmax+1)}
	for l in range(self.lmax+1):
	    for m in range(-l,l+1):
		func = spherical_harmonic(l,m,self.polars,self.asmuths)
		lm_matrix[l][m] = func
	self.lm_matrix = lm_matrix

    def ylm(self):
	func = self.lm_matrix[l][m]
	return func

class A_basis:
    def __init__(self,
		radial,
		angular):
	self.rb = radial
	self.ab = angular
	# set prefactor for chebexpcos basis
	self.prefac = 1/torch.sqrt(4*pi)

	def A(self,n,l,m,muj=0):
	    rbi = self.rb.r_nl(n,l)
	    abi = self.ab.ylm(l,m)
	    mudeltai = torch.zeros(rbi.shape[0], dtype= torch.float)
	    mudeltaj = torch.ones(rbi.shape[0], dtype= torch.float)
	    mask = (self.rb.species == muj)
	    mudelta = torch.clone(mudeltai)
	    mudelta[mask] = mudeltaj[mask]
	    phi = self.ab.ylm(l,m) * self.rb.r_nl(n,l) * mudelta * self.prefac

	return torch.sum(phi)
