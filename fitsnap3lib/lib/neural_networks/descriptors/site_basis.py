import numpy as np
import torch
import math
import torch.nn.functional as F

pi = 3.1415926535897932384626433832795028841971693993751058209749445923
e = 2.7182818284590452353602874713526624977572470936999595749669676277
def scalar_sqrt(num):
    return num**0.5

def spherical_harmonic(l,m,theta,phi):
    Y = {mi:None for mi in range(-l,l+1)}
    if l == 0:
        Y[0] =  (1/2) * scalar_sqrt(1/pi) * torch.ones(theta.shape[0])

    elif l == 1:
        Y = {mi:None for mi in range(-l,l+1)}
        Y[0] = (1/2) * scalar_sqrt(3/pi) * torch.cos(theta)
        Y[1] = (-1/2) * scalar_sqrt(3/(2*pi)) * torch.sin(theta) * torch.exp( (1j) * phi)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])

    elif l == 2:
        Y[0] = (1/4)*scalar_sqrt(5/pi) * ( (3 * ((torch.cos(theta))**2)  )  - 1  )
        Y[1] = (-1/2)*scalar_sqrt(15/(2*pi)) * torch.exp( (1j) * phi) * torch.sin(theta) * torch.cos(theta)
        Y[2] = (1/4) *scalar_sqrt(15/(2*pi)) * torch.exp( (2j) * phi) * ((torch.sin(theta))**2)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])
        Y[-2] = ( (-1)**2 ) * torch.conj(Y[2])

    elif l == 3:
        Y[0] = (1/4)*scalar_sqrt(7/pi) * ( (5 * ((torch.cos(theta))**3) ) - ( 3* torch.cos(theta)) )
        Y[1] = (-1/8)*scalar_sqrt(21/pi) *  torch.exp( (1j) * phi) * torch.sin(theta) * ((5 * ((torch.cos(theta))**2)) -1)
        Y[2] = (1/4)*scalar_sqrt(105/(2*pi)) * torch.exp( (2j) * phi) * ((torch.sin(theta))**2)*torch.cos(theta)
        Y[3] = (-1/8)*scalar_sqrt(35/pi) * torch.exp( (3j) * phi) * ((torch.sin(theta))**3)
        Y[-1] = ( (-1)**1 ) * torch.conj(Y[1])
        Y[-2] = ( (-1)**2 ) * torch.conj(Y[2])
        Y[-3] = ( (-1)**3 ) * torch.conj(Y[3])

    elif l == 4:
        Y[0] = (3/16)*scalar_sqrt(1/pi)* ( (35 * ((torch.cos(theta))**4) ) - (30 * ((torch.cos(theta))**2) ) + 3)
        Y[1] = (-3/8)*scalar_sqrt(5/(pi)) * torch.exp( (1j) * phi) * torch.sin(theta) * ((7 * ((torch.cos(theta))**3) ) - (3*torch.cos(theta)))
        Y[2] = (3/8)*scalar_sqrt(5/(2*pi)) * torch.exp( (2j) * phi) * ((torch.sin(theta))**2) *((7 * ((torch.cos(theta))**2) ) - 1)
        Y[3] = (-3/8)*scalar_sqrt(35/pi) * torch.exp( (3j) * phi) * ((torch.sin(theta))**3) * torch.cos(theta)
        Y[4] = (3/16)*scalar_sqrt(35/(2*pi)) * torch.exp( (4j) * phi) * ((torch.sin(theta))**4)
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
        self.rij_scale()
        self.init_R_nl()

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

    def rij_scale(self):
        r_arr = torch.norm(self.x, dim=1)
        self.r_arr = r_arr
        r_scale = r_arr/self.rc
        self.r_scale = r_scale
        numerator = torch.exp(-self.lmbda *(r_scale -1)) - 1
        #denominator = torch.exp(self.lmbda)-1
        denominator = (e**self.lmbda)-1
        rscale = 1 - (2 *(numerator/denominator))
        return rscale
        
    def cut_func(self,func):
        rscale_msk = torch.zeros(func.shape[0],dtype=torch.double)
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
            cheb1 = torch.special.chebyshev_polynomial_t(self.r_scale,k-1) #validated against the old scipy version
            funca = torch.add(torch.cos(pi_rscale),1)
            funcb = torch.mul( torch.subtract(1,cheb1),0.25)
            func = torch.mul(funca,funcb)

        return self.cut_func(func)

    def R_nl(self,n,l,crad):
        rnl = torch.zeros(self.r_scale.shape[0])
        gs = { i: None for i in range(self.nradbase +1)[1:] }
        for i in range(self.nradbase +1)[1:]:
            f = self.G(i)
            gs[i] = f
            c_nlk = crad[n-1][l][i-1]
            rnl = torch.add(rnl, f*c_nlk)
        self.gks = gs
        return rnl

    def init_R_nl(self,**kwargs):
        r_nls = {n:{l: None for l in range(self.lmax +1)} for n in range(1,self.nradmax+1) }
        crad = torch.zeros((self.nradbase,self.lmax+1,self.nradbase))
        for nind in range(self.nradbase):
            for lind in range(self.lmax+1):
                crad[nind][lind] = torch.tensor([1. if k ==nind else 0. for k in range(self.nradbase)])
        for n in range(1,self.nradmax+1):
            for l in range(self.lmax+1):
                r_nls[n][l] = self.R_nl(n,l,crad)
        self.r_nls = r_nls

    def r_nl(self,n,l):
        return self.r_nls[n][l]

    def set_basis(self,basis='ChebExpCos'):
        # dummy function until we add in other radial basis options
        self.basis = basis
        self.init_R_nl()

    def set_species(self,specs):
        self.chemflag = True
        if specs.shape[0] != self.r_scale.shape[0]:
            raise IndexError("size mismatch for chemical species")
        assert specs.shape[0] == self.r_scale.shape[0], "you must provide a species for each neighbor atom"
        self.species = specs
        return None

class AngularBasis:
    def __init__(self,
                  x, # (torch tensor): atomic positions
                  lmax): #int-max l index in Ylm
        self.x = x
        #self.r_arr = torch.norm(self.x, dim=1)        
        self.r_arr = torch.norm(self.x, dim=0)
        self.get_spherical_polar()
        self.lmax = lmax
        self.Ylm()

    def get_spherical_polar(self):
        rhats = torch.div(self.x,self.r_arr)
        carts = rhats[:, :2]
        last_rhat = rhats[:, 2]
        cartnorm = torch.norm(carts,dim=1)
        polars = torch.atan2(cartnorm,last_rhat)
        asmuths = torch.atan2(rhats[:,1],rhats[:,0])
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

    def ylm(self,l,m):
        func = self.lm_matrix[l][m]
        return func

class A_basis:
    def __init__(self,
                radial,
                angular):
        self.rb = radial
        self.ab = angular
        # set prefactor for chebexpcos basis
        self.prefac = 1/scalar_sqrt(4*pi)

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

class B_basis:
    def __init__(self,
            muvec,
            nvec,
            lvec,
            abase,
            ccs, # assumes m vectors are dict keys in string format: 'm1,m2,m3,m4', etc
                ):

        self.Abase = abase
        self.ccs = ccs
        self.mus = muvec
        self.ns = nvec 
        self.ls = lvec
        self.Alst = None
        self.B = None
        self.get_As()

    def get_As(self):
        mstrs = self.ccs.keys()
        mvecs = [[int(k) for k in cckey.split(',')] for cckey in mstrs]
        alst = torch.zeros((len(mvecs),self.Abase.rb.r_arr.shape[0])) * 1j
        for lstid,mvec in enumerate(mvecs):
            aprd = torch.ones((self.Abase.rb.r_arr.shape[0])) *1j
            for ind,m in enumerate(mvec):
                ai = self.Abase.A(self.ns[ind],self.ls[ind],m,self.mus[ind])
                aprd *= ai
            alst[lstid] = aprd
        self.Alst = alst

    def get_B(self):
        Bi = torch.zeros((self.Abase.rb.r_arr.shape[0]))*1j
        mstrs = self.ccs.keys()
        for im,mstr in enumerate(mstrs):
            Bi += self.Alst[im] * self.ccs[mstr]

        self.B = Bi
        return Bi
