from __future__ import annotations
import numpy as np
from scipy import sparse
from numpy.polynomial.chebyshev import chebfit, chebval
from .io import require

def normalized_adjacency(n:int, edges_src:np.ndarray, edges_dst:np.ndarray):
    data=np.ones(len(edges_src)*2,float); r=np.r_[edges_src,edges_dst]; c=np.r_[edges_dst,edges_src]
    A=sparse.csr_matrix((data,(r,c)),shape=(n,n)); A.sum_duplicates(); A.data[:]=1.0
    deg=np.asarray(A.sum(axis=1)).ravel(); dinv=np.zeros_like(deg,dtype=float); nz=deg>0; dinv[nz]=1/np.sqrt(deg[nz]); S=sparse.diags(dinv)@A@sparse.diags(dinv)
    return S.tocsr(),deg

def cheb_coeff_inverse_sqrt(rho:float,degree:int=64):
    require(0<=rho<1,'rho must be in [0,1)')
    if rho==0: return np.array([1.0]),0.0
    N=max(512,8*(degree+1)); k=np.arange(N); x=np.cos(np.pi*(k+0.5)/N); y=(1-rho*x)**-0.5
    c=chebfit(x,y,degree)
    xx=np.linspace(-1,1,20001); err=float(np.max(np.abs(chebval(xx,c)-(1-rho*xx)**-0.5)))
    return c,err

def cheb_apply(S, coeff, z):
    # Clenshaw for sum c_k T_k(S) z
    z=np.asarray(z,float)
    if len(coeff)==1:return coeff[0]*z
    b1=np.zeros_like(z); b2=np.zeros_like(z)
    for k in range(len(coeff)-1,0,-1):
        b0=2*(S@b1)-b2+coeff[k]*z; b2,b1=b1,b0
    return (S@b1)-b2+coeff[0]*z

def gmrf_draw(n,src,dst,rho,scale,innovation,degree=64):
    if rho==0:
        return scale*np.asarray(innovation,float),1/(scale*scale),0.0
    S,_=normalized_adjacency(n,np.asarray(src,int),np.asarray(dst,int))
    coeff,err=cheb_coeff_inverse_sqrt(rho,degree)
    require(err<=1e-12,f'Chebyshev inverse-square-root certificate too loose: {err}')
    draw=scale*cheb_apply(S,coeff,np.asarray(innovation,float))
    min_prec=(1-rho)/(scale*scale)
    return draw,float(min_prec),err
