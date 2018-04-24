#!/usr/bin/env python
import numpy as np
import cmath as cm # for sqrt of negative numbers
import matplotlib.pyplot as plt
import scipy.integrate as si
from matplotlib.widgets import Slider
# We use the sys module to read command-line arguments
# If you want to impress people, use getopt, and you'll
# be able to use those fancy option-value pair arguments
# (as in -e 4 or --do-plots)
import sys
import scipy.constants as sc

# In case I want to change my units
Ang = 1e-10
eV = sc.e
hbar_ = 1e20*Ang**2*sc.hbar

# Parse an input file and assign values to input parameters
# I do only the bare minimum (or less) of error checking.
# Note the text between the triple quotes. It is called a docstring and
# can be used to automatically generate documentation for a code.
# In principle, it should be much more detailed that this (e.g. it
# might contain a list of all input and output variables).
def read_input():
  """Read input parameters from file 'input'"""
  if len(sys.argv)!=2:
    print("Usage: prob3.py input-file-name")
    exit()
  # defaults
  z = np.array([0,1.e-10]) # positions at which potential changes
  V = np.array([0.,0.,0.]) # V_0,V_1, V_2,...,V_N,V_N+1
  E = 0.0 # energy of state
  # It is always best to enclose any file I/O in a with statement
  # That way, if the program exits prematurely, the file will still
  # be properly closed.
  with open(sys.argv[1],'r') as f:
    # I think that reading input files is my favorite thing to do in Python
    for line in f:
      tag = line.split('=')[0].strip()
      value = line.split('=')[1]
      if tag=='z':
        # values should be separated by commas. one can choose anything,
	# but whitespace characters are not the best choice
        temp = value.split(',')
        N = len(temp)
        z = np.zeros((N),dtype=np.float)
        for i in range(N):
          z[i] = np.float(temp[i])*Ang
      elif tag=='V':
        temp = value.split(',')
        N = len(temp)
        V = np.zeros((N),dtype=np.float)
        for i in range(N):
          V[i] = np.float(temp[i])*eV
      elif tag=='E':
        E0 = np.float(value.strip())*eV
      # if one wished, one could include a "default" case for unrecognized tags
      # instead, I just ignore them
    f.close()
  # I don't check if the entries are valid (e.g. if they are numbers) in any other way
  if len(z)+1 != len(V):
    print("V needs one more entry than z!")
    exit()
  return z,V,E0

def sysSolve(A,b):
  """Solve a set of linear equations using matrix algebra."""
  Ainv = np.linalg.inv(A)
  ans = np.dot(Ainv,b)
  # The different shapes of the b and coeff arrays give rise to this
  # awkwardness
  return np.array([ans[0,0],ans[1,0]])

# We don't actually need this, but maybe it's
# worth keeping in your back pocket.
def overDetSysSolve(A,b):
  """Solve an overdetermined set of linear equations using the least-squares method."""
  At = np.linalg.transpose(A)
  AtA = np.dot(At,A)
  Atb = np.dot(At,b)
  AtAinv = np.linalg.inv(AtA)
  x_approx = np.dot(AtAinv,Atb)
  err = np.dot(A,x_approx)-b
  return x_approx,err

def plotPot(z,V,ax):
  """Plot the piece-wise constant potential V."""
  zmin = min(z)
  zmax = max(z)
  Vmin = min(V)/eV
  Vmax = max(V)/eV
  Z = np.linspace(zmin-10*Ang,zmax+5*Ang,1001,endpoint=True)
  pot = np.zeros((len(Z)),dtype=np.float)
  for i in range(len(Z)):
    for j in range(len(z)):
      if Z[i]<z[j]:
        pot[i] = V[j]/eV
        break
      else:
        pot[i]=V[len(z)]/eV
  ax.plot(Z/Ang,pot)

def psi(x,coeffs,z,V,E):
  """Returns the piece-wise defined (non-normalized) wave function."""
  N = len(z)
  for i in range(N):
    if x<z[i]:
      ki = cm.sqrt(2.*sc.m_e*(E-V[i]))/hbar_
      wfn = coeffs[i,0]*np.exp(1j*ki*x)+coeffs[i,1]*np.exp(-1j*ki*x)
      break
    else:
      ki = cm.sqrt(2.*sc.m_e*(E-V[N]))/hbar_
      wfn = coeffs[N,0]*np.exp(1j*ki*x)
  return wfn

def psi2(x,coeffs,z,V,E):
  """Returns the piece-wise defined (non-normalized) square modulus of the wave function."""
  N = len(z)
  for i in range(N):
    if x<z[i]:
      ki = cm.sqrt(2.*sc.m_e*(E-V[i]))/hbar_
      wfn = coeffs[i,0]*np.exp(1j*ki*x)+coeffs[i,1]*np.exp(-1j*ki*x)
      break
    else:
      ki = cm.sqrt(2.*sc.m_e*(E-V[N]))/hbar_
      wfn = coeffs[N,0]*np.exp(1j*ki*x)
  return wfn.real**2+wfn.imag**2

z,V,E0 = read_input()
N = len(z)
coeffs = np.zeros((N+1,2),dtype=np.complex)

# The problem is solved as follows:
# We start from the right-most segment with only a right-moving wave
# of the form exp(i k_N+1 x). All prior segments contain left- and right-
# moving waves of the appropriate forms (where k_i = sqrt(2m(E-V_i))/hbar.

# E1 here is just to check that I get the same answer for Prob. 1, in which I use
# it as a unit of energy.
E1 = sc.hbar**2*np.pi**2/(2*sc.m_e*(20e-10)**2)

#E0 = 3.1*eV
coeffs[N,0] = 1.
fig,ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
for i in range(N-1,-1,-1):
  zi = z[i] # position of boundary
  Vi = V[i+1] # coming from the right
  Vim1 = V[i]
  ki = cm.sqrt(2*sc.m_e*(E0-Vi))/hbar_ # k_i+1
  kim1 = cm.sqrt(2*sc.m_e*(E0-Vim1))/hbar_ # k_i
  # now setting up our linear equations
  A = np.array([[np.exp(1j*kim1*zi),np.exp(-1j*kim1*zi)],\
    [1j*kim1*np.exp(1j*kim1*zi),-1j*kim1*np.exp(-1j*kim1*zi)]])
  b = np.array([[coeffs[i+1,0]*np.exp(1j*ki*zi)+coeffs[i+1,1]*np.exp(-1j*ki*zi)],\
    [1j*ki*coeffs[i+1,0]*np.exp(1j*ki*zi)-1j*ki*coeffs[i+1,1]*np.exp(-1j*ki*zi)]])
  coeffs[i] = sysSolve(A,b)

Z = np.linspace(min(z)-10*Ang,max(z)+5*Ang,5001,endpoint=True)
psifn = np.zeros((len(Z)),dtype=np.complex)
for i in range(len(Z)):
  psifn[i] = psi(Z[i],coeffs,z,V,E0)
ans,err = si.quad(psi2,min(z)-10*Ang,max(z)+10*Ang,args=(coeffs,z,V,E0))
ax.set_xlim(min(z)/Ang-5,max(z)/Ang+5)
ax.set_ylim(min(V)/eV-0.5,max(V)/eV+5)
plotPot(z,V,ax)
ax.axhline(E0/eV,ls='--',color='r')
ax.plot(Z/Ang,E0/eV+10*(psifn.real**2+psifn.imag**2)/(ans/Ang))

eax = plt.axes([0.25,0.05,0.65,0.03])
#eslide = Slider(eax,'Energy',0.1,50.0,valinit=E0/eV)
eslide = Slider(eax,'Energy',0.1,max(V)/eV+5,valinit=E0/eV)
def update(val):
  E = eslide.val*eV
  coeffs[N,0] = 1.
  for i in range(N-1,-1,-1):
    zi = z[i] # position of boundary
    Vi = V[i+1] # coming from the right
    Vim1 = V[i]
    ki = cm.sqrt(2*sc.m_e*(E-Vi))/hbar_ # k_i+1
    kim1 = cm.sqrt(2*sc.m_e*(E-Vim1))/hbar_ # k_i
    # now setting up our linear equations
    A = np.array([[np.exp(1j*kim1*zi),np.exp(-1j*kim1*zi)],\
      [1j*kim1*np.exp(1j*kim1*zi),-1j*kim1*np.exp(-1j*kim1*zi)]])
    b = np.array([[coeffs[i+1,0]*np.exp(1j*ki*zi)+coeffs[i+1,1]*np.exp(-1j*ki*zi)],\
      [1j*ki*coeffs[i+1,0]*np.exp(1j*ki*zi)-1j*ki*coeffs[i+1,1]*np.exp(-1j*ki*zi)]])
    coeffs[i] = sysSolve(A,b)

  #Z = np.linspace(min(z)-10*Ang,max(z)+5*Ang,5001,endpoint=True)
  #psifn = np.zeros((len(Z)),dtype=np.complex)
  for i in range(len(Z)):
    psifn[i] = psi(Z[i],coeffs,z,V,E)
  ans,err = si.quad(psi2,min(z)-10*Ang,max(z)+10*Ang,args=(coeffs,z,V,E))
  ax.clear()
  ax.set_xlim(min(z)/Ang-5,max(z)/Ang+5)
  ax.set_ylim(min(V)/eV-0.5,max(V)/eV+5)
  plotPot(z,V,ax)
  ax.axhline(E/eV,ls='--',color='r')
  ax.plot(Z/Ang,E/eV+10*(psifn.real**2+psifn.imag**2)/(ans/Ang))
  fig.canvas.draw()
eslide.on_changed(update)

  # In Problem 1, the wave functions are purely real, and we want to plot the actual eigenfunctions
  # not the probability density.
  #plt.plot(Z,E/eV+10*(psifn.real))
plt.show()
