import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import scipy.constants as sc
from matplotlib.widgets import Slider,Button

eV = sc.e
Ang = 1e-10
hbar = sc.hbar*1e20*Ang**2
L_0 = 5.0*Ang
V0_0 = 2.*eV
newL = L_0
newV = V0_0

def trans(E,p_height,p_width):
  k0 = cm.sqrt(2*sc.m_e*E)/hbar
  k1 = cm.sqrt(2*sc.m_e*(E-p_height))/hbar
  num = 4.*k0*k1*cm.exp(-1j*(k0-k1)*p_width)
  denom = (k0+k1)**2-(k0-k1)**2*cm.exp(2j*k1*p_width)
  # Floating point numbers are represented in binary, and so
  # the value stored is not generally exactly what is input.
  # As a result, direct comparison of floats will generally not
  # give the expected result.
  # Also, this is the expression for the special case when E=V0,
  # yielding a linear wave function within the barrier.
  if (abs((E-p_height)/eV)<1e-8):
    t = 2j*k0/((k0**2*p_width+2*1j*k0)*cm.exp(1j*k0*p_width))
  else:
    t = num/denom
  Tprob = np.absolute(t)**2
  return t,Tprob

def refl(E,p_height,p_width):
  k0 = cm.sqrt(2*sc.m_e*E)/hbar
  k1 = cm.sqrt(2*sc.m_e*(E-p_height))/hbar
  num = (k0**2-k1**2)*cm.sin(k1*p_width)
  denom = 2j*k0*k1*cm.cos(k1*p_width)+(k0**2+k1**2)*cm.sin(k1*p_width)
  # I got lazy and didn't include a check for E-V0=0 here as it
  # doesn't come up in the problem.
  r = num/denom
  Rprob = np.absolute(r)**2
  return r,Rprob

fig,ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
E = np.linspace(0.,6.*eV,1001,endpoint=True)
TP = np.zeros((E.shape[0]),np.float)
for i in range(len(E)):
  t,TP[i] = trans(E[i],V0_0,L_0)
ax.set_xlim(0,6)
p, = ax.plot(E/eV,TP)
ax.set_xlabel(r'E (eV)')
ax.set_ylabel(r'Transmission probability')
#plt.plot(E,1-TP)

#lengthax = plt.axes([0.25,0.05,0.65,0.03],axisbg='#e0f9ff')
lengthax = plt.axes([0.25,0.05,0.65,0.03],fc='#e0f9ff')
lengthslide = Slider(lengthax,r'Length of Barrier ($\AA$)',1.0,50.0,valinit=L_0/Ang)
#potax = plt.axes([0.25,0.105,0.65,0.03],axisbg='#e0f9ff')
potax = plt.axes([0.25,0.105,0.65,0.03],fc='#e0f9ff')
potslide = Slider(potax,r'Height of barrier (eV)',0.1,10.0,valinit=V0_0/eV)

def update(val):
  global newL, newV
  newL = lengthslide.val*Ang
  newV = potslide.val*eV
  ax.clear()
  E = np.linspace(0.,newV+5*eV,1001,endpoint=True)
  ax.set_xlim(0,max(E)/eV)
  for i in range(len(E)):
    t,TP[i] = trans(E[i],newV,newL)
  p, = ax.plot(E/eV,TP)
  fig.canvas.draw()
lengthslide.on_changed(update)
potslide.on_changed(update)

plt.show()
