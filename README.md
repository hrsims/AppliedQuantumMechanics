# AppliedQuantumMechanics
Python scripts from the AQM course at GRS

Transmission_interactive.py
A script to compute the transmission probability as a function of incident electron energy. The height of the barrier (in electron volts) and the length of the barrier (in angstroms) can be modified by the user to see how these parameters affect the behavior of the particle.

ArbStepPotential_interactive.py
A script, using the input file input, that computes the square modulus of the wave function of an electron in the presence of an arbitrary 1D step potential. The input file contains a list of arbitrary length of the positions at which the potential changes ("z = ..."), the values of the potential at those points (with extra values for before the first point and after the last, so if there are N values in the "z = " line, there will be N+2 in the "V = ..." line), and a starting energy ("E = "). The user can modify the energy of the incident electron using a slider. For simiplicity, it is assumed that the electron is incident from the left.

input
A sample input file for ArbStepPotential_interactive.py
