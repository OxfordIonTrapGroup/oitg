"""
Units and constants commonly used in physics, with the numeric value being
expressed in the respective SI base units.
"""

import math

#----fundamental constants ----
pi = math.pi
e = 1.602176565e-19;			# 1.602 176 565(35) x 10-19 C : CODATA 2010 : Electron charge
g = 9.80665;                                # m/s^2
kB = 1.3806488e-23;				# 1.380 6488(13) x 10-23 J K-1 : CODATA 2010 : Boltzmann constant
sigma_SB = 5.670373e-8;                     # 5.670 373(21) x 10-8 W m-2 K-4 : CODATA 2010 : Stefan-Boltzmann constant
h = 6.62606957e-34; 			# 6.626 069 57(29) x 10-34 J s : CODATA 2010 : Planck constant
hbar = h/(2*pi);
mu_B = 927.400968e-26; 			# 927.400 968(20) x 10-26 J T-1  : CODATA 2010 : Bohr magneton
mu_N = 5.05078353e-27;			# 5.050 783 53(11) x 10-27 J T-1 : CODATA 2010 : Nuclear magneton
c = 299792458;				# 299 792 458 m s-1 (exact) : CODATA 2010 : Speed of light in vacuum
mu0 = 4*pi*1e-7;				# 4*pi*1e-7 (exact) : Definition of magnetic constant
eps0 = 1/(mu0*c**2);			        # 1/(mu0*c^2) (exact) : Definition of electric constant
alpha = 7.2973525698e-3;			# 7.297 352 5698(24) x 10-3 : CODATA 2010 : Fine structure constant
gS = 2.00231930436153;			# 2.002 319 304 361 53(53) : CODATA 2010 : Electron g factor


#--- Unitless constants ----
NA = 6.02214129e23;               #  6.02214129(27) x 10^23  :  CODATA 2010 : Avogadro constant 


#------- length ----
m = 1;
km = 1e3*m;
cm = 1e-2*m;
mm = 1e-3*m;
um = 1e-6*m;
nm = 1e-9*m;
ang = 1e-10*m;
inch = 25.4*mm;
mil = 1e-3*inch;
ft = 12*inch;
yd = 3*ft;
mi = 5280*ft;
a0 = 5.2917721092e-11*m;		# 5.2917721092(17)×10-11 m : CODATA 2010 : Bohr radius


#------- Volume -------
cc = (cm)**3;
L = 1000*cc;
mL = cc;
floz = 29.5735297*cc;
pint = 473.176475*cc;
quart = 946.35295*cc;
gal = 3.78541197*L;


#----- mass ---------
kg = 1;
gm = 1e-3*kg;
mg = 1e-3*gm;
lb = 0.45359237*kg;
oz = (1/16)*lb;
amu = 1.660538921e-27*kg;		# 1.660 538 921(73) x 10-27 kg : CODATA 2010

#---- time -------
s = 1;
ms = 1e-3*s;
us = 1e-6*s;
ns = 1e-9*s;
ps = 1e-12*s;
fs = 1e-15*s;
minute = 60*s;
hr = 60*minute;
day = 24*hr;
yr = 365.242199*day; 

#---- frequency ---- 
Hz = 1/s;
kHz = 1e3 *Hz;
MHz = 1e6 *Hz;
GHz = 1e9 *Hz;
THz = 1e12*Hz;

#---- force -------
N = 1;
dyne = 1e-5*N;
lbf = 4.44822*N;


#----- energy -----
J = 1;
MJ = 1e6*J;
kJ = 1e3*J;
mJ = 1e-3*J;
uJ = 1e-6*J;
nJ = 1e-9*J;
pJ = 1e-12*J;
eV = e;
BTU = 1.0550559e3*J;
kWh = 3.6e6*J;
cal = 4.1868*J;
kCal = 1e3*cal;

#---- temperature ---
K = 1;
mK = 1e-3*K;
uK = 1e-6*K;
nK = 1e-9*K;

#---- pressure -----
Pa = 1;
torr = 133.322*Pa;
mtorr = 1e-3*torr;
bar = 1e5*Pa;
mbar = 1e-3*bar;
atm = 1.013e5*Pa;
psi = 6.895e3*Pa;


#----- power --- ---
W = 1;
MW = 1e6*W;
kW = 1e3*W;
mW = 1e-3*W;
uW = 1e-6*W;
nW = 1e-9*W;
pW = 1e-12*W;
hp = 745.69987*W;


#------ charge ------
C = 1;


#------ Voltage -----
V = 1;
kV = 1e3*V;
mV = 1e-3*V;
uV = 1e-6*V;
nV = 1e-9*V;


#----- Current ------
A = 1;
mA = 1e-3*A;
uA = 1e-6*A;
nA = 1e-9*A;
pA = 1e-12*A;
fA = 1e-15*A;


#----magnetic field -----
T = 1;
mT = 1e-3*T;
uT = 1e-6*T;
nT = 1e-9*T;
gauss = 1e-4*T;
m_gauss = 1e-3*gauss;
u_gauss = 1e-6*gauss;
