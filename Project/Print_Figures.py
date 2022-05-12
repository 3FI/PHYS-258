import numpy as np 
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import uncertainties as unc
import uncertainties.unumpy as unumpy  


def theoretical_fit (x, A, B, gamma ):
    #return (B * x) / (np.log(A * x) +  np.log(np.log( 1 + (1 / 1.01))))
    #Legacy : USed for determining secondary emission
    return (B*x) / np.log(A*x/np.log(1+1/(1.01)))

#This is the version of the function that is used to get the point on the best fit curve with their uncertainty (It uses the unumpy module)
def utheoretical_fit (x, A, B, gamma ):
    #return (B * x) / (np.log(A * x) +  np.log(np.log( 1 + (1 / gamma)))) #Legacy : USed for determining secondary emission
    return (B*x) / unumpy.log(A*x/unumpy.log(1+1/(1.01)))


A = 112 #Initialize default value
B = 2750 #Initialize default value
gamma = 1.01 #Legacy : Used for determining the secondary emission (Has been fixed to the 1.01 constant in theoretical fit)
gamma_min = 1
gamma_max = 1000

#When the code is in-between to line of # it gotta be modified for the data set
##################################
distance = unc.ufloat(0.048,0.001)
##################################

systematicErrorpressure = 0.010*0.133322
systematicErrorVoltage = 100

##################################
gas_coefficient=unc.ufloat(0.94,0.01) #Air
#gas_coefficient=unc.ufloat(0.63,0.02) #Argon
#gas_coefficient=unc.ufloat(0.93,0.02) #Nitrogen
##################################

#Initialize the PD and V (Import the data and seal it with their uncertainty)
x = np.linspace(10**(-3) ,60, 1000000)
P=np.genfromtxt("data.csv", delimiter=",", usecols=(0), skip_header=1)*0.133322
PD=[]
PD=unumpy.uarray(P,systematicErrorpressure)*distance/gas_coefficient
V=1000*np.genfromtxt("data.csv", delimiter=",", usecols=(1), skip_header=1)
V=unumpy.uarray(V, systematicErrorVoltage)

#Does the best fit
popt,pcov = optimize.curve_fit(theoretical_fit,xdata=unumpy.nominal_values(PD), ydata=unumpy.nominal_values(V), sigma=unumpy.std_devs(V), p0=[A,B,gamma], bounds=((0,0,gamma_min),(np.inf,np.inf,gamma_max)))
perr = np.sqrt(np.diag(pcov))

#Calculate the points (with uncertainty) on the best fit curve at every point of PD (They are used to get proper errorbar on the residuals)
bestFitPoints=utheoretical_fit(unumpy.nominal_values(PD),unc.ufloat(popt[0],perr[0]),unc.ufloat(popt[1],perr[1]),unc.ufloat(popt[2],perr[2]))


#Below is the creation of the two plots

##################################
x_text_position_normal = 0.6
x_text_position_log = 0.1
##################################


plt.figure()
printax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
plt.errorbar(x=unumpy.nominal_values(PD), y=unumpy.nominal_values(V), xerr=unumpy.std_devs(PD), yerr=unumpy.std_devs(V), fmt=".", label='data')
plt.plot(x,theoretical_fit(x,popt[0],popt[1],popt[2]), label='Best Fit')
plt.ylim(0.5*min(unumpy.nominal_values(V)),1.2*max(unumpy.nominal_values(V)))
plt.xlim(0,1.2*max(unumpy.nominal_values(PD)))
plt.legend()
plt.ylabel('Breakdown Voltage (V)')

##################################
plt.title('Paschen curve of Air for distance 0.048cm')
##################################

plt.text(x_text_position_normal,300,"A="+str(round(popt[0]))+"±"+str(round(perr[0]))+"\nB="+str(round(popt[1]))+"±"+str(round(perr[1])))
printax = plt.subplot2grid((3, 1), (2, 0))
plt.errorbar(unumpy.nominal_values(PD), unumpy.nominal_values(V-bestFitPoints),yerr= unumpy.std_devs(V-bestFitPoints),fmt='go', alpha=0.5 ,label = "Residuals")
plt.xlabel('Pressure * distance (kPa*cm)')

##################################
plt.savefig('data_2022-03-30_Air_normal')
##################################






plt.figure()
printax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
printax.set_xscale('log')
printax.set_yscale('log')
plt.errorbar(x=unumpy.nominal_values(PD), y=unumpy.nominal_values(V), xerr=unumpy.std_devs(PD), yerr=unumpy.std_devs(V), fmt=".", label='data')
plt.plot(x,theoretical_fit(x,popt[0],popt[1],popt[2]), label='Best Fit')
plt.ylim(0.5*min(unumpy.nominal_values(V)),1.2*max(unumpy.nominal_values(V)))
plt.xlim(10**-4,1.2*max(unumpy.nominal_values(PD)))
plt.legend()
plt.ylabel('Breakdown Voltage (V)')

##################################
plt.title('Logarithmic Paschen curve of Air for distance 0.048cm')
##################################

plt.text(x_text_position_log,300,"A="+str(round(popt[0]))+"±"+str(round(perr[0]))+"\nB="+str(round(popt[1]))+"±"+str(round(perr[1])))
printax = plt.subplot2grid((3, 1), (2, 0))
plt.errorbar(unumpy.nominal_values(PD), unumpy.nominal_values(V-bestFitPoints),yerr= unumpy.std_devs(V-bestFitPoints),fmt='go', alpha=0.5 ,label = "Residuals")
plt.xlabel('Pressure * distance (kPa*cm)')
printax.set_xscale('log')
printax.set_yscale('log')

##################################
plt.savefig('data_2022-03-30_Air_log')
##################################