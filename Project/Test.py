import numpy as np 
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import uncertainties as unc
import uncertainties.unumpy as unumpy  


def theoretical_fit (x, A, B, gamma ):
    #return (B * x) / (np.log(A * x) +  np.log(np.log( 1 + (1 / 1.01))))
    return (B*x) / np.log(A*x/np.log(1+1/(1.01)))

A = 112
B = 2750
gamma = 1
distance = unc.ufloat(0.048,0.001)
systematicErrorpressure = 0.010*0.133322
systematicErrorVoltage = 100
#gas_coefficient=unc.ufloat(0.63,0.02)
gas_coefficient=unc.ufloat(0.94,0.01)

gamma_min = 1
gamma_max = 1000
#BOUNDS
#UNCERTAINTY

fig,ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
plt.xlabel('Pressure * Distance (kPa*cm)')

plt.subplots_adjust(left=0.25, bottom=0.4)
ax = plt.subplot(2,1,1)
plt.title('Breakdown Voltage of Air for multiple pressure at distance : '+str(unumpy.nominal_values(distance)))
plt.ylabel('Breakdown Voltage (V)')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()


x = np.linspace(10**(-3) ,60, 1000000)
line, = plt.plot(x, theoretical_fit (x, A, B, gamma),label='Custom')


P=np.genfromtxt("data.csv", delimiter=",", usecols=(0), skip_header=1)*0.133322
PD=[]
PD=unumpy.uarray(P,systematicErrorpressure)*distance/gas_coefficient
#for i in range(len(P)):
#    if P[i]<10**(-3) : PD.append(unc.ufloat(P[i],1.33*10**(-5))/gas_coefficient*distance)
#    elif P[i]<0.26 : PD.append(unc.ufloat(P[i],P[i]*0.15)/gas_coefficient*distance)
#    elif P[i]<21.3 : PD.append(unc.ufloat(P[i],P[i]*0.5)/gas_coefficient*distance)
#    elif P[i]>21.3 : PD.append(unc.ufloat(P[i],P[i]*0.25)/gas_coefficient*distance)
V=1000*np.genfromtxt("data.csv", delimiter=",", usecols=(1), skip_header=1)
V=unumpy.uarray(V, systematicErrorVoltage)
(markerline, stemlines, baseline) = plt.errorbar(x=unumpy.nominal_values(PD), y=unumpy.nominal_values(V), xerr=unumpy.std_devs(PD), yerr=unumpy.std_devs(V), fmt=".", label='data')
data = markerline

plt.xlim(min(unumpy.nominal_values(PD)), max(unumpy.nominal_values(PD)))
plt.ylim(min(unumpy.nominal_values(V)), max(unumpy.nominal_values(V)))

popt,pcov = optimize.curve_fit(theoretical_fit,xdata=unumpy.nominal_values(PD), ydata=unumpy.nominal_values(V), sigma=unumpy.std_devs(V), p0=[A,B,gamma], bounds=((0,0,gamma_min),(np.inf,np.inf,gamma_max)))
perr = np.sqrt(np.diag(pcov))
fit, = plt.plot(x,theoretical_fit(x,popt[0],popt[1],popt[2]), label='Best Fit')
print(popt)
print(perr)

plt.legend()


residual_ax = plt.subplot(2,1,2)

def utheoretical_fit (x, A, B, gamma ):
    #return (B * x) / (np.log(A * x) +  np.log(np.log( 1 + (1 / gamma))))
    return (B*x) / unumpy.log(A*x/unumpy.log(1+1/(1.01)))

bestFitPoints=utheoretical_fit(unumpy.nominal_values(PD),unc.ufloat(popt[0],perr[0]),unc.ufloat(popt[1],perr[1]),unc.ufloat(popt[2],perr[2]))
(residual, rstemlines, rbaseline) = plt.errorbar(unumpy.nominal_values(PD), unumpy.nominal_values(V-bestFitPoints),yerr= unumpy.std_devs(V-bestFitPoints),fmt='go', alpha=0.5 ,label = "Residuals")


f_degree = len(PD)-3
print('-'*30)
print('Chi Square value:')
print(sum( ( (unumpy.nominal_values(V)-(theoretical_fit(unumpy.nominal_values(PD),*popt))) / unumpy.std_devs(V))**2 ) )
print('Expected value')
print(f_degree)
print('Minimum Chi-Square for a good fit')
print(f_degree - 2* (2*f_degree)**0.5)
print('Maximum Chi-Square for a good fit')
print(f_degree + 2* (2*f_degree)**0.5)
print('Doubtful minimum Chi-Square for a good fit')
print(f_degree - 3* (2*f_degree)**0.5)
print('Doubtful maximum Chi-Square for a good fit')
print(f_degree + 3* (2*f_degree)**0.5)
print('-'*30)


axA = plt.axes([0.25, 0.1, 0.65, 0.03])
axG = plt.axes([0.25, 0.2, 0.65, 0.03])
axB = plt.axes([0.1, 0.25, 0.0225, 0.63])
A_slider = Slider(
    ax=axA,
    label='A',
    valmin=0,
    valmax=50,
    valinit=A,
)
B_slider = Slider(
    ax=axB,
    label='B',
    valmin=0,
    valmax=3000,
    valinit=B,
    orientation="vertical",
)
G_slider = Slider(
    ax=axG,
    label='Gamma',
    valmin=gamma_min,
    valmax=gamma_max,
    valinit=gamma,
    orientation="horizontal",
)


def update(val):
    line.set_ydata(theoretical_fit(x, A_slider.val, B_slider.val,G_slider.val))
    fig.canvas.draw_idle()
A_slider.on_changed(update)
B_slider.on_changed(update)
G_slider.on_changed(update)


resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    A_slider.reset()
    B_slider.reset()
    G_slider.reset()
reset_button.on_clicked(reset)


updateax = plt.axes([0.6, 0.025, 0.1, 0.04])
update_button = Button(updateax, 'Update', hovercolor='0.975')
def update_event(event):
    update()
def update():

    print('update')
    P=np.genfromtxt("data.csv", delimiter=",", usecols=(0), skip_header=1)*0.133322
    PD=[]
    for i in range(len(P)):
        if P[i]<10**(-3) : PD.append(unc.ufloat(P[i],10**(-4))*distance)
        elif P[i]<0.26 : PD.append(unc.ufloat(P[i],P[i]*0.15)*distance)
        elif P[i]<21.3 : PD.append(unc.ufloat(P[i],P[i]*0.5)*distance)
        elif P[i]>21.3 : PD.append(unc.ufloat(P[i],P[i]*0.25)*distance)
    V=np.genfromtxt("data.csv", delimiter=",", usecols=(1), skip_header=1)
    V=unumpy.uarray(V, 10)
    data.set_xdata(unumpy.nominal_values(PD))
    data.set_ydata(unumpy.nominal_values(V))
    #UPDATE ERROR

    ax.set_xlim (min(unumpy.nominal_values(PD)), max(unumpy.nominal_values(PD)))
    ax.set_ylim(min(unumpy.nominal_values(V)), max(unumpy.nominal_values(V)))

    popt,pcov = optimize.curve_fit(theoretical_fit,xdata=unumpy.nominal_values(PD), ydata=unumpy.nominal_values(V), sigma=unumpy.std_devs(V), p0=[A,B,gamma], bounds=((0,0,gamma_min),(np.inf,np.inf,gamma_max)))
    perr = np.sqrt(np.diag(pcov))
    fit.set_ydata(theoretical_fit(x,popt[0],popt[1],popt[2]))
    BestFitText.set_text('Best Fit: A='+str(round(popt[0],2))+' +/- '+str(round(perr[0],2))+', B='+str(round(popt[1],2))+' +/- '+str(round(perr[1],2))+', Gamma='+str(round(popt[2],2))+' +/- '+str(round(perr[2],2)))

    bestFitPoints=utheoretical_fit(unumpy.nominal_values(PD),unc.ufloat(popt[0],perr[0]),unc.ufloat(popt[1],perr[1]),unc.ufloat(popt[2],perr[2]))
    residual.set_xdata(unumpy.nominal_values(PD))
    residual.set_ydata(unumpy.nominal_values(V-bestFitPoints))

    ax.set_title('Breakdown Voltage of Air for multiple pressure at distance : '+str(unumpy.nominal_values(distance)))
    residual_ax.set_xlim(min(unumpy.nominal_values(PD)), max(unumpy.nominal_values(PD)))
    residual_ax.set_ylim(min(unumpy.nominal_values(V-bestFitPoints)), max(unumpy.nominal_values(V-bestFitPoints)))

    fig.canvas.draw_idle()
update_button.on_clicked(update_event)


logax = plt.axes([0.4, 0.025, 0.1, 0.04])
log_button = Button(logax, 'Log', hovercolor='0.975')
def switch_log(event):
    if ax.get_yscale()=='linear':
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif ax.get_yscale()=='log':
        ax.set_xscale('linear')
        ax.set_yscale('linear')
log_button.on_clicked(switch_log)


distanceax = fig.add_axes([0.2, 0.025, 0.1, 0.04])
distanceax.set_navigate(True)
distance_textbox = TextBox(distanceax, "Distance",initial=unc.nominal_value(distance))
remove_error=False
def change_distance(expressure):
    global distance 
    global stemlines
    global baseline
    global remove_error
    distance = unc.ufloat(eval(expressure),0.1)
    if remove_error==False:
        for line in stemlines:
            line.remove()
        for line in baseline:
            line.remove()
        remove_error=True

    print('distance changed')
    update()
distance_textbox.on_submit(change_distance)


BestFitText = plt.text(1.7,6,'Best Fit: A='+str(round(popt[0],2))+' +/- '+str(round(perr[0],2))+', B='+str(round(popt[1],2))+' +/- '+str(round(perr[1],2))+', Gamma='+str(round(popt[2],2))+' +/- '+str(round(perr[2],2)))

    
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
fig.canvas.manager.full_screen_toggle()

plt.show()



plt.figure()
printax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#printax.set_xscale('log')
#printax.set_yscale('log')
plt.errorbar(x=unumpy.nominal_values(PD), y=unumpy.nominal_values(V), xerr=unumpy.std_devs(PD), yerr=unumpy.std_devs(V), fmt=".", label='data')
plt.plot(x,theoretical_fit(x,popt[0],popt[1],popt[2]), label='Best Fit')
plt.ylim(0.5*min(unumpy.nominal_values(V)),1.2*max(unumpy.nominal_values(V)))
plt.xlim(0,1.2*max(unumpy.nominal_values(PD)))
plt.legend()
plt.ylabel('Breakdown Voltage (V)')
plt.title('Logarithmic Paschen curve of Air for distance 0.048cm')
plt.text(0.6,300,"A="+str(round(popt[0]))+"±"+str(round(perr[0]))+"\nB="+str(round(popt[1]))+"±"+str(round(perr[1])))
printax = plt.subplot2grid((3, 1), (2, 0))
plt.errorbar(unumpy.nominal_values(PD), unumpy.nominal_values(V-bestFitPoints),yerr= unumpy.std_devs(V-bestFitPoints),fmt='go', alpha=0.5 ,label = "Residuals")
plt.xlabel('Pressure * distance (kPa*cm)')
#printax.set_xscale('log')
#printax.set_yscale('log')
plt.savefig('data_2022-03-30_Air_normal')





plt.figure()
plt.plot(np.arange(1,4.5,0.001),np.arange(1,4.5,0.001),label='Theory')
Argon_Pirani = unumpy.uarray([4.0,3.4,3.6,3.2,3.0,2.8,2.7,2.6,2.5,2.1,1.8],0.01*np.array([4.0,3.4,3.6,3.2,3.0,2.8,2.7,2.6,2.5,2.1,1.8]))
Nitrogen_Pirani = unumpy.uarray([4.2,4.0,3.6,3.2,3.0,2.8,2.6,2.4,2.2,2.0,1.8],0.01*np.array([4.2,4.0,3.6,3.2,3.0,2.8,2.6,2.4,2.2,2.0,1.8]))
Air_Pirani = unumpy.uarray([4.0,3.9,3.8,3.7,3.6,3.5,3.4,3.2,3.1,3.0,2.7,2.6,2.4,2.2,2.0,1.8,1.7,1.5],0.01*np.array([4.0,3.9,3.8,3.7,3.6,3.5,3.4,3.2,3.1,3.0,2.7,2.6,2.4,2.2,2.0,1.8,1.7,1.5]))
Argon_EQ = unumpy.uarray([2.22,2.06,2.09,1.97,1.91,1.82,1.81,1.75,1.71,1.51,1.315],0.01)
Nitrogen_EQ = unumpy.uarray([4.23,3.95,3.34,2.69,2.54,2.43,2.35,2.22,2.06,1.90,1.71],0.01)
Air_EQ = unumpy.uarray([4.05,3.83,3.75,3.63,3.38,3.29,3.13,2.75,2.62,2.57,2.41,2.31,2.22,2.06,1.93,1.74,1.62,1.35],0.01)

plt.errorbar(unumpy.nominal_values(Argon_Pirani),unumpy.nominal_values(Argon_EQ), xerr=unumpy.std_devs(Argon_Pirani) ,yerr=unumpy.std_devs(Argon_EQ) , fmt='g.', label='Argon')
plt.errorbar(unumpy.nominal_values(Nitrogen_Pirani),unumpy.nominal_values(Nitrogen_EQ), xerr=unumpy.std_devs(Nitrogen_Pirani) ,yerr=unumpy.std_devs(Nitrogen_EQ) ,fmt='r.', label='Nitrogen')
plt.errorbar(unumpy.nominal_values(Air_Pirani),unumpy.nominal_values(Air_EQ) , xerr=unumpy.std_devs(Air_Pirani) ,yerr=unumpy.std_devs(Air_EQ) , fmt='b.',  label='Air')

def linear(x,m):
    return m*x
Argon_popt,Argon_pcov = optimize.curve_fit(linear,xdata=unumpy.nominal_values(Argon_Pirani), ydata=unumpy.nominal_values(Argon_EQ), p0=[1])
plt.plot(np.arange(1,4.5,0.001),linear(np.arange(1,4.5,0.001),Argon_popt), 'g-',label='Argon Fit')
Argon_perr=np.sqrt(np.diag(Argon_pcov))
Nitrogen_popt,Nitrogen_pcov = optimize.curve_fit(linear,xdata=unumpy.nominal_values(Nitrogen_Pirani), ydata=unumpy.nominal_values(Nitrogen_EQ), p0=[1])
plt.plot(np.arange(1,4.5,0.001),linear(np.arange(1,4.5,0.001),Nitrogen_popt), 'r-',label='Nitrogen Fit')
Nitrogen_perr=np.sqrt(np.diag(Nitrogen_pcov))
Air_popt,Air_pcov = optimize.curve_fit(linear,xdata=unumpy.nominal_values(Air_Pirani), ydata=unumpy.nominal_values(Air_EQ), p0=[1])
plt.plot(np.arange(1,4.5,0.001),linear(np.arange(1,4.5,0.001),Air_popt), 'b-',label='Air Fit')
Air_perr=np.sqrt(np.diag(Air_pcov))

plt.xlabel('Pressure from Pirani (mbar)')
plt.ylabel('Pressure from EQ0404 (mbar)')
plt.title('Coefficients for the thermal conductivity of the different gases')

print(str(Argon_popt)+str(Argon_perr)+'\n'+str(Nitrogen_popt)+str(Nitrogen_perr)+'\n'+str(Air_popt)+str(Air_perr))

plt.text(3,1.1,"Argon:    "+str(round(Argon_popt[0],2))+"±"+str(round(Argon_perr[0],2))+"\nNitrogen: "+str(round(Nitrogen_popt[0],2))+"±"+str(round(Nitrogen_perr[0],2))+"\nAir:      "+str(round(Air_popt[0],2))+"±"+str(round(Air_perr[0],2)))

plt.legend()
plt.savefig('Figures\Gas_Pressure_Coefficient')