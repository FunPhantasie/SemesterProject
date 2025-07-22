import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# set OSIRIS= True for OSIRIS simulations
# set OSIRIS= False for iPic3D simulations
OSIRIS= True

# folder with the results
# You must fill in here the directory where your simulations are stored 
resFolder=""
# e.g. resFolder="/home/fs/users/<yourusername>/"
# simulation name
# You must fill in here the directory (inside resFolder) of the simulation that you are interested in 
simName=""
# The script will also save figures created by this script in that directory

#The following flags produces 3 plots of the energy evolution
#of the following channels
#ion energy, electron energy, electric field energy, magnetic field energy

#1) plots the total energy of each channel on a linear scale
#The next two plots show the change in energy (from the value at t=0)
#2) do the plot on a log scale
#3) do the plot on a linear scale

plotfullenergy = True 
doln = True 
dolinear = True 

#plots the best fit of the growth rate and numerical heating rate as well as the theoretical growth rate
plotfit = True 

#default time range to calculate the slopes for the growh rate (and heating)
if OSIRIS:
    mint=500
    maxt=1500
else:
    mint=200
    maxt=1800
#You will likely need to change this
#mint=
#maxt=

gamma_th_Omega_ci= 0.78



""" These are iPic3D specific variables
    you will need to set this only if OSIRIS= False"""
#how often the conserved quantities are saved
ConsQuantCycSave=10    
""" END: These are iPic3D specific variables"""

saveDir=resFolder + simName

def extractValue(text, line):
    line = re.sub('[\n]', '', line)
    lineSp=re.compile(' +').split(line)
    
    
    if text =='B0x' or text== 'dt':
        n=2
    elif text =='qom':
        n=3
    
    ext= float(lineSp[n])
   
    if text =='qom':
        return 1./ext
    else:
        return ext


if OSIRIS:

    print("I am looking at OSIRIS simulations")
    #check if there is a file in the directory called *.1d
    #and use first one for inputfile
    prog=re.compile(r'.*\.1d')
    localfiles=os.listdir(saveDir)
    if any((match := prog.match(item)) for item in localfiles):
        inputfile=saveDir+"/"+match.group(0)
    B0=readvar('init_b0',inputfile)
    mi=readvar('rqm',inputfile,1)
    dt=readvar('dt',inputfile)
    maxlin=int(maxt/dt)
    minlin=int(mint/dt)
    
    fld = np.loadtxt(saveDir+'/HIST/fld_ene',skiprows=2)
    par1 = np.loadtxt(saveDir+'/HIST/par01_ene',skiprows=2)
    par2 = np.loadtxt(saveDir+'/HIST/par02_ene',skiprows=2)
    time=fld[:,1]
    p1=par1[:,3]
    p2=par2[:,3]
    if  np.shape(p1)[0] > np.shape(time)[0] or np.shape(p2)[0] > np.shape(time)[0] :
        p1 = p1[0:np.shape(time)[0]]
        p2 = p2[0:np.shape(time)[0]]
    if  np.shape(p1)[0] < np.shape(time)[0]:
        print('shocking! the fields are winning')
    b1=fld[:,2]
    b2=fld[:,3]
    b3=fld[:,4]
    e1=fld[:,5]
    e2=fld[:,6]
    e3=fld[:,7]
    btot=b1+b2+b3
    etot=e1+e2+e3
    emf=btot+etot
    ptot=p1+p2
    tote=emf+ptot
    engperc= (tote-tote[0])*100/tote[0]
    
else:
    
    print("I am looking at iPic3D simulations")
    
    #read variables from inpufile
    inputfile= resFolder + simName + '/' + simName + '.inp'
    inputFound= os.path.isfile(inputfile)
    
    if not inputFound:
        #print(inputfile)
        print('I was looking for the input file')
        print(inputfile)
        print('I could not find it, so I quit')
        sys.exit()

    file_read = open(inputfile, "r")
    lines = file_read.readlines()
    
    textB0x= 'B0x'
    textmi= 'qom'
    textdt= 'dt'
    
    for line in lines:
        if textB0x in line:
            B0 =extractValue(textB0x, line)
            print("B0= " + repr(B0))
        
        if textmi in line:
            mi= extractValue(textmi, line)
            print("mi= " + repr(mi))
            
        if textdt in line:
            dt= extractValue(textdt, line)
            print("dt= " + repr(dt)) 
            
    #read energies from ConservedQuantities.txt    
    consQuantitiesFile= resFolder + simName + '/ConservedQuantities.txt'
    consFound= os.path.isfile(consQuantitiesFile)
    
    if not consFound:
        #print(inputfile)
        print('I was looking for the conserved quantity file')
        print(consQuantitiesFile)
        print('I could not find it, so I quit')
        sys.exit()
    
    f=open(consQuantitiesFile,"r")
    lines=f.readlines()
    etot_l=[];btot_l=[]; cyc_l=[]; p1_l=[]; p2_l=[]
    for x in lines:
        x = re.sub('[\n]', '', x)
        #print(re.compile('\t').split(x))
        cyc_l.append(float(x.split('\t')[0]))
        #the +1 to account for the space
        etot_l.append(float(x.split('\t')[3+1]))
        btot_l.append(float(x.split('\t')[4+1]))
        p1_l.append(float(x.split('\t')[6+1]))
        p2_l.append(float(x.split('\t')[7+1]))
        f.close()

    #list to array
    cyc = np.array(cyc_l); time=cyc*dt
    etot = np.array(etot_l)
    btot = np.array(btot_l)
    p1 = np.array(p1_l)
    p2 = np.array(p2_l)

    emf=btot+etot
    ptot=p1+p2
    tote=emf+ptot
    engperc= (tote-tote[0])*100/tote[0]
    
    maxlin=int(maxt/dt/ConsQuantCycSave)
    minlin=int(mint/dt/ConsQuantCycSave)
    
gamma_th= gamma_th_Omega_ci/mi*B0

if plotfullenergy:
    plt.figure()
    plt.plot(time,btot/tote[0], label='B field')
    plt.plot(time,p1/tote[0], label='Electrons')
    plt.plot(time,p2/tote[0], label='Ions')
    plt.plot(time,etot/tote[0], label='E field')
    plt.plot(time,tote/tote[0], label= 'Total')
    plt.xlabel("$t \omega_{pe}$")
    plt.ylabel("$E/ E_{tot,0}$")
    plt.legend(loc="center right")

    plt.savefig(saveDir + '/Enorm.png')

if doln:

    plt.figure()
    plt.xlabel("$t \omega_{pe}$")
    plt.ylabel("Change in energy ($|\Delta E|/ E_{tot,0}$)")
    plt.yscale('log')
    plt.plot(time,(abs(etot-etot[0]))/tote[0],label='E field')
    plt.plot(time,(abs(btot-btot[0]))/tote[0],label='B field')
    plt.plot(time,(abs(p1-p1[0]))/tote[0],label='Electrons')
    plt.plot(time,(abs(p2-p2[0]))/tote[0],label='Ions')
    plt.axvline(x=time[minlin])
    plt.axvline(x=time[maxlin])
    
    if plotfit:
           
        myfit = np.polyfit(time[minlin:maxlin],np.log((btot[minlin:maxlin] -btot[0])/tote[0])/2,1)
        plt.plot(time,np.exp(2*(myfit[0]*time+myfit[1])), label='fit, $\gamma/ \Omega_{ci}$=' + repr(round(myfit[0]*mi/B0,2)))
        plt.plot(time,np.exp(2*(gamma_th*time+myfit[1])), label='th,  $\gamma/ \Omega_{ci}$=' + repr(round(gamma_th_Omega_ci,2)))
        plt.plot(time,np.exp(2*(myfit[0]*time+myfit[1])))
        
        print("the growth rate is " +str(myfit[0]) + " omega_pe")
        print("the growth rate is " +str(myfit[0]*mi/B0) + " Omega_ci")
        print("the theoretical growth rate is " +str(gamma_th) + " omega_pe")
        print("the theoretical growth rate is " +str(gamma_th_Omega_ci) + " Omega_ci")

    plt.legend(loc="center right")
    plt.savefig(saveDir + '/Echange.png')

if dolinear:

    plt.figure()
    plt.xlabel("$t \omega_{pe}$")
    plt.ylabel("Change in energy ($\Delta E/ E_{tot,0}$)")
    plt.plot(time,(etot-etot[0])/tote[0],label='E field')
    plt.plot(time,(btot-btot[0])/tote[0],label='B field')
    plt.plot(time,(p1-p1[0])/tote[0],label='Electrons')
    plt.plot(time,(p2-p2[0])/tote[0],label='Ions')
    plt.plot(time,(tote-tote[0])/tote[0],label='Total')
    if plotfit == True:
        myfit = np.polyfit(time[minlin:maxlin],(tote[minlin:maxlin] -tote[0])/tote[0],1)
        plt.plot(time,myfit[0]*time+myfit[1], label='fit num heating '+  repr(round(myfit[0],7)))
        print("the numerical heating rate is " +str(myfit[0]) +" [E_tot omega_pe]")
    plt.legend(loc="best")
    plt.savefig(saveDir + '/EnumHeat.png')

plt.show()
