import os
from numpy import *
from pylab import *
import __init__ as snana
from mpfit import mpfit
import classify
from scipy import integrate
try : 
    SNDATA_ROOT=os.environ['SNDATA_ROOT']
except : 
    SNDATA_ROOT = os.path.abspath('.')
#SALT2 parameters from Kessler et al. 2009
cmean=0.04
csigma=0.13
x1mean=-0.13
x1sigma=1.24

#range of parameters
crange=1
x1range=6
pkmjdrange=50
avrange=7

def chi2fit(datfile,clobber=False,sigma_cc=0.15,sigma_Ia=0.08,Nsim=1000,simrange=35):
    """do chi2 fitting to a supernova light 
    curve using SNANA simulations"""
    sn=snana.SuperNova(SNDATA_ROOT+'/SIM/'+datfile)
    sn.getClassSim(Nsim=Nsim*2.,clobber=clobber)

    #no information from observations far from peak 
    gr=where(abs(sn.MJD-sn.SEARCH_PEAKMJD) < simrange )[0]

    #Type Ia chi-squared
    mIa = zeros([len(sn.ClassSim.Ia.PTROBS_MIN),len(gr)])
    for MJD,FLT,i in zip(sn.MJD[gr],sn.FLT[gr],range(len(sn.MJD[gr]))):
        sn.ClassSim.Ia.samplephot(MJD)
        mIa[:,i] = sn.ClassSim.Ia.__dict__['%s%i'%(FLT, int(MJD))]
#    sn.ClassSim.Ia.getClassCurves()
#    sn.ClassSim.Ibc.getClassCurves()
#    sn.ClassSim.II.getClassCurves()

    chisqIa = []
    chidenomIa = []
    Iaparams = []
    modelc,modelx1=[],[]
    count=0
    for i,x1,color in zip(range(len(sn.ClassSim.Ia.PTROBS_MIN)),sn.ClassSim.Ia.SIM_SALT2x1,sn.ClassSim.Ia.SIM_SALT2c):
        if not len(where(mIa[i,:]==99)[0]) and count < Nsim:
            #fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mIa[i,:]}#[magIa[0][i],mag}
            #mvals=mpfit.mpfit(lsfunc,[1.],functkw=fa)
            param=chisqmin(10**(-0.4*(mIa[i,:]-27.5)),sn.FLUXCAL[gr],0*10**(-0.4*(mIa[i,:]-27.5)),sn.FLUXCALERR[gr])
            Iaparams = Iaparams + [param]
            model = array(param)*10**(-0.4*(mIa[i,:]-27.5))
            modelc+=[color]
            modelx1+=[x1]
            chisqIa = chisqIa + [sum((model - sn.FLUXCAL[gr])**2/(sn.FLUXCALERR[gr]**2+(sigma_Ia*model)**2.))]
            chidenomIa += [product(sqrt(2*pi)*sqrt(sn.FLUXCALERR[gr]**2+(sigma_Ia*model)**2.))]
            count+=1

    #Type Ib/c chi-squared
    mIbc = zeros([len(sn.ClassSim.Ibc.PTROBS_MIN),len(sn.MJD[gr])])
    for MJD,FLT,i in zip(sn.MJD[gr],sn.FLT[gr],range(len(sn.MJD[gr]))):
        sn.ClassSim.Ibc.samplephot(MJD)
        mIbc[:,i] = sn.ClassSim.Ibc.__dict__['%s%i'%(FLT, int(MJD))]

    chisqIbc = []
    chidenomIbc = []
    Ibcparams = []
    modelav_Ibc = []
    count=0
    for i,av in zip(range(len(sn.ClassSim.Ibc.PTROBS_MIN)),sn.ClassSim.Ibc.DUMP['AV']):
        if not len(where(mIbc[i,:]==99)[0]) and count < Nsim:
            count+=1
            #fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mIbc[i,:]}
            #mvals=mpfit.mpfit(lsfunc,[1.],functkw=fa)
            param=chisqmin(10**(-0.4*(mIbc[i,:]-27.5)),sn.FLUXCAL[gr],0*sigma_cc*10**(-0.4*(mIbc[i,:]-27.5)),sn.FLUXCALERR[gr])
            Ibcparams = Ibcparams + [param]
            model = array(param)*10**(-0.4*(mIbc[i,:]-27.5))
            modelav_Ibc+=[av]

            chisqIbc = chisqIbc + [sum((model - sn.FLUXCAL[gr])**2/(sn.FLUXCALERR[gr]**2+(sigma_cc*model)**2.))]
            chidenomIbc += [product(sqrt(2*pi)*sqrt(sn.FLUXCALERR[gr]**2+(sigma_cc*model)**2.))]

    #Type II chi-squared
    mII = zeros([len(sn.ClassSim.II.PTROBS_MIN),len(sn.MJD[gr])])
    for MJD,FLT,i in zip(sn.MJD[gr],sn.FLT[gr],range(len(sn.MJD[gr]))):
        sn.ClassSim.II.samplephot(MJD)
        mII[:,i] = sn.ClassSim.II.__dict__['%s%i'%(FLT, int(MJD))]

    chisqII = []
    chidenomII = []
    IIparams = []
    modelav_II = []
    count=0
    for i,av in zip(range(len(sn.ClassSim.II.PTROBS_MIN)),sn.ClassSim.Ibc.DUMP['AV']):
        if not len(where(mII[i,:]==99)[0]) and count < Nsim:
            count+=1
            #fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mII[i,:]}
            #mvals=mpfit.mpfit(lsfunc,[1.],functkw=fa)
            param=chisqmin(10**(-0.4*(mII[i,:]-27.5)),sn.FLUXCAL[gr],0*sigma_cc*10**(-0.4*(mII[i,:]-27.5)),sn.FLUXCALERR[gr])
            IIparams = IIparams + [param]
            model = array(param)*10**(-0.4*(mII[i,:]-27.5))
            modelav_II+=[av]
            
            chisqII = chisqII + [sum((model - sn.FLUXCAL[gr])**2/(sn.FLUXCALERR[gr]**2+(sigma_cc*model)**2.))]
            chidenomII += [product(sqrt(2*pi)*sqrt(sn.FLUXCALERR[gr]**2+(sigma_cc*model)**2.))]

    print """Type Ia SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqIa),len(chisqIa))
    print """Type Ibc SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqIbc),len(chisqIbc))
    print """Type II SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqII),len(chisqII))

    #the spacing of all parameters
    delIa = pkmjdrange*crange*x1range/(float(len(chisqIa)))#**3.
    delIbc = avrange*pkmjdrange/(float(len(chisqIbc)))#**2.
    delII = avrange*pkmjdrange/(float(len(chisqII)))#**2.

    ratesIa,ratesIbc,ratesII=classify.classfracz(1.914)

    #now compute the likelihoods
    eIa = ratesIa*delIa*exp(-array(chisqIa)/2.)/array(chidenomIa)*gauss(array(modelc),cmean,csigma)*gauss(array(modelx1),x1mean,x1sigma)
    eIbc = ratesIbc*delIbc*exp(-array(chisqIbc)/2.)/array(chidenomIbc)*avfunc(array(modelav_Ibc))
    eII = ratesII*delII*exp(-array(chisqII)/2.)/array(chidenomII)*avfunc(array(modelav_II))

    chiIa = ratesIa*delIa*sum(exp(-array(chisqIa)/2.)/array(chidenomIa)*gauss(array(modelc),cmean,csigma)*gauss(array(modelx1),x1mean,x1sigma))
    chiIbc = ratesIbc*delIbc*sum(exp(-array(chisqIbc)/2.)/array(chidenomIbc)*avfunc(array(modelav_Ibc)))
    chiII = ratesII*delII*sum(exp(-array(chisqII)/2.)/array(chidenomII)*avfunc(array(modelav_II)))

    p_Ia = chiIa/(double(chiIa)+chiIbc+chiII)
    p_Ibc = chiIbc/(double(chiIa)+chiIbc+chiII)
    p_II = chiII/(double(chiIa)+chiIbc+chiII)

    print """likelihoods:
Type Ia   Type Ib/c  Type II
%.8f      %.8f       %.8f"""%(p_Ia,p_Ibc,p_II)

    print """odds ratios:
Type Ia/Ib/c  Type Ia/II
%.4f      %.4f"""%(chiIa/chiIbc,chiIa/chiII)

    Iarow = where(chisqIa == min(chisqIa))[0][0]
    Ibcrow = where(chisqIbc == min(chisqIbc))[0][0]
    IIrow = where(chisqII == min(chisqII))[0][0]

    import pdb; pdb.set_trace()

    return mIa, mIbc, mII, eIa/chiIa, eIbc/chiIbc, eII/chiII, Iarow, Ibcrow, IIrow, Iaparams, Ibcparams, IIparams

def avfunc(av):
    dahlen_av=loadtxt('/Users/David/dahlen_av.dat',usecols=[0],unpack=True)
    n_av=histogram(2.5*dahlen_av,bins=1000,range=[0,7])
    scale=1/integrate.trapz(n_av[0]/100000.,x=n_av[1][:-1])

    func=[]
    for a in av:
        row=where((a-n_av[1])**2.==min((a-n_av[1])**2.))[0]
        if row[0]==1000.: row-=1
        func+=[scale*n_av[0][row][0]/100000.]
        #if scale*n_av[0][row]/100000. < 1: func+=[scale*n_av[0][row][0]/100000.]
        #if scale*n_av[0][row]/100000. > 1: func+=[1.]

    return array(func)

def chisqmin(sim,obs,sigmasim,sigmaobs):
    """takes arrays of observed and model data
    finds the slope to best match them"""

    num=sum(sim*obs/(sigmasim**2.+sigmaobs**2.))
    denom=sum(sim**2./(sigmasim**2.+sigmaobs**2.))
    a=num/denom

    return a

def gauss(x,mean,sigma):
    p=1/sqrt(2*pi*sigma**2.)*exp(-(x-mean)**2./(2*sigma**2.))
    return p

def lsfunc(p,model_mag=None,x=None,y=None,err=None,fjac=None):
    """the function called by mpfit
    allows magnitudes to be scaled 
    linearly to match the data"""
    status=0
    return [status,y-(10**(-0.4*(model_mag-27.5))*p[0])]

def lsfunc_color(p,model_mag=None,x=None,y=None,err=None,fjac=None,flt=None):
    """the function called by mpfit
    allows magnitudes to be scaled 
    linearly to match the data"""
    status=0
    scale=ones(len(flt))
    
    for f,i in zip(unique(flt),range(len(unique(flt)))):
        row=where(flt==f)
        scale[row]=p[i]

    return [status,y-(10**(-0.4*(model_mag-27.5))*scale)]

def timestr( time ) :

    if time>=0 : pm = 'p'
    else : pm = 'm'
    return( '%s%02i'%(pm,abs(time) ) )

def plotbestlc(datfile,clobber=True,paperfig=True,showrange=True,Nsim=2000,cadence=0.5):
    """plot the best-fitting light curves 
    for Type Ia, Ib/c and II SNe
    from SNANA simulations"""

    import rcpar
    close('all')
    rcpar.lcpaperfig()
    import classify

    #set up the axes
    ax3 = axes( [0.12,0.09,0.75,0.27] )
    ax2 = axes( [0.12,0.36,0.75,0.27] )
    ax1 = axes( [0.12,0.63,0.75,0.27] )

    sn=snana.SuperNova(SNDATA_ROOT+'/SIM/'+datfile)

    #find the best-fit light curves
    mjd = 4*range(int(min(sn.MJD)-15),int(max(sn.MJD)+34),1)
    snanafilt = ['H']*int(len(mjd)/4.)+['J']*int(len(mjd)/4.)+['Z']*int(len(mjd)/4.)+['I']*int(len(mjd)/4.)

    pkmjdrange = [55511,55561]
#    pkmjdrange = [sn.mjdpk-sn.mjdpkerr,sn.mjdpk+sn.mjdpkerr]
    zmin = sn.z-sn.zerr
    zmax = sn.z+sn.zerr
    simname = 'HST_classify_%s'%(sn.name)
    simname = classify.doClassSim(
        simroot=simname, zrange=[zmin,zmax],
        Nsim=Nsim*2., bandlist=snanafilt,
        mjdlist = mjd, pkmjdrange=pkmjdrange,
        clobber=clobber, cadence=cadence,flatdist=True )
    sn.getClassSim(Nsim=Nsim,clobber=False)

    #plot the real data
    for filt,color,label in zip(['J','H'][::-1],['g','r'][::-1],['F125W','F160W'][::-1]):
        row=where(sn.FLT == filt)
        ax1.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color)
        ax2.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color,label=label)
        ax3.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color)
    
    mIa, mIbc, mII, chisqIa, chisqIbc, chisqII, Iarow, Ibcrow, IIrow, Iaparams, Ibcparams, IIparams = chi2fit(datfile,Nsim=Nsim)
    Iascale,Ibcscale,IIscale = Iaparams[Iarow],Ibcparams[Ibcrow],IIparams[IIrow]

    #Type Ia - best-fit light curve and 68% of the evidence
    x,indIa=[],[]
    for i,j in zip(sort(chisqIa)[::-1],argsort(chisqIa[::-1])):
        x += [i]
        indIa += [j]
        if x>0.68: continue

    mIabest = zeros([len(indIa),len(mjd)])
    m = zeros([len(sn.ClassSim.Ia.PTROBS_MIN),len(mjd)])
    x1 = zeros(len(sn.ClassSim.Ia.PTROBS_MIN))
    c = zeros(len(sn.ClassSim.Ia.PTROBS_MIN))
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.Ia.samplephot(MJD)
        m[:,i] = sn.ClassSim.Ia.__dict__['%s%i'%(FLT, int(MJD))]
    c = sn.ClassSim.Ia.SIM_SALT2c
    x1 = sn.ClassSim.Ia.SIM_SALT2x1
    count,j=0,0
    for i in range(len(sn.ClassSim.Ia.PTROBS_MIN)):
        if not len(where(mIa[i,:]==99)[0]):
            if count==Iarow: 
                bestlc_Ia = m[i,:]
                bestx1_Ia = x1[i]
                bestc_Ia = c[i]
            if count in indIa: 
                mIabest[j,:] = Iaparams[j]*10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplc,bottomlc=[],[]
    for col in range(len(mjd)):
        toplc+=[max(mIabest[:,col])]
        bottomlc+=[min(mIabest[:,col])]

    print 'x1 and c from best fit'
    print bestx1_Ia,bestc_Ia

    #Type Ibc - best-fit light curve and 68% of the evidence
    x,indIbc=[],[]
    for i,j in zip(sort(chisqIbc)[::-1],argsort(chisqIbc[::-1])):
        x += [i]
        indIbc += [j]
        if x>0.68: continue

    mIbcbest = zeros([len(indIbc),len(mjd)])
    m = zeros([len(sn.ClassSim.Ibc.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.Ibc.samplephot(MJD)
        m[:,i] = sn.ClassSim.Ibc.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.Ibc.PTROBS_MIN)):
        if not len(where(mIbc[i,:]==99)[0]):
            if count==Ibcrow: bestlc_Ibc = m[i,:]
            if count in indIbc: 
                mIbcbest[j,:] = Ibcparams[j]*10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplcIbc,bottomlcIbc=[],[]
    for col in range(len(mjd)):
        toplcIbc+=[max(mIbcbest[:,col])]
        bottomlcIbc+=[min(mIbcbest[:,col])]

    #Type II - best-fit light curve and 68% of the evidence
    x,indII=[],[]
    for i,j in zip(sort(chisqII)[::-1],argsort(chisqII[::-1])):
        x += [i]
        indII += [j]
        if x>0.68: continue

    mIIbest = zeros([len(indII),len(mjd)])
    m = zeros([len(sn.ClassSim.II.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.II.samplephot(MJD)
        m[:,i] = sn.ClassSim.II.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.II.PTROBS_MIN)):
        if not len(where(mII[i,:]==99)[0]):
            if count==IIrow: bestlc_II = m[i,:]
            if count in indII: 
                mIIbest[j,:] = IIparams[j]*10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplcII,bottomlcII=[],[]
    for col in range(len(mjd)):
        toplcII+=[max(mIIbest[:,col])]
        bottomlcII+=[min(mIIbest[:,col])]

    #plot the best-fit light curves
    for filt,color in zip(['J','H'],['g','r']):
        row=where(array(snanafilt) == filt)[0]
        mjd,bestlc_Ia,bestlc_Ibc,bestlc_II=array(mjd),array(bestlc_Ia),array(bestlc_Ibc),array(bestlc_II)
        ax1.plot(mjd[row],Iascale*10**(-0.4*(bestlc_Ia[row]-27.5)),color=color)
        ax2.plot(mjd[row],Ibcscale*10**(-0.4*(bestlc_Ibc[row]-27.5)),color=color)
        ax3.plot(mjd[row],IIscale*10**(-0.4*(bestlc_II[row]-27.5)),color=color)
        if showrange:
            fill_between(ax1,mjd[row],array(bottomlc)[row],array(toplc)[row],color=color,alpha=0.2)
            fill_between(ax2,mjd[row],array(bottomlcIbc)[row],array(toplcIbc)[row],color=color,alpha=0.2)
            fill_between(ax3,mjd[row],array(bottomlcII)[row],array(toplcII)[row],color=color,alpha=0.2)

    #for SN Wilson, the specific commands to make things look good
    if paperfig:
            ax1.set_ylim([-2,30])
            ax2.set_ylim([-2,25])
            ax3.set_ylim([-2,25])
            
            ax1.set_xlim([55500,55630])
            ax2.set_xlim([55500,55630])
            ax3.set_xlim([55500,55630])
            
            ax1.set_xticks([55500,55525,55550,55575,55600,55625])
            ax1.set_xticklabels(['','','','','',''])
            ax2.set_xticks([55500,55525,55550,55575,55600,55625])
            ax2.set_xticklabels(['','','','','',''])
            ax3.set_xticks([55500,55525,55550,55575,55600,55625])
            ax3.set_xticklabels(['55500','55525','55550','55575','55600','55625'])

            ax1.set_yticks([0,5,10,15,20,25,30])
            ax2.set_yticks([0,5,10,15,20,25])
            ax3.set_yticks([0,5,10,15,20,25])

            ax1.set_yticklabels(['','5','10','15','20','25',''])
            ax2.set_yticklabels(['','5','10','15','20',''])
            ax3.set_yticklabels(['','5','10','15','20',''])

            ax1.yaxis.set_ticks_position('both')
            ax2.yaxis.set_ticks_position('both')
            ax3.yaxis.set_ticks_position('both')
            
            ax1.text(55590,17,r"""Type Ia
Best $\chi^2/\nu$ = 18.8/11""",color='r',fontsize='medium')
            ax2.text(55590,15,r"""Type Ib/c
Best $\chi^2/\nu$ = 41.9/11""",color='g',fontsize='medium')
            ax3.text(55590,15,r"""Type II
Best $\chi^2/\nu$ = 47.9/11""",color='b',fontsize='medium')

            ax3.set_xlabel('MJD',fontsize='medium')
            ax2.set_ylabel('Flux',fontsize='medium')

            ax1.xaxis.set_ticks_position('top')

            ax2.legend(loc='lower right',prop={'size':10},numpoints=1)

            ax3top = ax1.twiny()
            ax3top.set_xlabel('Time relative to best fit Ia peak (rest frame)',fontsize='medium')
            ax3top.set_xlim( (array(ax1.get_xlim()) - 55539.)/2.914 )
            ax3top.set_xticks([-10,0,10,20,30])
            ax1.xaxis.set_ticklabels(['','','','',''])

            ax3.set_xticks([55500,55525,55550,55575,55600,55625])
            ax3.set_xticklabels(['55500','55525','55550','55575','55600','55625'])

    return

def chisqhist(datfile):

    ax = axes( [0.12,0.12,0.85,0.85] )

    mIa, mIbc, mII, chisqIa, chisqIbc, chisqII, Iarow, Ibcrow, IIrow, Iascale, Ibcscale, IIscale = chi2fit(datfile)

    histIa = hist(chisqIa,bins=20)
    histIbc = hist(chisqIbc,bins=20)
    histII = hist(chisqII,bins=20)

    ax.plot(histIa,color='r',label='Type Ia')
    ax.plot(histIbc,color='g',label='Type Ibc')
    ax.plot(histII,color='b',label='Type II')

    ax.set_xlabel('$\chi^2$')
    ax.set_xlabel('Number')

    return

def fill_between(ax, x, y1, y2, **kwargs):

   # add x,y2 in reverse order for proper polygon filling                                                                                              
   from matplotlib.patches import Polygon
   verts = zip(x,y1) + [(x[i], y2[i]) for i in range(len(x)-1,-1,-1)]
   poly = Polygon(verts, **kwargs)
   ax.add_patch(poly)
   ax.autoscale_view()
   return poly

def chi2magfit(datfile,sigma=0.0,clobber=False,Nsim=2000,ret=True):
    """do chi2 fitting to a supernova light 
    curve using SNANA simulations using a magnitude 
    prior and a 10% model uncertainty"""
    sn=snana.SuperNova(SNDATA_ROOT+'/SIM/'+datfile)
    sn.getClassSim(Nsim=Nsim,clobber=clobber)

    #Type Ia chi-squared
    mIa = zeros([len(sn.ClassSim.Ia.PTROBS_MIN),len(sn.MJD)])
    for MJD,FLT,i in zip(sn.MJD,sn.FLT,range(len(sn.MJD))):
        sn.ClassSim.Ia.samplephot(MJD)
        mIa[:,i] = sn.ClassSim.Ia.__dict__['%s%i'%(FLT, int(MJD))]

    chisqIa = []
    Iaparams = []
    normIa = []
    for i in range(len(sn.ClassSim.Ia.PTROBS_MIN)):
        if not len(where(mIa[i,:]==99)[0]):
            #fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mIa[i,:]}
            #mvals=mpfit.mpfit(lsfunc,[1.],functkw=fa)
            #Iaparams = Iaparams + [mvals.params[0]]
            model = 10**(-0.4*(mIa[i,:]-27.5))
            normIa = normIa + [product((2*pi*(sn.FLUXCALERR**2+(sigma*model)**2.))**(-1/2.))]
            chisqIa = chisqIa + [sum((model - sn.FLUXCAL)**2/(sn.FLUXCALERR**2+(sigma*model)**2.))]

    #Type Ib/c chi-squared
    mIbc = zeros([len(sn.ClassSim.Ibc.PTROBS_MIN),len(sn.MJD)])
    for MJD,FLT,i in zip(sn.MJD,sn.FLT,range(len(sn.MJD))):
        sn.ClassSim.Ibc.samplephot(MJD)
        mIbc[:,i] = sn.ClassSim.Ibc.__dict__['%s%i'%(FLT, int(MJD))]

    chisqIbc = []
    Ibcparams = []
    normIbc = []
    #count=0
    for i in range(len(sn.ClassSim.Ibc.PTROBS_MIN)):
        if not len(where(mIbc[i,:]==99)[0]):
            #count=count+1
            #fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mIbc[i,:]}
            #mvals=mpfit.mpfit(lsfunc,[1.],functkw=fa)
            #Ibcparams = Ibcparams + [mvals.params[0]]
            model = 10**(-0.4*(mIbc[i,:]-27.5))
            normIbc = normIbc + [product((2*pi*(sn.FLUXCALERR**2+(sigma*model)**2.))**(-1/2.))]
            chisqIbc = chisqIbc + [sum((model - sn.FLUXCAL)**2/(sn.FLUXCALERR**2+(sigma*model)**2.))]

    #Type II chi-squared
    mII = zeros([len(sn.ClassSim.II.PTROBS_MIN),len(sn.MJD)])
    for MJD,FLT,i in zip(sn.MJD,sn.FLT,range(len(sn.MJD))):
        sn.ClassSim.II.samplephot(MJD)
        mII[:,i] = sn.ClassSim.II.__dict__['%s%i'%(FLT, int(MJD))]

    chisqII = []
    IIparams = []
    normII = []
    for i in range(len(sn.ClassSim.II.PTROBS_MIN)):
        if not len(where(mII[i,:]==99)[0]):
            #fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mII[i,:]}
            #mvals=mpfit.mpfit(lsfunc,[1.],functkw=fa)
            #IIparams = IIparams + [mvals.params[0]]
            model = 10**(-0.4*(mII[i,:]-27.5))
            normII = normII + [product((2*pi*(sn.FLUXCALERR**2+(sigma*model)**2.))**(-1/2.))]
            chisqII = chisqII + [sum((model - sn.FLUXCAL)**2/(sn.FLUXCALERR**2+(sigma*model)**2.))]

    print """Type Ia SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqIa),len(chisqIa))
    print """Type Ibc SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqIbc),len(chisqIbc))
    print """Type II SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqII),len(chisqII))

    #now compute the likelihoods
    eIa = array(normIa)*exp(-array(chisqIa)/2.)
    eIbc = array(normIbc)*exp(-array(chisqIbc)/2.)
    eII = array(normII)*exp(-array(chisqII)/2.)

    chiIa = sum(eIa)
    chiIbc = sum(eIbc)
    chiII = sum(eII)

    p_Ia = chiIa/(chiIa+chiIbc+chiII)
    p_Ibc = chiIbc/(chiIa+chiIbc+chiII)
    p_II = chiII/(chiIa+chiIbc+chiII)

    print """likelihoods:
Type Ia   Type Ib/c  Type II
%.4f      %.4f       %.4f"""%(p_Ia,p_Ibc,p_II)

    print """odds ratios:
Type Ia/Ib/c  Type Ia/II
%.4f      %.4f"""%(chiIa/chiIbc,chiIa/chiII)

    Iarow = where(chisqIa == min(chisqIa))[0][0]
    Ibcrow = where(chisqIbc == min(chisqIbc))[0][0]
    IIrow = where(chisqII == min(chisqII))[0][0]

    if ret: return mIa, mIbc, mII, eIa/chiIa, eIbc/chiIbc, eII/chiII, Iarow, Ibcrow, IIrow, Iaparams, Ibcparams, IIparams
    else: return

def plotbestmaglc(datfile,clobber=True,paperfig=True,showrange=True,Nsim=2000,cadence=0.5):
    """plot the best-fitting light curves 
    for Type Ia, Ib/c and II SNe
    from SNANA simulations"""

    import rcpar
    close('all')
    rcpar.lcpaperfig()
    import classify

    #set up the axes
    ax1 = axes( [0.12,0.09,0.75,0.27] )
    ax2 = axes( [0.12,0.36,0.75,0.27] )
    ax3 = axes( [0.12,0.63,0.75,0.27] )

    sn=snana.SuperNova(SNDATA_ROOT+'/SIM/'+datfile)

    pkmjdrange = [sn.mjdpk-sn.mjdpkerr,sn.mjdpk+sn.mjdpkerr]
    zmin = sn.z-sn.zerr
    zmax = sn.z+sn.zerr
    simname = 'HST_classify_%s'%(sn.name)
    simname = classify.doClassSim(
        simroot=simname, zrange=[zmin,zmax],
        classfractions=classify.classfracz(sn.z),
        NsimTot=Nsim, bandlist=''.join(sn.bandlist),
        mjdlist = range(int(min(sn.MJD)-35),int(max(sn.MJD)+35),1), pkmjdrange=pkmjdrange,
        clobber=clobber, cadence=cadence )
    sn.getClassSim(Nsim=Nsim,clobber=False)

    #plot the real data
    for filt,color,label in zip(['I','Z','J','H'][::-1],['darkorange','purple','g','r'][::-1],['F814W','F850LP','F125W','F160W'][::-1]):
        row=where(sn.FLT == filt)
        ax1.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color)
        ax2.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color,label=label)
        ax3.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color)
    
    mIa, mIbc, mII, chisqIa, chisqIbc, chisqII, Iarow, Ibcrow, IIrow, Iaparams, Ibcparams, IIparams = chi2magfit(datfile)

    #find the best-fit light curves
    mjd = 4*range(int(min(sn.MJD)-35),int(max(sn.MJD)+35),1)
    snanafilt = ['H']*int(len(mjd)/4.)+['J']*int(len(mjd)/4.)+['Z']*int(len(mjd)/4.)+['I']*int(len(mjd)/4.)

    #Type Ia - best-fit light curve and 68% of the evidence
    x,indIa=[],[]
    for i,j in zip(sort(chisqIa)[::-1],argsort(chisqIa[::-1])):
        x += [i]
        indIa += [j]
        if x>0.68: continue

    mIabest = zeros([len(indIa),len(mjd)])
    m = zeros([len(sn.ClassSim.Ia.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.Ia.samplephot(MJD)
        m[:,i] = sn.ClassSim.Ia.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.Ia.PTROBS_MIN)):
        if not len(where(mIa[i,:]==99)[0]):
            if count==Iarow: bestlc_Ia = m[i,:]
            if count in indIa: 
                mIabest[j,:] = 10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplc,bottomlc=[],[]
    for col in range(len(mjd)):
        toplc+=[max(mIabest[:,col])]
        bottomlc+=[min(mIabest[:,col])]

    #Type Ibc - best-fit light curve and 68% of the evidence
    x,indIbc=[],[]
    for i,j in zip(sort(chisqIbc)[::-1],argsort(chisqIbc[::-1])):
        x += [i]
        indIbc += [j]
        if x>0.68: continue

    mIbcbest = zeros([len(indIbc),len(mjd)])
    m = zeros([len(sn.ClassSim.Ibc.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.Ibc.samplephot(MJD)
        m[:,i] = sn.ClassSim.Ibc.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.Ibc.PTROBS_MIN)):
        if not len(where(mIbc[i,:]==99)[0]):
            if count==Ibcrow: bestlc_Ibc = m[i,:]
            if count in indIbc: 
                mIbcbest[j,:] = 10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplcIbc,bottomlcIbc=[],[]
    for col in range(len(mjd)):
        toplcIbc+=[max(mIbcbest[:,col])]
        bottomlcIbc+=[min(mIbcbest[:,col])]

    #Type II - best-fit light curve and 68% of the evidence
    x,indII=[],[]
    for i,j in zip(sort(chisqII)[::-1],argsort(chisqII[::-1])):
        x += [i]
        indII += [j]
        if x>0.68: continue

    mIIbest = zeros([len(indII),len(mjd)])
    m = zeros([len(sn.ClassSim.II.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.II.samplephot(MJD)
        m[:,i] = sn.ClassSim.II.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.II.PTROBS_MIN)):
        if not len(where(mII[i,:]==99)[0]):
            if count==IIrow: bestlc_II = m[i,:]
            if count in indII: 
                mIIbest[j,:] = 10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplcII,bottomlcII=[],[]
    for col in range(len(mjd)):
        toplcII+=[max(mIIbest[:,col])]
        bottomlcII+=[min(mIIbest[:,col])]

    #plot the best-fit light curves
    for filt,color in zip(['I','Z','J','H'],['darkorange','purple','g','r']):
        row=where(array(snanafilt) == filt)[0]
        mjd,bestlc_Ia,bestlc_Ibc,bestlc_II=array(mjd),array(bestlc_Ia),array(bestlc_Ibc),array(bestlc_II)
        ax1.plot(mjd[row],10**(-0.4*(bestlc_Ia[row]-27.5)),color=color)
        ax2.plot(mjd[row],10**(-0.4*(bestlc_Ibc[row]-27.5)),color=color)
        ax3.plot(mjd[row],10**(-0.4*(bestlc_II[row]-27.5)),color=color)
        if showrange:
            fill_between(ax1,mjd[row],array(bottomlc)[row],array(toplc)[row],color=color,alpha=0.2)
            fill_between(ax2,mjd[row],array(bottomlcIbc)[row],array(toplcIbc)[row],color=color,alpha=0.2)
            fill_between(ax3,mjd[row],array(bottomlcII)[row],array(toplcII)[row],color=color,alpha=0.2)

    #for SN Wilson, the specific commands to make things look good
    if paperfig:
            ax1.set_ylim([-2,30])
            ax2.set_ylim([-2,25])
            ax3.set_ylim([-2,25])
            
            ax1.set_xlim([55500,55630])
            ax2.set_xlim([55500,55630])
            ax3.set_xlim([55500,55630])
            
            ax2.set_xticks([55500,55525,55550,55575,55600,55625])
            ax2.set_xticklabels(['','','','','',''])
            ax3.set_xticks([55500,55525,55550,55575,55600,55625])
            ax3.set_xticklabels(['55500','55525','55550','55575','55600','55625'])

            ax1.set_yticks([0,5,10,15,20,25,30])
            ax2.set_yticks([0,5,10,15,20,25])
            ax3.set_yticks([0,5,10,15,20,25])

            ax1.set_yticklabels(['','5','10','15','20','25',''])
            ax2.set_yticklabels(['','5','10','15','20',''])
            ax3.set_yticklabels(['','5','10','15','20',''])

            ax1.yaxis.set_ticks_position('both')
            ax2.yaxis.set_ticks_position('both')
            ax3.yaxis.set_ticks_position('both')
            
            ax1.text(55590,17,r"""Type Ia
Best $\chi^2$ = 26.0""",color='r',fontsize='medium')
            ax2.text(55590,15,r"""Type Ib/c
Best $\chi^2$ = 90.8""",color='g',fontsize='medium')
            ax3.text(55590,15,r"""Type II
Best $\chi^2$ = 118.6""",color='b',fontsize='medium')

            ax1.set_xlabel('MJD',fontsize='medium')
            ax2.set_ylabel('Flux',fontsize='medium')

            ax3.xaxis.set_ticks_position('top')

            ax2.legend(loc='lower right',prop={'size':10},numpoints=1)

            ax3top = ax3.twiny()
            ax3top.set_xlabel('Time relative to peak (rest frame)',fontsize='medium')
            ax3top.set_xlim( (array(ax1.get_xlim()) - 55539.)/2.914 )
            ax3top.set_xticks([-10,0,10,20,30])
            ax3.xaxis.set_ticklabels(['','','','',''])

            ax1.set_xticks([55500,55525,55550,55575,55600,55625])
            ax1.set_xticklabels(['55500','55525','55550','55575','55600','55625'])

    return

def chi2colorfit(datfile,clobber=False,Nsim=20000):
    """do chi2 fitting to a supernova light 
    curve using SNANA simulations.  Keep color
    as a free parameter"""
    sn=snana.SuperNova(SNDATA_ROOT+'/SIM/'+datfile)
    sn.getClassSim(Nsim=Nsim,clobber=clobber)

    #Type Ia chi-squared
    mIa = zeros([len(sn.ClassSim.Ia.PTROBS_MIN),len(sn.MJD)])
    for MJD,FLT,i in zip(sn.MJD,sn.FLT,range(len(sn.MJD))):
        sn.ClassSim.Ia.samplephot(MJD)
        mIa[:,i] = sn.ClassSim.Ia.__dict__['%s%i'%(FLT, int(MJD))]

    chisqIa = []
    Iaparams = []
    for i in range(len(sn.ClassSim.Ia.PTROBS_MIN)):
        if not len(where(mIa[i,:]==99)[0]):
            fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mIa[i,:],'flt':sn.FLT}
            mvals=mpfit.mpfit(lsfunc_color,ones(len(unique(sn.FLT))),functkw=fa)
            Iaparams = Iaparams + [mvals.params]
            
            scale=ones(len(sn.FLT))
            for f,j in zip(unique(sn.FLT),range(len(unique(sn.FLT)))):
                row=where(sn.FLT==f)[0]
                scale[row]=mvals.params[j]
            model = array(scale)*10**(-0.4*(mIa[i,:]-27.5))

            chisqIa = chisqIa + [sum((model - sn.FLUXCAL)**2/sn.FLUXCALERR**2)]

    #Type Ib/c chi-squared
    mIbc = zeros([len(sn.ClassSim.Ibc.PTROBS_MIN),len(sn.MJD)])
    for MJD,FLT,i in zip(sn.MJD,sn.FLT,range(len(sn.MJD))):
        sn.ClassSim.Ibc.samplephot(MJD)
        mIbc[:,i] = sn.ClassSim.Ibc.__dict__['%s%i'%(FLT, int(MJD))]

    chisqIbc = []
    Ibcparams = []
    count=0
    for i in range(len(sn.ClassSim.Ibc.PTROBS_MIN)):
        if not len(where(mIbc[i,:]==99)[0]):
            count=count+1
            fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mIbc[i,:],'flt':sn.FLT}
            mvals=mpfit.mpfit(lsfunc_color,ones(len(unique(sn.FLT))),functkw=fa)
            Ibcparams = Ibcparams + [mvals.params]

            scale=ones(len(sn.FLT))
            for f,j in zip(unique(sn.FLT),range(len(unique(sn.FLT)))):
                row=where(sn.FLT==f)[0]
                scale[row]=mvals.params[j]
            model = array(scale)*10**(-0.4*(mIbc[i,:]-27.5))
            
            chisqIbc = chisqIbc + [sum((model - sn.FLUXCAL)**2/sn.FLUXCALERR**2)]

    #Type II chi-squared
    mII = zeros([len(sn.ClassSim.II.PTROBS_MIN),len(sn.MJD)])
    for MJD,FLT,i in zip(sn.MJD,sn.FLT,range(len(sn.MJD))):
        sn.ClassSim.II.samplephot(MJD)
        mII[:,i] = sn.ClassSim.II.__dict__['%s%i'%(FLT, int(MJD))]

    chisqII = []
    IIparams = []
    for i in range(len(sn.ClassSim.II.PTROBS_MIN)):
        if not len(where(mII[i,:]==99)[0]):
            fa = {'x':sn.MJD,'y':sn.FLUXCAL,'err':zeros(len(sn.MJD)),'model_mag':mII[i,:],'flt':sn.FLT}
            mvals=mpfit.mpfit(lsfunc_color,ones(len(unique(sn.FLT))),functkw=fa)
            IIparams = IIparams + [mvals.params]

            scale=ones(len(sn.FLT))
            for f,j in zip(unique(sn.FLT),range(len(unique(sn.FLT)))):
                row=where(sn.FLT==f)[0]
                scale[row]=mvals.params[j]
            model = array(scale)*10**(-0.4*(mII[i,:]-27.5))
            
            chisqII = chisqII + [sum((model - sn.FLUXCAL)**2/sn.FLUXCALERR**2)]

    print """Type Ia SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqIa),len(chisqIa))
    print """Type Ibc SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqIbc),len(chisqIbc))
    print """Type II SNe have minimum chi squared: %.2f
computed from %.f SNe"""%(min(chisqII),len(chisqII))

    #now compute the likelihoods
    eIa = exp(-array(chisqIa)/2.)
    eIbc = exp(-array(chisqIbc)/2.)
    eII = exp(-array(chisqII)/2.)

    chiIa = sum(exp(-array(chisqIa)/2.))
    chiIbc = sum(exp(-array(chisqIbc)/2.))
    chiII = sum(exp(-array(chisqII)/2.))

    p_Ia = chiIa/(chiIa+chiIbc+chiII)
    p_Ibc = chiIbc/(chiIa+chiIbc+chiII)
    p_II = chiII/(chiIa+chiIbc+chiII)

    print """likelihoods:
Type Ia   Type Ib/c  Type II
%.4f      %.4f       %.4f"""%(p_Ia,p_Ibc,p_II)

    print """odds ratios:
Type Ia/Ib/c  Type Ia/II
%.4f      %.4f"""%(chiIa/chiIbc,chiIa/chiII)

    Iarow = where(chisqIa == min(chisqIa))[0][0]
    Ibcrow = where(chisqIbc == min(chisqIbc))[0][0]
    IIrow = where(chisqII == min(chisqII))[0][0]

    return mIa, mIbc, mII, eIa/chiIa, eIbc/chiIbc, eII/chiII, Iarow, Ibcrow, IIrow, Iaparams, Ibcparams, IIparams

def plotbestcolorlc(datfile,clobber=True,paperfig=True,showrange=True,Nsim=2000,cadence=0.5):
    """plot the best-fitting light curves 
    for Type Ia, Ib/c and II SNe
    from SNANA simulations.  Keeps color as a
    free parameter."""

    import rcpar
    close('all')
    rcpar.lcpaperfig()
    import classify

    #set up the axes
    ax3 = axes( [0.12,0.09,0.75,0.27] )
    ax2 = axes( [0.12,0.36,0.75,0.27] )
    ax1 = axes( [0.12,0.63,0.75,0.27] )

    sn=snana.SuperNova(SNDATA_ROOT+'/SIM/'+datfile)

    pkmjdrange = [sn.mjdpk-sn.mjdpkerr,sn.mjdpk+sn.mjdpkerr]
    zmin = sn.z-sn.zerr
    zmax = sn.z+sn.zerr
    simname = 'HST_classify_%s'%(sn.name)
    simname = classify.doClassSim(
        simroot=simname, zrange=[zmin,zmax],
        classfractions=classify.classfracz(sn.z),
        NsimTot=Nsim, bandlist=''.join(sn.bandlist),
        mjdlist = range(int(min(sn.MJD)-35),int(max(sn.MJD)+35),1), pkmjdrange=pkmjdrange,
        clobber=clobber, cadence=cadence )
    sn.getClassSim(Nsim=Nsim,clobber=False)

    #plot the real data
    for filt,color,label in zip(['J','H'][::-1],['g','r'][::-1],['F125W','F160W'][::-1]):
        row=where(sn.FLT == filt)
        ax1.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color)
        ax2.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color,label=label)
        ax3.errorbar(sn.MJD[row],sn.FLUXCAL[row],yerr=sn.FLUXCALERR[row],fmt='o',color=color)
    
    mIa, mIbc, mII, chisqIa, chisqIbc, chisqII, Iarow, Ibcrow, IIrow, Iaparams, Ibcparams, IIparams = chi2colorfit(datfile)

    #find the best-fit light curves
    mjd = 4*range(int(min(sn.MJD)-35),int(max(sn.MJD)+35),1)
    snanafilt = ['H']*int(len(mjd)/4.)+['J']*int(len(mjd)/4.)+['Z']*int(len(mjd)/4.)+['I']*int(len(mjd)/4.)
    Iascale = [Iaparams[Iarow][0]]*int(len(mjd)/4.)+[Iaparams[Iarow][2]]*int(len(mjd)/4.)+[Iaparams[Iarow][1]]*int(len(mjd)/4.)+[Iaparams[Iarow][3]]*int(len(mjd)/4.)
    Ibcscale = [Ibcparams[Ibcrow][0]]*int(len(mjd)/4.)+[Ibcparams[Ibcrow][2]]*int(len(mjd)/4.)+[Ibcparams[Ibcrow][1]]*int(len(mjd)/4.)+[Ibcparams[Ibcrow][3]]*int(len(mjd)/4.)
    IIscale = [IIparams[IIrow][0]]*int(len(mjd)/4.)+[IIparams[IIrow][2]]*int(len(mjd)/4.)+[IIparams[IIrow][1]]*int(len(mjd)/4.)+[IIparams[IIrow][3]]*int(len(mjd)/4.)

    #Type Ia - best-fit light curve and 68% of the evidence
    x,indIa=[],[]
    for i,j in zip(sort(chisqIa)[::-1],argsort(chisqIa[::-1])):
        x += [i]
        indIa += [j]
        if x>0.68: continue

    mIabest = zeros([len(indIa),len(mjd)])
    m = zeros([len(sn.ClassSim.Ia.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.Ia.samplephot(MJD)
        m[:,i] = sn.ClassSim.Ia.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.Ia.PTROBS_MIN)):
        if not len(where(mIa[i,:]==99)[0]):
            if count==Iarow: bestlc_Ia = m[i,:]
            if count in indIa: 
                pars = [Iaparams[j][0]]*int(len(mjd)/4.)+[Iaparams[j][2]]*int(len(mjd)/4.)+[Iaparams[j][1]]*int(len(mjd)/4.)+[Iaparams[j][3]]*int(len(mjd)/4.)
                mIabest[j,:] = pars*10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplc,bottomlc=[],[]
    for col in range(len(mjd)):
        toplc+=[max(mIabest[:,col])]
        bottomlc+=[min(mIabest[:,col])]

    #Type Ibc - best-fit light curve and 68% of the evidence
    x,indIbc=[],[]
    for i,j in zip(sort(chisqIbc)[::-1],argsort(chisqIbc[::-1])):
        x += [i]
        indIbc += [j]
        if x>0.68: continue

    mIbcbest = zeros([len(indIbc),len(mjd)])
    m = zeros([len(sn.ClassSim.Ibc.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.Ibc.samplephot(MJD)
        m[:,i] = sn.ClassSim.Ibc.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.Ibc.PTROBS_MIN)):
        if not len(where(mIbc[i,:]==99)[0]):
            if count==Ibcrow: bestlc_Ibc = m[i,:]
            if count in indIbc: 
                pars = [Ibcparams[j][0]]*int(len(mjd)/4.)+[Ibcparams[j][2]]*int(len(mjd)/4.)+[Ibcparams[j][1]]*int(len(mjd)/4.)+[Ibcparams[j][3]]*int(len(mjd)/4.)
                mIbcbest[j,:] = pars*10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplcIbc,bottomlcIbc=[],[]
    for col in range(len(mjd)):
        toplcIbc+=[max(mIbcbest[:,col])]
        bottomlcIbc+=[min(mIbcbest[:,col])]

    #Type II - best-fit light curve and 68% of the evidence
    x,indII=[],[]
    for i,j in zip(sort(chisqII)[::-1],argsort(chisqII[::-1])):
        x += [i]
        indII += [j]
        if x>0.68: continue

    mIIbest = zeros([len(indII),len(mjd)])
    m = zeros([len(sn.ClassSim.II.PTROBS_MIN),len(mjd)])
    for MJD,FLT,i in zip(mjd,snanafilt,range(len(mjd))):
        sn.ClassSim.II.samplephot(MJD)
        m[:,i] = sn.ClassSim.II.__dict__['%s%i'%(FLT, int(MJD))]
    count,j=0,0
    for i in range(len(sn.ClassSim.II.PTROBS_MIN)):
        if not len(where(mII[i,:]==99)[0]):
            if count==IIrow: bestlc_II = m[i,:]
            if count in indII: 
                pars = [IIparams[j][0]]*int(len(mjd)/4.)+[IIparams[j][2]]*int(len(mjd)/4.)+[IIparams[j][1]]*int(len(mjd)/4.)+[IIparams[j][3]]*int(len(mjd)/4.)
                mIIbest[j,:] = pars*10**(-0.4*(m[i,:]-27.5))
                j=j+1
            count=count+1
    toplcII,bottomlcII=[],[]
    for col in range(len(mjd)):
        toplcII+=[max(mIIbest[:,col])]
        bottomlcII+=[min(mIIbest[:,col])]

    #plot the best-fit light curves
    for filt,color in zip(['J','H'],['g','r']):
        row=where(array(snanafilt) == filt)[0]
        mjd,bestlc_Ia,bestlc_Ibc,bestlc_II=array(mjd),array(bestlc_Ia),array(bestlc_Ibc),array(bestlc_II)
        ax1.plot(mjd[row],array(Iascale)[row]*10**(-0.4*(bestlc_Ia[row]-27.5)),color=color)
        ax2.plot(mjd[row],array(Ibcscale)[row]*10**(-0.4*(bestlc_Ibc[row]-27.5)),color=color)
        ax3.plot(mjd[row],array(IIscale)[row]*10**(-0.4*(bestlc_II[row]-27.5)),color=color)
        if showrange:
            fill_between(ax1,mjd[row],array(bottomlc)[row],array(toplc)[row],color=color,alpha=0.2)
            fill_between(ax2,mjd[row],array(bottomlcIbc)[row],array(toplcIbc)[row],color=color,alpha=0.2)
            fill_between(ax3,mjd[row],array(bottomlcII)[row],array(toplcII)[row],color=color,alpha=0.2)

    #for SN Wilson, the specific commands to make things look good
    if paperfig:
            ax1.set_ylim([-2,30])
            ax2.set_ylim([-2,25])
            ax3.set_ylim([-2,25])
            
            ax1.set_xlim([55500,55630])
            ax2.set_xlim([55500,55630])
            ax3.set_xlim([55500,55630])
            
            ax2.set_xticks([55500,55525,55550,55575,55600,55625])
            ax2.set_xticklabels(['','','','','',''])
            ax3.set_xticks([55500,55525,55550,55575,55600,55625])
            ax3.set_xticklabels(['55500','55525','55550','55575','55600','55625'])

            ax1.set_yticks([0,5,10,15,20,25,30])
            ax2.set_yticks([0,5,10,15,20,25])
            ax3.set_yticks([0,5,10,15,20,25])

            ax1.set_yticklabels(['','5','10','15','20','25',''])
            ax2.set_yticklabels(['','5','10','15','20',''])
            ax3.set_yticklabels(['','5','10','15','20',''])

            ax1.yaxis.set_ticks_position('both')
            ax2.yaxis.set_ticks_position('both')
            ax3.yaxis.set_ticks_position('both')
            
            ax1.text(55590,17,r"""Type Ia
Best $\chi^2$ = 23.5""",color='r',fontsize='medium')
            ax2.text(55590,15,r"""Type Ib/c
Best $\chi^2$ = 26.9""",color='g',fontsize='medium')
            ax3.text(55590,15,r"""Type II
Best $\chi^2$ = 112.5""",color='b',fontsize='medium')

            ax3.set_xlabel('MJD',fontsize='medium')
            ax2.set_ylabel('Flux',fontsize='medium')

            ax1.xaxis.set_ticks_position('top')

            ax2.legend(loc='lower right',prop={'size':10},numpoints=1)

            ax3top = ax1.twiny()
            ax3top.set_xlabel('Time relative to peak (rest frame)',fontsize='medium')
            ax3top.set_xlim( (array(ax1.get_xlim()) - 55539.)/2.914 )
            ax3top.set_xticks([-10,0,10,20,30])
            ax1.xaxis.set_ticklabels(['','','','',''])

            ax3.set_xticks([55500,55525,55550,55575,55600,55625])
            ax3.set_xticklabels(['55500','55525','55550','55575','55600','55625'])

    return
