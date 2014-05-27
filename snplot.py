from numpy import *
from pylab import *
import glob
import os
import pyfits
#from __init__ import SuperNova
from snana import SuperNova
import exceptions

filt_color = {'g':'b',
              'r':'r',
              'i':'g',
              'z':'darkorange',
              'y':'purple',
              'S':'DarkMagenta','T':'DarkOrchid','U':'DeepPink','C':'HotPink','W':'k',
              'B':'DarkSlateBlue','G':'ForestGreen','V':'Teal','R':'FireBrick',
              'X':'Chocolate','I':'Crimson','Z':'Sienna',
              'Y':'DarkCyan','M':'DarkBlue','J':'DarkGreen','N':'DarkOrange','H':'DarkRed',
              'L':'DarkTurquoise','O':'DarkSlateGrey','P':'SeaGreen','Q':'Brown' }
filter2band = { 'F218W':'D','F225W':'S','F275W':'T','F300X':'E',
                'F336W':'U','F390W':'C','F350LP':'W',
                'F435W':'B','F475W':'G','F555W':'F','F606W':'V','F625W':'R',
                'F775W':'X','F814W':'I','F850LP':'Z',
                'F125W':'J','F160W':'H','F125W+F160W':'A',
                'F105W':'Y','F110W':'M','F140W':'N',
                'F098M':'L','F127M':'O','F139M':'P','F153M':'Q',
                'G141':'4','G102':'2','blank':'0'
                }


def plotlc(lcfile,ax=''):
    """plot SNANA light curve"""

    if not ax:
        close('all')
        ax = axes([0.1,0.1,0.8,0.8])

    sn = SuperNova(lcfile)
    
    #plot the data points
    for m,f,e,flt in zip(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR,sn.FLT):
        ax.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')

    ax.set_ylabel('Flux')
    ax.set_xlabel('MJD')

    #plot the SALT2 results
    salt2file = glob.glob('%s.OUT'%sn.SNID)
    if salt2file:
        salt2file = salt2file[0]
        for i in sn.FILTERS:
            try:
                mjd,f,df = read_list(salt2file,filt=i)
                f,df = f[argsort(mjd)],df[argsort(mjd)]
                mjd.sort()
                ax.plot(mjd,f,color=filt_color[i])
                fill_between(ax,mjd,f-df,f+df,color=filt_color[i],alpha=0.2)
            except: print('Filter %s was not fit by SNANA'%i)

        #include the salt2 parameters and the appropriate xlim, ylim
        fitresfile = glob.glob('snfit_%s.fitres'%sn.SNID)[0]
        try: 
            z,ze,c,ce,x1,x1e,mjd,mjde,mb,mbe,chi2,dof = loadtxt(fitresfile,unpack=True,skiprows=6,
                                                                usecols=[2,3,6,7,8,9,10,11,12,13,17,18])

            ax.text(mjd + 20, max(sn.FLUXCAL)*2./3.,"""SN %s
$z$ = %.2f
$x_1$ = %.2f +/- %.2f
$C$ = %.2f +/- %.2f
$m_B$ = %.2f +/- %.2f
$\chi^2$ = %.2f"""%(sn.SNID,z,x1,x1e,c,ce,mb,mbe,chi2/dof),fontsize='small')
                
        except: print('Empty fitres file')

        ax.set_ylim([0,max(sn.FLUXCAL)+5])
        ax.set_xlim([mjd-20,mjd+60])

    else:
        import doSALT2
        doSALT2.run(lcfile)
        try: salt2file = glob.glob('%s.OUT'%sn.SNID)[0]
        except: 
            doSALT2.run(lcfile,mjdint=10)
            salt2file = glob.glob('%s.OUT'%sn.SNID)[0]
        for i in sn.FILTERS:
            try:
                mjd,f,df = read_list(salt2file,filt=i)
                f,df = f[argsort(mjd)],df[argsort(mjd)]
                mjd.sort()
                ax.plot(mjd,f,color=filt_color[i])
                fill_between(ax,mjd,f-df,f+df,color=filt_color[i],alpha=0.2)
            except: print('Filter %s was not fit by SNANA'%i)

        #include the salt2 parameters and the appropriate xlim, ylim
        fitresfile = glob.glob('snfit_%s.fitres'%sn.SNID)[0]
        z,ze,c,ce,x1,x1e,mjd,mjde,mb,mbe,chi2,dof = loadtxt(fitresfile,unpack=True,skiprows=6,
                                                            usecols=[2,3,6,7,8,9,10,11,12,13,17,18])

        ax.set_ylim([0,max(sn.FLUXCAL)+5])
        ax.set_xlim([mjd-20,mjd+60])

        ax.text(mjd + 20, max(sn.FLUXCAL)*2./3.,"""SN %s
$z$ = %.2f
$x_1$ = %.2f +/- %.2f
$C$ = %.2f +/- %.2f
$m_B$ = %.2f +/- %.2f
$\chi^2$ = %.2f"""%(sn.SNID,z,x1,x1e,c,ce,mb,mbe,chi2/dof),fontsize='small')

    return

def psnidplot(lcfile,ax='',specz=True):
    """plot PSNID classification curves"""

    import doPSNID
    import exceptions

    if not ax:
        close('all')
        ax1 = axes( [0.12,0.09,0.75,0.27]  )
        ax2 = axes( [0.12,0.36,0.75,0.27] )
        ax3 = axes( [0.12,0.63,0.75,0.27] )

    sn = SuperNova(lcfile)
    p_Ia,p_Ibc,p_II,chi2_Ia,chi2_Ibc,chi2_II,ndof_Ia,ndof_Ibc,ndof_II= doPSNID.run(lcfile)
    
    fitresfile = glob.glob('snfit_%s.fitres'%sn.SNID)
    if fitresfile:
        try: z,ze,c,ce,x1,x1e,mjd,mjde,mb,mbe,chi2,dof = loadtxt(fitresfile[0],unpack=True,skiprows=6,
                                                                 usecols=[2,3,6,7,8,9,10,11,12,13,17,18])
        except: 
            mjd = sn.MJD[where(sn.FLUXCAL == max(sn.FLUXCAL))[0]]
            z = sn.z
    else:
        import doSALT2
        doSALT2.run(lcfile,lcout=False)
        fitresfile = glob.glob('snfit_%s.fitres'%sn.SNID)
        try: z,ze,c,ce,x1,x1e,mjd,mjde,mb,mbe,chi2,dof = loadtxt(fitresfile[0],unpack=True,skiprows=6,
                                                            usecols=[2,3,6,7,8,9,10,11,12,13,17,18])
        except: 
            mjd = sn.MJD[where(sn.FLUXCAL == max(sn.FLUXCAL))[0]]
            z = sn.z

    fileroot = '/usr/local/psnid_ps1/output/images_stiter9999/SN%s/curves/'%sn.SNID    
    if specz:
        Iafile = fileroot + 'SN%s_z-Ia.qdp'%sn.SNID
        Ibcfile = fileroot + 'SN%s_z-Ibc.qdp'%sn.SNID
        IIfile = fileroot + 'SN%s_z-II.qdp'%sn.SNID
    else:
        Iafile = fileroot + 'SN%s_nz-Ia.qdp'%sn.SNID
        Ibcfile = fileroot + 'SN%s_nz-Ibc.qdp'%sn.SNID
        IIfile = fileroot + 'SN%s_nz-II.qdp'%sn.SNID

    for m,f,e,flt in zip(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR,sn.FLT):
        ax1.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax2.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax3.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')

    ax2.set_ylabel('Flux')
    ax1.set_xlabel('MJD')
    ax1.set_ylim([0,max(sn.FLUXCAL)+5])
    ax1.set_xlim([mjd-20,mjd+60])
    ax2.set_ylim([0,max(sn.FLUXCAL)+5])
    ax2.set_xlim([mjd-20,mjd+60])
    ax3.set_ylim([0,max(sn.FLUXCAL)+5])
    ax3.set_xlim([mjd-20,mjd+60])

    ax3.text(mjd + 20, max(sn.FLUXCAL)*1./2.,r"""SN%s
$z$ = %.2f
Type Ia
$\chi^2/\nu$ = %.1f/%i"""%(sn.SNID,z,chi2_Ia,ndof_Ia))
    ax2.text(mjd + 20, max(sn.FLUXCAL)*2./3.,r"""Type Ibc
$\chi^2/\nu$ = %.1f/%i"""%(chi2_Ibc,ndof_Ibc))
    ax1.text(mjd + 20, max(sn.FLUXCAL)*2./3.,r"""Type II
$\chi^2/\nu$ = %.1f/%i"""%(chi2_II,ndof_II))

    for f,i in zip(sn.FILTERS,range(len(sn.FILTERS))):
        mjdIa,mjdIbc,mjdII,magIa,magIbc,magII = [],[],[],[],[],[]
        for file in [Iafile,Ibcfile,IIfile]:
            a=open(file,'r')
            count=-1
            for line in a:
                if 'no no no' in line: count += 1
                if 'no no no' not in line and count==i+3:
                    if file == Iafile:
                        mjdIa += [float(filter(None,line.split(' '))[0])]
                        magIa += [10**(-0.4*(float(filter(None,line.split(' '))[1]) - 27.5))]
                    if file == Ibcfile:
                        mjdIbc += [float(filter(None,line.split(' '))[0])]
                        magIbc += [10**(-0.4*(float(filter(None,line.split(' '))[1]) - 27.5))]
                    if file == IIfile:
                        mjdII += [float(filter(None,line.split(' '))[0])]
                        magII += [10**(-0.4*(float(filter(None,line.split(' '))[1]) - 27.5))]

            a.close()
        mjdIa = array(mjdIa) + 55000.
        mjdIbc = array(mjdIbc) + 55000.
        mjdII = array(mjdII) + 55000.

        ax3.plot(mjdIa,magIa,color=filt_color[f])
        ax2.plot(mjdIbc,magIbc,color=filt_color[f],label=i)
        ax1.plot(mjdII,magII,color=filt_color[f])

    return

def plotBayesGrid(lcfile,ax='',trestrange=[-15,30],CCmoderr=0.1,
                  Nsim=2000,kcorfile = 'PS1/kcor_tonry_SALT2.fits',
                  nlogz=0):
    close('all')

    import snana
    sn = snana.SuperNova(lcfile)
    sn.doGridClassify(clobber=3,noCosmoPrior=True,kcorfile = kcorfile, 
                      trestrange=trestrange, modelerror = [0.08, CCmoderr], Nsim = Nsim, nlogz = nlogz)

    if not ax:
        close('all')
        ax1 = axes( [0.12,0.09,0.75,0.27]  )
        ax2 = axes( [0.12,0.36,0.75,0.27] )
        ax3 = axes( [0.12,0.63,0.75,0.27] )

    #plot the data points
    for m,f,e,flt in zip(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR,sn.FLT):
        ax1.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax2.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax3.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')

    for f in sn.FILTERS:
        row = where(sn.ClassSim.Ia.FLT[0] == f)[0]
        ax1.plot(sn.trestbestIa*(1+sn.z) + sn.pkmjd,sn.bestlcIa[row],color=filt_color[f])
        ax2.plot(sn.trestbestIbc*(1+sn.z) + sn.pkmjd,sn.bestlcIbc[row],color=filt_color[f])
        ax3.plot(sn.trestbestII*(1+sn.z) + sn.pkmjd,sn.bestlcII[row],color=filt_color[f])

    ax2.set_ylabel('Flux')
    ax1.set_xlabel('MJD')
    ax1.set_ylim([0,max(sn.FLUXCAL)+5])
    ax1.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
    ax2.set_ylim([0,max(sn.FLUXCAL)+5])
    ax2.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
    ax3.set_ylim([0,max(sn.FLUXCAL)+5])
    ax3.set_xlim([sn.pkmjd-20,sn.pkmjd+60])

    print('# PIa  PIbc  PII')
    print sn.PIa,sn.PIbc,sn.PII
    import pdb; pdb.set_trace()

    return

def plotcaracalla(lcfile='/Users/David/caracalla.dat',ax='',trestrange=[-15,30],CCmoderr=0.1,
                  Nsim=2000,kcorfile = 'PS1/kcor_tonry_SALT2.fits',
                  nlogz=0):

    close('all')

    import snana
    sn = snana.SuperNova(lcfile)
    sn.doGridClassify(clobber=3,noCosmoPrior=True,kcorfile = kcorfile, 
                      trestrange=trestrange, modelerror = [0.08, CCmoderr], Nsim = Nsim, nlogz = nlogz)

    if not ax:
        close('all')
        import rcpar
        rcpar.lcpaperfig()
        ax3 = axes( [0.12,0.09,0.75,0.27] )
        ax2 = axes( [0.12,0.36,0.75,0.27] )
        ax1 = axes( [0.12,0.63,0.75,0.27] )

    #plot the data points
    for m,f,e,flt in zip(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR,sn.FLT):
        ax1.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax2.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax3.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
    for flt in sn.FILTERS:
        row = where(array(filter2band.values()) == flt)[0]
        filter = filter2band.keys()[row]
        ax2.plot(min(sn.MJD)-500,sn.FLUXCAL[0],'.',color=filt_color[flt],label=filter)

    for f in sn.FILTERS:
        row = where(sn.ClassSim.Ia.FLT[0] == f)[0]
        ax1.plot(sn.trestbestIa*(1+sn.z) + sn.pkmjd,sn.bestlcIa[row],color=filt_color[f])
        ax2.plot(sn.trestbestIbc*(1+sn.z) + sn.pkmjd,sn.bestlcIbc[row],color=filt_color[f])
        ax3.plot(sn.trestbestII*(1+sn.z) + sn.pkmjd,sn.bestlcII[row],color=filt_color[f])

    ax2.set_ylabel('Flux')
    ax3.set_xlabel('MJD')
    ax1.set_ylim([0,max(sn.FLUXCAL)+10])
    ax1.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
    ax2.set_ylim([0,max(sn.FLUXCAL)+10])
    ax2.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
    ax3.set_ylim([0,max(sn.FLUXCAL)+10])
    ax3.set_xlim([sn.pkmjd-20,sn.pkmjd+60])

    ax1.set_xticklabels(['','','','','',''])
    ax2.set_xticklabels(['','','','','',''])
    ax3.set_xticks([56090,56100,56110,56120,56130,56140,56150,56160])
    ax3.set_xticklabels(['56090','56100','56110','56120','56230','56140','56150','56160'])

    ax1.text(sn.pkmjd+30,max(sn.FLUXCAL)*3/4.,r"""Type Ia
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2Ia),sn.Ndof),color='k',fontsize='medium')
    ax2.text(sn.pkmjd+30,max(sn.FLUXCAL)*3/4.,r"""Type Ib/c
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2Ibc),sn.Ndof),color='k',fontsize='medium')
    ax3.text(sn.pkmjd+30,max(sn.FLUXCAL)*3/4.,r"""Type II
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2II),sn.Ndof),color='k',fontsize='medium')

    ax2.legend(numpoints=1,prop={'size':8})

    print('# PIa  PIbc  PII')
    print sn.PIa,sn.PIbc,sn.PII
    import pdb; pdb.set_trace()

    return

def plottib(lcfile='/Users/David/tib.DAT',ax='',trestrange=[-15,30],CCmoderr=0.1,
            Nsim=2000,kcorfile = 'PS1/kcor_tonry_SALT2.fits',
            nlogz=0):

    close('all')

    import snana
    sn = snana.SuperNova(lcfile)
    sn.doGridClassify(clobber=3,noCosmoPrior=True,kcorfile = kcorfile, 
                      trestrange=trestrange, modelerror = [0.08, CCmoderr], Nsim = Nsim, nlogz = nlogz)

    if not ax:
        close('all')
        import rcpar
        rcpar.lcpaperfig()
        ax3 = axes( [0.12,0.09,0.75,0.27] )
        ax2 = axes( [0.12,0.36,0.75,0.27] )
        ax1 = axes( [0.12,0.63,0.75,0.27] )

    #plot the data points
    for m,f,e,flt in zip(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR,sn.FLT):
        ax1.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax2.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
        ax3.errorbar(m,f,yerr=e,color=filt_color[flt],fmt='.')
    for flt in sn.FILTERS:
        row = where(array(filter2band.values()) == flt)[0]
        filter = filter2band.keys()[row]
        ax2.plot(min(sn.MJD)-500,sn.FLUXCAL[0],'.',color=filt_color[flt],label=filter)

    for f in sn.FILTERS:
        row = where(sn.ClassSim.Ia.FLT[0] == f)[0]
        ax1.plot(sn.trestbestIa*(1+sn.z) + sn.pkmjd,sn.bestlcIa[row],color=filt_color[f])
        ax2.plot(sn.trestbestIbc*(1+sn.z) + sn.pkmjd,sn.bestlcIbc[row],color=filt_color[f])
        ax3.plot(sn.trestbestII*(1+sn.z) + sn.pkmjd,sn.bestlcII[row],color=filt_color[f])

    ax2.set_ylabel('Flux')
    ax3.set_xlabel('MJD')
    ax1.set_ylim([0,max(sn.FLUXCAL)+10])
    ax1.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
    ax2.set_ylim([0,max(sn.FLUXCAL)+10])
    ax2.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
    ax3.set_ylim([0,max(sn.FLUXCAL)+10])
    ax3.set_xlim([sn.pkmjd-20,sn.pkmjd+60])

    ax1.set_xticklabels(['','','','','',''])
    ax2.set_xticklabels(['','','','','',''])
    ax3.set_xticks([55550,55560,55570,55580,55590,55600,55610,55620])
    ax3.set_xticklabels(['55510','55520','55530','55540','55550','55560','55570','55580'])

    ax1.text(sn.pkmjd+30,max(sn.FLUXCAL)*3/4.,r"""Type Ia
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2Ia),sn.Ndof),color='k',fontsize='medium')
    ax2.text(sn.pkmjd+30,max(sn.FLUXCAL)*3/4.,r"""Type Ib/c
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2Ibc),sn.Ndof),color='k',fontsize='medium')
    ax3.text(sn.pkmjd+30,max(sn.FLUXCAL)*3/4.,r"""Type II
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2II),sn.Ndof),color='k',fontsize='medium')

    ax2.legend(numpoints=1,prop={'size':8})

    print('# PIa  PIbc  PII')
    print sn.PIa,sn.PIbc,sn.PII
    import pdb; pdb.set_trace()

    return

def mktib(lcfile='/Users/David/tib.DAT',ax='',trestrange=[-15,30],CCmoderr=0.1,
          Nsim=2000,kcorfile = 'PS1/kcor_tonry_SALT2.fits',
          nlogz=0,sn=''):
    """Separate plots of best-fit tiberius light curves for Brandon"""

    close('all')

    import snana
    if not sn:
        sn = snana.SuperNova(lcfile)
        sn.doGridClassify(clobber=3,noCosmoPrior=True,kcorfile = kcorfile, 
                          trestrange=trestrange, modelerror = [0.08, CCmoderr], Nsim = Nsim, nlogz = nlogz)

    if not ax:
        close('all')
        import rcpar
        reload(rcpar)
        fig = rcpar.fullpaperfig()
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(332)
        ax3 = fig.add_subplot(333)
        ax4 = fig.add_subplot(334)
        ax5 = fig.add_subplot(335)
        ax6 = fig.add_subplot(336)
        ax7 = fig.add_subplot(337)
        ax8 = fig.add_subplot(338)
        ax9 = fig.add_subplot(339)

        filtorder = array([0,4,1,6,7,5,2,3])
        axlist = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
        axlist = array(axlist)[filtorder]

    #plot the data points
    for m,f,e,flt in zip(sn.MJD,sn.FLUXCAL,sn.FLUXCALERR,sn.FLT):
        row = where(flt == unique(sn.FLT))[0][0]
        limrow = where(flt == sn.FLT)[0]
        axlist[row].set_ylim([0,max(sn.FLUXCAL[limrow])+10])
        if flt == 'H' or flt == 'J' or flt == 'Z': axlist[row].set_ylim([0,max(sn.FLUXCAL[limrow])+20])
        if flt == 'R': axlist[row].set_ylim([0,10])
        if flt == 'V': axlist[row].set_ylim([0,6])
        axlist[row].errorbar(m,f,yerr=e,color='k',fmt='o')
        frow = where(array(filter2band.values()) == flt)[0]
        filter = filter2band.keys()[frow]
        if flt != 'H' and flt != 'J' and flt != 'Z' and flt != 'R' and flt != 'V':
            axlist[row].text(sn.pkmjd+30,max(sn.FLUXCAL[limrow]+10.)*6/8.,'%s'%filter,fontsize=20,style='normal')
        elif flt == 'H' or flt == 'J' or flt == 'Z':
            axlist[row].text(sn.pkmjd+30,max(sn.FLUXCAL[limrow]+20.)*6/8.,'%s'%filter,fontsize=20,style='normal')
        elif flt == 'R':
            axlist[row].text(sn.pkmjd+30,10.*6/8.,'%s'%filter,fontsize=20,style='normal')
        elif flt == 'V':
            axlist[row].text(sn.pkmjd+30,6.*6/8.,'%s'%filter,fontsize=20,style='normal')

    for f,ax in zip(unique(sn.FLT),axlist):
        row = where(sn.ClassSim.Ia.FLT[0] == f)[0]
        if f == 'H': 
            pkmjdrow = where(sn.maxLikeIaModel.FLUXCAL[row] == max(sn.maxLikeIaModel.FLUXCAL[row]))[0]
            pkmjd = sn.maxLikeIaModel.MJD[pkmjdrow]
        ax.plot(sn.maxLikeIaModel.MJD[row],sn.maxLikeIaModel.FLUXCAL[row],color='r')
        ax.plot(sn.maxLikeIbcModel.MJD[row],sn.maxLikeIbcModel.FLUXCAL[row],color='g')
        ax.plot(sn.maxLikeIIModel.MJD[row],sn.maxLikeIIModel.FLUXCAL[row],color='b')

    for ax,i in zip(axlist,range(8)):
        ax.set_ylabel('Flux')
        ax.set_xlabel('MJD',fontsize='small')
        ax.set_xlim([sn.pkmjd-20,sn.pkmjd+60])
        ax.set_xticks([55550,55560,55570,55580,55590,55600,55610,55620])
        ax.set_xticklabels(['55550','55560','55570','55580','55590','55600','55610','55620'])
        axtop = ax.twiny()
        if i == 0 or i == 2 or i == 6: axtop.set_xlabel('Time relative SN Ia peak (rest frame)',fontsize='small')
        axtop.set_xlim( (array(ax.get_xlim()) - pkmjd)/(1+sn.z) )
        axtop.set_xticks([-10,0,10,20])

    ax9.set_xticks([-1,-2])
    ax9.set_yticks([-1,-1])
    ax9.set_xticklabels(['','','','','',''])
    ax9.set_yticklabels(['','','','','',''])
    ax9.set_ylim([0,1])
    ax9.set_xlim([0,1])

    ax9.text(0.1,0.7,r"""Type Ia (red curves)
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2Ia),sn.Ndof),color='r',fontsize='large')
    ax9.text(0.1,0.4,r"""Type Ib/c (green curves)
Best match: SN SDSS-002744 (Type Ib)
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2Ibc),sn.Ndof),color='g',fontsize='large')
    ax9.text(0.1,0.1,r"""Type II (blue curves)
Best match: SN SDSS-015339 (Type IIP)
Best $\chi^2/\nu$ = %.1f/%i"""%(min(sn.chi2II),sn.Ndof),color='b',fontsize='large')

    #ax2.legend(numpoints=1,prop={'size':8})

    print('# PIa  PIbc  PII')
    print sn.PIa,sn.PIbc,sn.PII
    #import pdb; pdb.set_trace()

    return

def mkdid():

    close('all')
    import rcpar
    reload(rcpar)
    fig = rcpar.fullpaperfig()
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)
    axlist = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

    filters = ['H','N','M','Y','Z','I','X','R','V']
    files = glob.glob('/Users/David/Didius_salt/HST*fl')
    for flt,ax,i in zip(filters,axlist,range(9)):
        for f in files:
            if '-1-%s'%flt in f:
                mjd,flux,dflux,code = loadtxt(f,unpack=True,usecols=[0,2,3,4])
        datrow = where(code == 1)[0]
        modrow = where(code == 0)[0]
        if flt == 'H':
            pkmjdrow = where(flux[modrow] == max(flux[modrow]))[0]
            pkmjd = mjd[modrow][pkmjdrow]
        ax.errorbar(mjd[datrow],flux[datrow],yerr=dflux[datrow],fmt='o',color='k')
        ax.plot(mjd[modrow],flux[modrow],color='r')
        frow = where(array(filter2band.values()) == flt)[0]
        filter = filter2band.keys()[frow]
        ax.text(56050,ax.get_ylim()[1]*3/4.,filter,fontsize=20)
        axtop = ax.twiny()
        axtop.set_xlim( (array(ax.get_xlim()) - pkmjd)/(1+0.851) )
        axtop.set_xticks([-40,-20,0,20,40,60,80])
        ax.set_xlabel('MJD')
        ax.set_ylabel('Flux')
        if i == 0 or i == 1 or i == 2: axtop.set_xlabel('Time relative SN Ia peak (rest frame)',fontsize='small')
        fill_between(ax,mjd[modrow],flux[modrow]-dflux[modrow],flux[modrow]+dflux[modrow],alpha=0.2,color='r')

    savefig('/Users/David/Didius.pdf')

    return

def read_list(fname,filt='g'):
    mjd,F,dF=[],[],[]

    file=open(fname,'r')
    for line in file:
        if 'MJD' in line and 'PEAKMJD' not in line: mjd=mjd+[float(line.split(' ')[8])]
        if 'OBSFILT' in line:
            if 'OBSFILT:  %s'%filt in line:
                F=F+[float(filter(None,line.split(' '))[7])]
                try: dF=dF+[float(filter(None,line.split(' '))[9])]
                except: dF=dF+[float(filter(None,line.split(' '))[8][2:])]

    return(array(mjd),array(F),array(dF))

def fill_between(ax, x, y1, y2, **kwargs):
   from matplotlib.patches import Polygon
   verts = zip(x,y1) + [(x[i], y2[i]) for i in range(len(x)-1,-1,-1)]
   poly = Polygon(verts, **kwargs)
   ax.add_patch(poly)
   ax.autoscale_view()
   return poly
