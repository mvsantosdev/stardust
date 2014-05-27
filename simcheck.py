# 2013.04.26
# S.Rodney

# checking if the simulated SN distributions are in line with the observed 
# mags and colors for 0.5 < z < 1.0

# TODO:  handle upper limits 
# TODO : handle missing nicknames
# TODO : adjust default x scaling

import os
import sys
from hstsnpipe.tools import snana
import numpy as np
from matplotlib import pyplot as pl
from matplotlib import patches

sndataroot = os.environ['SNDATA_ROOT']

DATFILELIST = [
'HST_CANDELS1_adams.dat',
'HST_CANDELS1_agnew.dat',
'HST_CANDELS1_aidan.dat',
'HST_CANDELS1_benjamin.dat',
'HST_CANDELS1_buchanan.dat',
'HST_CANDELS1_bush.dat',
'HST_CANDELS1_carter.dat',
'HST_CANDELS1_cleveland.dat',
'HST_CANDELS1_clinton.dat',
'HST_CANDELS1_eisenhower.dat',
'HST_CANDELS1_fdr.dat',
'HST_CANDELS1_ford.dat',
'HST_CANDELS1_garfield.dat',
'HST_CANDELS1_grant.dat',
'HST_CANDELS1_harrison.dat',
'HST_CANDELS1_hayes.dat',
'HST_CANDELS1_herbert.dat',
'HST_CANDELS1_hoover.dat',
'HST_CANDELS1_humphrey.dat',
'HST_CANDELS1_jackson.dat',
'HST_CANDELS1_jefferson.dat',
'HST_CANDELS1_johnson.dat',
'HST_CANDELS1_kennedy.dat',
'HST_CANDELS1_lbj.dat',
'HST_CANDELS1_lincoln.dat',
'HST_CANDELS1_madison.dat',
'HST_CANDELS1_mckinley.dat',
'HST_CANDELS1_mikulski.dat',
'HST_CANDELS1_mondale.dat',
'HST_CANDELS1_pierce.dat',
'HST_CANDELS1_polk.dat',
'HST_CANDELS1_primo.dat',
'HST_CANDELS1_quayle.dat',
'HST_CANDELS1_quincy.dat',
'HST_CANDELS1_reagan.dat',
'HST_CANDELS1_rockefeller.dat',
'HST_CANDELS1_roosevelt.dat',
'HST_CANDELS1_taylor.dat',
'HST_CANDELS1_truman.dat',
'HST_CANDELS1_tumbleweed.dat',
'HST_CANDELS1_vanburen.dat',
'HST_CANDELS1_washington.dat',
'HST_CANDELS1_wilson.dat',
'HST_CANDELS1_workman.dat',
]


def colorcheck_midz1(): 
    datfilelist1 = [
'HST_CANDELS1_taylor.dat',
'HST_CANDELS1_pierce.dat',
'HST_CANDELS1_ford.dat',
'HST_CANDELS1_eisenhower.dat',
'HST_CANDELS1_garfield.dat',
]
    fig = pl.figure( 1, figsize=(19,12) ) 
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    pl.clf()
    fig = pl.figure( 2, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    for irow, datfile in zip( range(5), datfilelist1) : 
        colorCheck( datfile, 5, irow, [1,2] )

def colorcheck_midz2(): 
    datfilelist2 = [
'HST_CANDELS1_workman.dat',
'HST_CANDELS1_roosevelt.dat',
'HST_CANDELS1_jackson.dat',
'HST_CANDELS1_buchanan.dat',
'HST_CANDELS1_reagan.dat',
]
    fig = pl.figure( 3, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    fig = pl.figure( 4, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    for irow, datfile in zip( range(5), datfilelist2) : 
        colorCheck( datfile, 5, irow, [3,4] )

def colorcheck_midz3(): 
    datfilelist3 = [
'HST_CANDELS1_harrison.dat',
'HST_CANDELS1_fdr.dat',
'HST_CANDELS1_aidan.dat',
'HST_CANDELS1_adams.dat',
'HST_CANDELS1_vanburen.dat',
]
    fig = pl.figure( 5, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    fig = pl.figure( 6, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    for irow, datfile in zip( range(5), datfilelist3) : 
        colorCheck( datfile, 5, irow, [5,6] )

def colorcheck_midz4(): 
    datfilelist4 = [
'HST_CANDELS1_mondale.dat',
'HST_CANDELS1_lbj.dat',
'HST_CANDELS1_lincoln.dat',
'HST_CANDELS1_mikulski.dat',
'HST_CANDELS1_madison.dat',
]
    fig = pl.figure( 7, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    fig = pl.figure( 8, figsize=(19,12) ) 
    pl.clf()
    fig.subplots_adjust( left=0.05, bottom=0.04, right=0.96, top=0.93, wspace=0.0, hspace=0.20 )
    for irow, datfile in zip( range(5), datfilelist4) : 
        colorCheck( datfile, 5, irow, [7,8] )


def colorCheck(datfile, nrow, irow, ifiglist=[1,2], clobber=False, verbose=1):

    sn = snana.SuperNova(datfile )
    sn.getClassSim( 'HST_colormag', Nsim=2000, dustmodel='mid', simpriors=True, clobber=clobber, verbose=verbose ) 

    pkbands = np.unique([ sn.FLT[i] for i in range(len(sn.MJD)) if abs(sn.MJD[i]-sn.pkmjdobs)<=sn.pkmjdobserr ])
    sn.ClassSim.Ia.samplephot( sn.pkmjdobs, tmatch=sn.pkmjdobserr, bandlist=pkbands )
    sn.ClassSim.Ibc.samplephot( sn.pkmjdobs, tmatch=sn.pkmjdobserr, bandlist=pkbands )
    sn.ClassSim.II.samplephot( sn.pkmjdobs, tmatch=sn.pkmjdobserr, bandlist=pkbands )

    ipk = np.where( np.abs(sn.MJD - sn.pkmjdobs)< sn.pkmjdobserr )[0]

    for ifig,redfilt in zip(ifiglist,['H','J']) :
        if redfilt not in pkbands : continue

        fig = pl.figure( ifig ) 
        ax1 = fig.add_subplot( nrow, 4, 1 )
        
        RpkSimIa = sn.ClassSim.Ia.__dict__['%s%i'%(redfilt,int(sn.pkmjdobs))]
        RpkSimIbc = sn.ClassSim.Ibc.__dict__['%s%i'%(redfilt,int(sn.pkmjdobs))]
        RpkSimII = sn.ClassSim.II.__dict__['%s%i'%(redfilt,int(sn.pkmjdobs))]
        
        ipkR =  np.where(  sn.FLT[ipk] == redfilt )[0]
        if not len(ipkR) : continue
        snR = sn.MAG[ipk][ipkR][0]
        snRerr = sn.MAGERR[ipk][ipkR][0]

        for icol,bluefilt in zip( range(4),['W','V','I','Z']):
            ax = fig.add_subplot( nrow, 4, irow*4 + icol + 1, sharex=ax1 )
            if icol == 0 : ax.set_ylabel(sn.nickname)
            if irow == 0 : ax.set_title( '%s-%s'%(bluefilt,redfilt) )
            if bluefilt not in pkbands : continue

            ipkB =  np.where(  sn.FLT[ipk] == bluefilt )[0]
            if not len(ipkB) : continue
            snB = sn.MAG[ipk][ipkB][0]
            snBerr = sn.MAGERR[ipk][ipkB][0]

            BpkSimIa = sn.ClassSim.Ia.__dict__['%s%i'%(bluefilt,int(sn.pkmjdobs))]
            BpkSimIbc = sn.ClassSim.Ibc.__dict__['%s%i'%(bluefilt,int(sn.pkmjdobs))]
            BpkSimII = sn.ClassSim.II.__dict__['%s%i'%(bluefilt,int(sn.pkmjdobs))]

            CpkSimIa  = BpkSimIa  - RpkSimIa
            CpkSimIbc = BpkSimIbc - RpkSimIbc
            CpkSimII  = BpkSimII  - RpkSimII

            CIa,cbins  = np.histogram( CpkSimIa,  bins=np.arange(-5,12,0.2) )
            CIbc,cbins = np.histogram( CpkSimIbc, bins=np.arange(-5,12,0.2) )
            CII,cbins  = np.histogram( CpkSimII,  bins=np.arange(-5,12,0.2) )

            ax.plot( cbins[:-1], CIa, 'r-', drawstyle='steps-mid' )
            ax.plot( cbins[:-1], CIbc, 'g-', drawstyle='steps-mid' )
            ax.plot( cbins[:-1], CII, 'b-', drawstyle='steps-mid' )

            snC = snB - snR
            snCerr = np.sqrt( snBerr**2 + snRerr**2 )
            snCmin = snC - snCerr
            snCmax = snC + snCerr
            if snBerr<0 : snCmin = snC
            if snRerr<0 : snCmax = snC

            ymin,ymax=ax.get_ylim()
            snCbar = patches.Rectangle( [ snCmin, 0.0], snCmax-snCmin, ymax, color='0.5', alpha=0.5, zorder=-100 )
            ax.add_patch( snCbar )
            ax.set_xlim([-2,6])
        fig.suptitle( '(W,V,I,Z)-%s band color distributions'%redfilt )



def sedcheck( z=0.68, days=[-10,0,20], cctype='Ibc'):
    """ plot the rest-frame SED for Ia , Ib/c and II SNe, 
    overlaying the braod-band filter curves (blue-shifted)
    to see the origin of the color distributions.
    """
    fig = pl.figure(1,figsize=(19,12))
    pl.clf()

    ncol = len(days)
    for icol,day in zip(range(ncol),days) : 
        # reagan: z=0.68, day=-10
        # buchanan: z=0.68, day=+20

        fig.add_subplot( 1,ncol,icol+1)
        plotsed( sedfile='Hsiao07.extrap.dat', day=day, z=0.68, color='r',ls='-',lw=2 )
        if cctype=='II': plotIIseds( day=day,z=0.68)
        elif cctype=='Ibc': plotIbcseds( day=day,z=0.68)
        elif cctype in ['CC','all']: 
            plotIIseds( day=day,z=0.68)
            plotIbcseds( day=day,z=0.68)
        plotbands( 'WVIJ', z=z )
        ax = pl.gca()
        ax.set_xlim(1800,10000)
        ax.text( 0.95, 0.95, 't = %i'%int(day), transform=ax.transAxes, ha='right',va='top',fontsize='x-large' )
        

def plotIIseds( day=0, z=0.68 ):
    plotsed( sedfile='SDSS-000018.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-003818.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-013376.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-014450.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-014599.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-015031.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-015320.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-015339.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-017564.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-017862.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018109.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018297.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018408.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018441.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018457.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018590.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018596.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018700.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018713.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018734.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018793.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018834.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-018892.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-020038.SED', day=day, z=z, color='b',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-012842.SED', day=day, z=z, color='c',ls='-',lw=1 )
    plotsed( sedfile='SDSS-013449.SED', day=day, z=z, color='c',ls='-',lw=1 )
    plotsed( sedfile='Nugent+Scolnic_IIL.SED', day=day, z=z, color='m',ls='-',lw=1 )


def plotIbcseds( day=0, z=0.68 ):
    plotsed( sedfile='CSP-2004gv.SED',  day=day, z=z, color='g',ls='-',lw=0.5 )
    plotsed( sedfile='CSP-2006ep.SED',  day=day, z=z, color='g',ls='-',lw=0.5 )
    plotsed( sedfile='CSP-2007Y.SED',   day=day, z=z, color='g',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-000020.SED', day=day, z=z, color='g',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-002744.SED', day=day, z=z, color='g',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-014492.SED', day=day, z=z, color='g',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-019323.SED', day=day, z=z, color='g',ls='-',lw=0.5 )

    plotsed( sedfile='SNLS-04D4jv.SED', day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='CSP-2004fe.SED',  day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='CSP-2004gq.SED',  day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-004012.SED', day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-013195.SED', day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-014475.SED', day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-015475.SED', day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='SDSS-017548.SED', day=day, z=z, color='c',ls='-',lw=0.5 )
    plotsed( sedfile='SNLS-04D1la.SED', day=day, z=z, color='c',ls='-',lw=0.5 )


def plotsed( sedfile = 'Hsiao07.extrap.dat', day=0, z=0.68, **kwarg ):
    """ plot the sed, normalizing such that the integrated J band = 1 """
    from scipy import interp

    w,f = getsed( sedfile, day=day )
    wJ,fJ = getbandpass(  'J' )

    fJint = interp( w, wJ/(1+z), fJ, left=0, right=0 )
    dw = np.diff(w).mean()
    normfactor = (fJint * f).sum() * dw
   
    ax1 = pl.gca()
    fmag = -2.5*np.log10( np.where( f>0, f/normfactor, np.ones(len(f))*1e-6) ) + 25
    ax1.plot( w, fmag, zorder=20, **kwarg )
    ax1.set_xlim(1800,10000)
    #ax1.set_yticks([])
    ax1.set_ylim(38,27)
    ax1.set_ylabel('-2.5 log( f ) + constant' )
    ax1.set_xlabel('rest wavelength' )

def plotbands( bands='WVIJ', z=0.68 ):

    wJ,fJ = getbandpass(  'J' )
    wW,fW = getbandpass(  'W' )
    wV,fV = getbandpass(  'V' )
    wI,fI = getbandpass(  'I' )
    
    ax1 = pl.gca()
    ax2 = ax1.twinx()
    if 'J' in bands : ax2.fill_between( wJ/(1+z), fJ, color='r', zorder=-20, alpha=0.3 )
    if 'V' in bands : ax2.fill_between( wV/(1+z), fV, color='b', zorder=-20, alpha=0.3 )
    if 'I' in bands : ax2.fill_between( wI/(1+z), fI, color='g', zorder=-20, alpha=0.3 )
    if 'W' in bands : ax2.fill_between( wW/(1+z), fW, color='k', zorder=-40, alpha=0.3 )
    ax2.set_ylim(0,0.3)
    ax2.set_yticks([])

def getbandpass( band='J' ):
    srcdir = sys.argv[0]
    if srcdir.endswith('python'): srcdir = __file__
    filtdir = os.path.abspath( os.path.dirname( srcdir ) +'/../figs/FILTER' )

    if band=='V': 
        return( np.loadtxt( os.path.join(filtdir,'ACS_WFC_F606W.dat'), unpack=True ) )
    elif band=='I':
        return( np.loadtxt( os.path.join(filtdir,'ACS_WFC_F814W.dat'), unpack=True ) )
    elif band=='Z':
        return( np.loadtxt( os.path.join(filtdir,'ACS_WFC_F850LP.dat'), unpack=True ) )
    elif band=='W':
        return( np.loadtxt( os.path.join(filtdir,'WFC3_UVIS_F350LP.dat'), unpack=True ) )
    elif band=='J':
        return( np.loadtxt( os.path.join(filtdir,'WFC3_IR_F125W.dat'), unpack=True ) )
    elif band=='H':
        return( np.loadtxt( os.path.join(filtdir,'WFC3_IR_F160W.dat'), unpack=True ) )


def getsed( sedfile = 'Hsiao07.extrap.dat', day=None) : 
    if not os.path.isfile( sedfile ) : 
        sedfile = os.path.join( sndataroot, 'snsed/%s'%sedfile) 
    if not os.path.isfile( sedfile ) : 
        sedfile = os.path.join( sndataroot, 'snsed/non1a/%s'%os.path.basename(sedfile) )
    if not os.path.isfile( sedfile ) : 
        print("cannot find %s"%os.path.basename(sedfile) ) 

    d,w,f = np.loadtxt( sedfile, unpack=True ) 

    if day!=None : 
        dout = d[ np.where( np.abs(d-day)<0.9 ) ]
        wout = w[ np.where( np.abs(d-day)<0.9 ) ]
        fout = f[ np.where( np.abs(d-day)<0.9 ) ]
        return( wout, fout )
    else : 
        days = unique( d ) 
        dout = dict( [ [day, d[ np.where( d==day ) ]] for day in days ] )
        wout = dict( [ [day, w[ np.where( d==day ) ]] for day in days ] )
        fout = dict( [ [day, f[ np.where( d==day ) ]] for day in days ] )
        return( dout, wout, fout )
