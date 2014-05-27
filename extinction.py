# 2013.04.15
# S.Rodney

from math import pi, sqrt
import numpy as np

sqrt2pi = sqrt(2*pi)

def meanAv( ):
    from scipy.integrate import quad

    avhighIa = lambda av : av * highIa( av )
    int1 = quad( avhighIa, 0, 7 )
    int2 = quad( highIa, 0, 7 )
    meanAvhighIa = int1[0] / int2[0]
    print("High Ia (Neill+ 2006) mean Av = %.3f"%( meanAvhighIa ) )

    avhighCC = lambda av : av * highCC( av )
    int3 = quad( avhighCC, 0, 7 ) 
    int4 = quad( highCC, 0, 7 )
    meanAvhighCC = int3[0] / int4[0]
    print("High CC (R+P:2005) mean Av = %.3f"%meanAvhighCC)

    avmidIa = lambda av : av * midIa( av )
    int1 = quad( avmidIa, 0, 7 )
    int2 = quad( midIa, 0, 7 )
    meanAvmidIa = int1[0] / int2[0]
    print("Mid Ia (Kessler+ 2009) mean Av = %.3f"%( meanAvmidIa ) )

    avmidCC = lambda av : av * midCC( av )
    int3 = quad( avmidCC, 0, 7 ) 
    int4 = quad( midCC, 0, 7 )
    meanAvmidCC = int3[0] / int4[0]
    print("Mid CC  mean Av = %.3f"%meanAvmidCC)

    avlowIa = lambda av : av * lowIa( av )
    int1 = quad( avlowIa, 0, 7 )
    int2 = quad( lowIa, 0, 7 )
    meanAvlowIa = int1[0] / int2[0]
    print("Low Ia  mean Av = %.3f"%( meanAvlowIa ) )

    avlowCC = lambda av : av * lowCC( av )
    int3 = quad( avlowCC, 0, 7 ) 
    int4 = quad( lowCC, 0, 7 )
    meanAvlowCC = int3[0] / int4[0]
    print("Low CC  mean Av = %.3f"%meanAvlowCC)

    return( [meanAvhighIa,meanAvmidIa,meanAvlowIa],  [meanAvhighCC,meanAvmidCC,meanAvlowCC] )


# extinction models 

def SNANAdust( Av, sigma=0, tau=0, R0=0, noNegativeAv=True ):
    if not np.iterable( Av ) : Av = np.array( [Av] )

    # gaussian core 
    core = lambda sigma,av : np.exp( -av**2 / (2*sigma**2) )
    # Exponential tail 
    tail = lambda tau,av : np.exp( -av/tau )

    if tau!=0 and noNegativeAv: 
        tailOut = np.where( Av>=0, tail(tau,Av), 0 )
    elif tau!=0 : 
        tailOut = tail(tau,Av)
    else : 
        tailOut = np.zeros( len( Av ) )

    if sigma!=0 and noNegativeAv: 
        coreOut = np.where( Av>=0, core(sigma,Av), 0 )
    elif sigma!=0 : 
        coreOut = core(sigma,Av)
    else : 
        coreOut = np.zeros( len( Av ) )

    if len(Av) == 1 : 
        coreOut = coreOut[0]
        tailOut = tailOut[0]
    if sigma==0 : return( tailOut )
    elif tau==0 : return( coreOut )
    else : return( R0 * coreOut + tailOut )


def N06( Av ):
    """ the Neill et al 2006 baseline model : 
    a gaussian that nods to the Riello & Patat distributions"""
    return( SNANAdust( Av, sigma=0.62 ) )

def K09( Av ):
    """ From Kessler+ 2009 (pure expontential) """
    return( SNANAdust( Av, tau=0.33) )

def highIa( Av ) : 
    """ The High Dust model for SNIa """
    return( N06(Av) )

def midIa( Av ) : 
    """ the Mid Dust model for CCSNe """
    return( K09(Av) )

def lowIa( Av ) : 
    """ The Low Dust model for SNIa """
    return( SNANAdust( Av, sigma=0.15, tau=0.15, R0=1 ) )


def highIa_c( c ) : 
    """ the c distribution for the high Dust model for SNIa """
    from hstsnpipe.tools.snana import bifgauss
    return( bifgauss( c, 0, 0.08, 0.55) )

def midIa_c( c ) : 
    """ the c distribution for the Mid Dust model for SNIa """
    from hstsnpipe.tools.snana import bifgauss
    # from scipy.interpolate import interp1d
    gauss = lambda x,mu,sig : ( 1/np.sqrt(2*np.pi*sig**2) ) * np.exp(-(mu-x)**2/(2*sig**2))
    return( np.where( c<=-0.05, gauss(c,-0.05,0.05)/gauss(-0.05,-0.05,0.05), K09(c+0.05) ))

def lowIa_c( c ) : 
    """ the c distribution for the Low Dust model for SNIa """
    from hstsnpipe.tools.snana import bifgauss
    return( bifgauss( c, -0.05, 0.04, 0.12) )



def RP05( Av, tau=1. ) :
    """ The Riello & Patat 2005 model, as implemented in Dahlen+ 2008:
    sharp cusp, gaussian core, exponential tail
    """
    # Cuspy center
    sigmaA = 0.02
    A = 2.5 / (sqrt2pi*sigmaA)
    cusp = A * np.exp( -Av**2 / (2*sigmaA**2) )
    
    # gaussian "core" dominates out to Av~2
    sigmaB = 0.4
    B = 0.8 / (sqrt2pi*sigmaA)
    core = B * np.exp( -Av**2 / (2*sigmaB**2) )

    # Exponential tail 
    tail =  10*np.exp( -Av/tau )

    return( cusp + core + tail )

def WV07( Av, A=1, B=0.5, tau=0.4, sigma=0.1 ) :
    """ the Galactic Line-of-sight 'glos' prior from Wood-Vasey+ 2007 """
    return( (A/tau) * np.exp(-Av/tau) + (2*B/(sqrt2pi*sigma))*np.exp(-Av**2/(2*sigma**2)) )


def RP05CC( Av ):
    """ Riello+Patat 2005 distribution, as applied by 
    Tomas for the Dahlen+ 2012 CC rates and modified 
    by Steve for SNANA implementation """
    return( SNANAdust( Av, tau=1.7, sigma=0.6, R0=4 ) )


def highCC( Av ) :
    """ the High Dust model for CCSNe """
    #return( RP05CC( Av ) ) 
    return( SNANAdust( Av, tau=2.8, sigma=0.8, R0=3 ) )

def midCC( Av ) :                
    """ the Mid Dust model for CCSNe """
    return( SNANAdust( Av, tau=1.7, sigma=0.6, R0=4 ) )

def lowCC( Av ) :                
    """ the Low Dust model for CCSNe """
    return( SNANAdust( Av, tau=0.5, sigma=0.15, R0=1 ) )


# Dictionary of Av models (both CC and Ia) keyed by 
# the function name
AvModelDict = {
    'SNANAdust':SNANAdust,
    'N06':N06,'K09':K09,
    'highIa':highIa,'midIa':midIa,'lowIa':lowIa,
    'RP05':RP05,'WV07':WV07,'RP05CC':RP05CC,
    'highCC':highCC,'midCC':midCC,'lowCC':lowCC }


def plotCCAv( datfile = 'extinction_100000_RP1.dat'):
    """ 
    The datfile is from Tomas Dahlen, giving the 
    100,000 random Av values that he generated for the 
    Dahlen+ 2012 CCSN rates paper.  This is a set of Av
    values designed for the CCSN population, following 
    Riello+Patat 2005.  
    Plot a histogram and fit an exponential function to it.
    """
    from matplotlib import pyplot as pl
    import sys,os
    
    thisfile = sys.argv[0]
    if 'ipython' in thisfile : thisfile = __file__
    thispath = os.path.abspath( os.path.dirname( thisfile ) )
    datfile = os.path.join( thispath, datfile )

    avlist, inclist = np.loadtxt( datfile, unpack=True ) 
    histvals,binedges = np.histogram( avlist, bins=60, range=[0,7.5] )
    histvals = histvals/float(histvals.max())
    loghist = np.log10( histvals )
    loghist /= float(loghist.max())
    #pl.bar( binedges[:-1], loghist, width=binedges[1]-binedges[0], 
    #pl.bar( binedges[:-1], histvals,width=binedges[1]-binedges[0], 
    #        alpha=0.5, color='b' )
    pl.plot( binedges[:-1], histvals,drawstyle='steps-mid', color='0.4', lw=1.5 )

    # gaussian "core" dominates out to Av~2
    core = lambda sigma,Av : np.exp( -Av**2 / (2*sigma**2) )
    # Exponential tail 
    tail = lambda tau,Av : np.exp( -Av/tau )

    av = np.arange( 0, 7.5, 0.05 )

    sigmaA, R0A, tauA = 0.6, 4, 1.7
    dndav = R0A * core(sigmaA,av) + tail(tauA,av)
    dndav /= dndav.max() * 8
    pl.plot( av, dndav, color='m', marker=' ', lw=2, ls='-' )

    sigmaB, R0B, tauB = 0.1, 10, 1.
    dndav = R0B * core(sigmaB,av) + tail(tauB,av)
    dndav /= dndav.max() 
    pl.plot( av, dndav, color='b', marker=' ', lw=2, ls='-' )

    pl.plot( av, 0.3*N06(av), color='r', ls='-.', lw=2)

    ax = pl.gca()
    ax.set_xlabel('CCSN Host Galaxy A$_V$')
    ax.set_ylabel('dN/dAv')
    ax.text( 6.8, 0.5, 'Fitting the Dahlen+ 2012 CCSN Av distribution with an \n empirical curve, using the SNANA dust model form :',
             color='k', ha='right', va='bottom', fontsize='large' )
    ax.text( 6.8, 0.4, 'dN/dAv = %i * gauss(sigma=%.1f) + exp(tau=%.1f)'%(R0A,sigmaA,tauA), 
             color='m', ha='right', va='top', fontsize='large' )

    ax.text( 6.8, 0.2, 'dN/dAv = %i * gauss(sigma=%.1f) + exp(tau=%.1f)'%(R0B,sigmaB,tauB), 
             color='b', ha='right', va='top', fontsize='large' )

    ax.text( 2.5, 5e-4, "'High' dust model for SNIa\n (Neill+ 2006)", ha='left',va='top',color='r')

    ax.semilogy()
    ax.set_xlim( 0, 7 )
    ax.set_ylim( 1e-4, 2 )
        



def plotAvpriors( Avstep = 0.01):
    from matplotlib import pyplot as pl
    Av = np.arange( 0, 7, Avstep ) 

    meanIa, meanCC = meanAv()

    # normalize each so that the integral is unity
    highIaAv = highIa( Av )
    highIaAv /= highIaAv.sum() * Avstep

    highCCAv = highCC( Av )
    highCCAv /= highCCAv.sum() * Avstep

    midIaAv = midIa( Av )
    midIaAv /= midIaAv.sum() * Avstep

    midCCAv = midCC( Av )
    midCCAv /= midCCAv.sum() * Avstep

    lowIaAv = lowIa( Av )
    lowIaAv /= lowIaAv.sum() * Avstep

    lowCCAv = lowCC( Av )
    lowCCAv /= lowCCAv.sum() * Avstep


    RP05Av = RP05( Av )
    RP05Av /= RP05Av.sum() * Avstep

    #WV07Av = WV07( Av )
    #WV07Av /= WV07Av.sum() * Avstep
    ax1 = pl.subplot( 121 )
    ax1.plot( Av, highIaAv, 'r-', label=r'High Ia; $\langle A_V \rangle$=%.1f'%meanIa[0])
    ax1.plot( Av, midIaAv, 'g--', label=r'Mid Ia; $\langle A_V \rangle$=%.1f'%meanIa[1])
    ax1.plot( Av, lowIaAv, 'b:', label=r'Low Ia; $\langle A_V \rangle$=%.1f'%meanIa[2])

    ax2 = pl.subplot( 122, sharex=ax1, sharey=ax1 )
    ax2.plot( Av, highCCAv, 'r-', label=r'High CC; $\langle A_V \rangle$=%.1f'%meanCC[0])
    ax2.plot( Av, midCCAv, 'g--', label=r'Mid CC; $\langle A_V \rangle$=%.1f'%meanCC[1])
    ax2.plot( Av, lowCCAv, 'b:', label=r'Low CC; $\langle A_V \rangle$=%.1f'%meanCC[2])

    ax1.legend( loc='upper right', frameon=False, handlelength=1.5 )
    ax2.legend( loc='upper right', frameon=False, handlelength=1.5 )
    ax1.set_xlabel( 'A$_V$')
    ax2.set_xlabel( 'A$_V$')
    ax1.set_ylabel( 'dN/dA$_V$')
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks_position('both')
    ax1.semilogy()
    ax2.semilogy()
    ax1.set_ylim( 1e-4, 12 )
    ax1.set_xlim( -0.05, 7.1 )

    fig = pl.gcf()
    fig.subplots_adjust( wspace=0, left=0.12, right=0.9, top=0.95, bottom=0.12)

def plotCcurves():
    """ plot the distribution of SALT2 C parameter values from SNLS
    and overlay the lines that match the p(Av) dust models
    """
    import os
    import sys
    from matplotlib import pyplot as pl
    from hstsnpipe import tools
    from hstsnpipe.tools import snana
    from hstsnpipe.tools.snana import bifgauss

    thisfile = sys.argv[0]
    if 'ipython' in thisfile : thisfile = __file__
    thispath = os.path.abspath( os.path.dirname( thisfile ) )
    snlsdatfile = os.path.join( thispath, "snls.dat" )
    col,ecol = np.loadtxt( snlsdatfile, usecols=[7,8], unpack=True )


    cbins,cedges = np.histogram(col, bins=20 )
    bars = pl.bar( cedges[:-1], 6.1*cbins/float(cbins.max()), width=cedges[1]-cedges[0], alpha=0.5, color='b' )
    
    c = np.arange( -0.4, 1.2, 0.01 )
    pcdefault = 6*snana.bifgauss( c, 0, 0.08, 0.14 )
    pl.plot( c, pcdefault, 'k-', label='SNANA default' )

    # High dust : (~Model C = Neill 2006)
    pchigh = snana.bifgauss( c, 0, 0.08, 0.38 )
    pl.plot( c, 6*pchigh, 'r--', label=r'High Dust ($\sim$Neill+2006)' )

    # Middle dust (~Kessler 2009)
    pcmid = snana.bifgauss( c, 0, 0.08, 0.25 )
    pl.plot( c, 6*pcmid, 'g--', label=r'Mid Dust ($\sim$Kessler+2009)' )

    # low dust : (~Barbary12 minimal dust model)
    pclow = snana.bifgauss( c, 0, 0.08, 0.1 )
    pl.plot( c, 6*pclow, 'b--', label=r'Low Dust (Barbary+2012)'  )

    pl.grid()
    
    ax = pl.gca()
    ax.set_xlabel(r'SALT2 Color $\mathscr{C}$')
    ax.set_ylabel(r'P($\mathscr{C}$)')

    ax.legend( loc='upper right', frameon=False, handlelength=2.5, handletextpad=0.5)
    ax.text( -0.36, 3.5, "SNLS3\nCompilation", ha='left',va='top',color='b')

    pl.draw()



def convolveCAv(  AvModel='midIa', cmin=-0.3, cmax=2.5 ):
    """ convolve a narrow gaussian (the intrinsic SNIa color
    distribution) with an exponential (or gauss+exponential) Av
    distribution to get the resulting observed distribution 
    of SALT2 colors

    The input parameter Avmodel can be either a function that takes Av
    as its single input parameter, or a string, corresponding to the
    name of an Av model function (e.g. 'midIa' or 'highCC')

    Note that we assume the SALT2 beta = 4.1, corresponding to Rv=3.1,
    a.k.a. milky way dust (Scolnic+ 2013).

    Returns two arrays, the input distribution of C values
    and the convolved probability at each value of C
    """
    if isinstance(AvModel, basestring) : 
        AvModel = AvModelDict[ AvModel ]

    gauss = lambda x,mu,sig : ( 1/np.sqrt(2*np.pi*sig**2) ) * np.exp(-(mu-x)**2/(2*sig**2))

    # Define the range of allowable C values
    #  - this can go slightly negative, for very blue SNIa with 0 dust
    Cin = np.arange( cmin, cmax, 0.01 )


    # Define the intrinsic probability distribution of C values 
    #  - narrow gaussian 
    #  - centered on c=-0.1 (corresponding to Av=0, Kessler:2009a)
    #  - sigma=0.04  (Scolnic:2013, section 3.1)
    Cdist = gauss( Cin, -0.1, 0.04 )

    # Add 0.1 to C to convert the C grid to Av  (assuming Beta=4.1), 
    # then use the user-specified Av distribution model to define the 
    # distribution of host galaxy extinction values
    # Note: we limit this to positive Av values so that the 
    # numpy 1-d convolution operation produces an output array 
    # that is appropriately shifted to match the Cin abscissa array
    Avin = ( Cin+0.1)[ np.where( Cin>=-0.1 ) ]
    hostAvdist = AvModel( Avin ) 

    # convolve the two probability distributions, then normalize
    # so that the resulting distribution integrates to unity
    Cobs = np.convolve( Cdist, hostAvdist, mode='full' )
    Cobs = ( Cobs / ( np.sum( Cobs ) * np.diff( Cin )[0] ))[:len(Cin)]

    return( Cin, Cobs )


