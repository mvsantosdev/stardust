#! /usr/bin/env python
# 2012.04.16 S.Rodney
#  Set up, run, read, and plot a SNANA simulation 
#  to build a comparison sample for SN classification


import pyfits
import os 
import exceptions
import numpy as np
import time
import glob
from collections import Sequence

if 'SNANA_DIR' in os.environ.keys() : 
    SNANA_DIR = os.environ['SNANA_DIR']
    SNDATA_ROOT = os.environ['SNDATA_ROOT']
else : 
    SNANA_DIR = os.path.abspath('.')
    SNDATA_ROOT = os.path.abspath('.')

class ClassSim( Sequence ) :
    """ simple container for holding the output of 
    a three-part SNANA classification simulation: 
       Ia :  the SNIa simulation
       Ibc : the SNIb/c simulation
       II  : the SNII simulation
    """
    def __init__( self, Ia, Ibc, II) : 
        self.Ia = Ia
        self.Ibc = Ibc
        self.II = II
    
    def __items__(self ):
        return( [self.Ia,self.Ibc,self.II] )

    def __getitem__( self, index ) : 
        return( self.__items__()[index] )

    def __len__( self ) :
        return( len( self.__items__() ) )

    @property
    def NSIMTOT( self ):
        """ Total number of simulated SNe"""
        return( self.Ia.nsim + self.II.nsim + self.Ibc.nsim )



def doClassSim( **kwarg ): 
    return( doMonteCarloSim( **kwarg ) )

def doMonteCarloSim( simroot='HST_classify', survey='HST',field='default',
                     Nsim=1000, zrange=[0.4,2.3], avrange=[0,7],
                     x1range=[-3.0,3.0], crange=[-0.4,1.0], pkmjdrange=[],  
                     bands='VXIZWJH', etimes=[],
                     mjd0 = 55500.0, Nepoch = 100, cadence=3,
                     mjdlist = [], bandlist=[], etimelist=[], 
                     ratemodel='flat', dustmodel='mid',
                     simpriors=True, perfect=True, 
                     clobber=False, verbose=False  ):
    """ master function to run the CC and the Ia simulations
    to produce a comparison sample for classification of an observed SN.

    OPTIONS : 
    Nsim= total number of simulated SNe in each of three classes (Ia, Ibc, II)
    ratemodel : the form of the SNR(z) model ['constant','powerlaw','flat']
        'constant' = constant volumetric rate with redshift
        'powerlaw' = double power law in z. The powerlaw 
           parameters are automatically adjusted to ~match
           the observed SNR(z) according to the simtype. 
         'flat' = dN/dz is flat 
    dustmodel : choose the p(Av) or p(c) dust model from ['high','mid','low','flat']
    simpriors : include realistic priors in the simulation, 
       i.e. use the SNANA monte carlo simulator to choose shape and 
       color parameters from apprpriate distributions.
       When False, set the Av, x1, and c distributions to be flat 
         (sigma->infinity) and turn off all mag smearing (appropriate
         for simulations to be used as input to the bayesian classifier)
         (False overrides the ratemodel and dustmodel settings)

    There are two ways to provide the sequence of observations: 
    
    A) regularly spaced, identical epochs
       The obs dates are specified with  mjd0, cadence and Nepoch.
       The filter set (same in each epoch) is given as a string in bands. 
       Exposure times (same for each epoch) are in etimes

    B) heterogeneous epochs, explicitly defined
       each observation has a single entry in each of mjdlist, bandlist, etimelist
       (so all three lists must be the same length)

    Option A is defined with generic default values, but if  mjdlist, bandlist 
    (and etimelist) are provided, then option B supercedes option A. 

    RETURNS : 
     the root name of the resulting SimTable set (use rdClassSim to read in results)
    """
    import simulate

    if not len(pkmjdrange):
        # all simulated SNe peak exactly midway through the survey:
        if mjdlist : 
            pkmjdrange = [mjdlist[int(len(mjdlist)/2)],mjdlist[int(len(mjdlist)/2)]]
        else : 
            pkmjdrange = [mjd0+cadence*(Nepoch/2),mjd0+cadence*(Nepoch/2)]

    # The TREST range must be large enough to encompass all of the observation
    # dates in mjdlist (for the minimum redshift), or else SNANA will skip over 
    # those dates that lie outside of TREST_RANGE and we will end up with 
    # simulated SN light curves of different sizes
    if len(mjdlist): 
        mjdobsmin = np.min(mjdlist)
        mjdobsmax = np.max(mjdlist)
    else : 
        mjdobsmin = mjd0
        mjdobsmax = mjd0 + Nepoch *cadence 
    zmin = zrange[0]
    trestmin = (mjdobsmin - pkmjdrange[1] -2)/(1+zmin)
    trestmax =  (mjdobsmax - pkmjdrange[0] +2)/(1+zmin)
    trestrange = [trestmin, trestmax]

    if len(bandlist) : bands = ''.join(np.unique( bandlist ))

    if not len(etimes) : etimes = (1000*np.ones(len(bands))).tolist()
    if not len(etimelist) : etimelist = (1000*np.ones(len(mjdlist))).tolist()

    # make the shared .simlib file
    simlibfile = 'sim_%s.simlib'%( simroot )
    if ( ( not os.path.exists( os.path.join( SNDATA_ROOT, 'simlib/'+simlibfile) )
           and not os.path.exists( simlibfile ) ) 
         or clobber ):
        simulate.mksimlib( simlibfile, survey=survey, field=field,
                           bands=bands, etimes = etimes, 
                           mjd0=mjd0, cadence=cadence, Nepoch=Nepoch, 
                           mjdlist=mjdlist, bandlist=bandlist, etimelist=etimelist,
                           perfect=perfect, clobber=clobber  )
    
    # 2012.10.18  SR: now simulating the same number in each class, so that 
    # we can apply a prior on the relative rates after computing likelihoods
    # (e.g. for testing alternate SN rate prescriptions)
    NsimIa,NsimIbc,NsimII = Nsim,Nsim,Nsim

    if simpriors: 
        x1sigma = [1.5,0.9]
        smear = True
    else : 
        x1sigma = [1e12,1e12]
        dustmodel = 'flat'
        ratemodel = 'flat'
        smear = True  # 2013.05.17 SR: now smearing is always on

    # make the .input file for the Ia simulation
    inputIa = simulate.mkinput( simname=simroot+'_Ia', simlibfile=simlibfile, survey=survey, field=field,
                                simtype='Ia', Nsim=NsimIa, ratemodel=ratemodel, dustmodel=dustmodel,
                                applySearchEff=False, applyCutWin=False,
                                smear=smear, clobber=clobber, 
                                GENSOURCE='RANDOM',
                                GENRANGE_SALT2x1=['%.1f %.1f'%(tuple(x1range)),'# x1 (stretch) range'],
                                GENSIGMA_SALT2x1=['%.1f %.1f'%(tuple(x1sigma)),'# bifurcated sigmas'],
                                GENRANGE_SALT2c=['%.1f %.1f'%(tuple(crange)),'# color range'],
                                GENFILTERS= [bands,'# List of filters to simulate'],
                                GENRANGE_PEAKMJD=['%.1f %.1f'%(tuple(pkmjdrange)),'# range of simulated peak MJD dates'],
                                GENRANGE_TREST=['%.1f %.1f'%(tuple(trestrange)),'# rest epoch relative to peak (days)'],
                                GENRANGE_REDSHIFT=['%.3e %.3e'%(tuple(zrange)), '# simulated redshift range'],
                                )

    # make the .input file for the Ibc simulation
    inputIbc = simulate.mkinput( simname=simroot+'_Ibc', simlibfile=simlibfile, survey=survey, field=field,
                                 simtype='Ibc', Nsim=NsimIbc, ratemodel=ratemodel, dustmodel=dustmodel,
                                 applySearchEff=False, applyCutWin=False, 
                                 smear=smear, clobber=clobber, 
                                 GENSOURCE='RANDOM',
                                 GENRANGE_AV=['%.1f %.1f'%(tuple(avrange)),'# CCM89 extinc param range'],
                                 GENFILTERS= [bands,'# List of filters to simulate'],
                                 GENRANGE_PEAKMJD=['%.1f %.1f'%(tuple(pkmjdrange)),'# range of simulated peak MJD dates'],
                                 GENRANGE_TREST=['%.1f %.1f'%(tuple(trestrange)),'# rest epoch relative to peak (days)'],
                                 GENRANGE_REDSHIFT=['%.3f %.3f'%(tuple(zrange)), '# simulated redshift range'],
                                 )
    
    # make the .input file for the II simulation
    inputII = simulate.mkinput( simname=simroot+'_II', simlibfile=simlibfile, survey=survey, field=field, 
                                simtype='II', Nsim=NsimII, ratemodel=ratemodel, dustmodel=dustmodel,
                                applySearchEff=False, applyCutWin=False, 
                                smear=smear, clobber=clobber, 
                                GENSOURCE='RANDOM',
                                GENRANGE_AV=['%.1f %.1f'%(tuple(avrange)),'# CCM89 extinc param range'],
                                GENFILTERS= [bands,'# List of filters to simulate'],
                                GENRANGE_PEAKMJD=['%.1f %.1f'%(tuple(pkmjdrange)),'# range of simulated peak MJD dates'],
                                GENRANGE_TREST=['%.1f %.1f'%(tuple(trestrange)),'# rest epoch relative to peak (days)'],
                                GENRANGE_REDSHIFT=['%.3f %.3f'%(tuple(zrange)), '# simulated redshift range'],
                                )

    # run the simulations 
    simulate.dosim( inputIa, perfect=perfect, verbose=verbose )
    simulate.dosim( inputIbc, perfect=perfect, verbose=verbose )
    simulate.dosim( inputII, perfect=perfect, verbose=verbose )

    return( simroot )


def rdClassSim( simroot, verbose=False ) :
    """ 
    Check for existing simulation results, 
    then read in classification simulation products
    and return a list containing three SimTable
    objects: [Ia,II,Ibc]
    """
    from __init__ import SimTable

    simnameIa = simroot + '_Ia'
    simnameIbc = simroot + '_Ibc'
    simnameII = simroot + '_II'

    # check first for existence in SNDATA_ROOT
    photfileIa = os.path.join( SNDATA_ROOT, 'SIM/'+simnameIa+'/%s_PHOT.FITS'%simnameIa)
    photfileIbc = os.path.join( SNDATA_ROOT, 'SIM/'+simnameIbc+'/%s_PHOT.FITS'%simnameIbc)
    photfileII = os.path.join( SNDATA_ROOT, 'SIM/'+simnameII+'/%s_PHOT.FITS'%simnameII)

    gridfileIa = os.path.join( SNDATA_ROOT, 'SIM/'+simnameIa+'/%s.GRID'%simnameIa)
    gridfileIbc = os.path.join( SNDATA_ROOT, 'SIM/'+simnameIbc+'/%s.GRID'%simnameIbc)
    gridfileII = os.path.join( SNDATA_ROOT, 'SIM/'+simnameII+'/%s.GRID'%simnameII)

    if ( not (os.path.isfile( photfileII ) and 
              os.path.isfile( photfileIbc ) and  
              os.path.isfile( photfileIa ) ) and
         not  (os.path.isfile( gridfileII ) and 
              os.path.isfile( gridfileIbc ) and  
              os.path.isfile( gridfileIa ) ) ) : 
        raise exceptions.IOError("No classification sim products for %s"%simroot)

    # read in the sim products
    simdatIa = SimTable( simnameIa, verbose=verbose )
    simdatIbc = SimTable( simnameIbc, verbose=verbose )
    simdatII = SimTable( simnameII, verbose=verbose )
    return( ClassSim( simdatIa, simdatIbc, simdatII ) )


def plotClassSim( simset, xaxis='W-H', yaxis='H',
                  tobsrange=[0,0], snmags={}, Nbins=50,
                  plotstyle='contourf', verbose=True,
                  plottype='colormag',ccbands='') : 
    """ 
    Plot color mag contours for a classification 
    simulation.  

    Required argument 'simset' must be a list of three 
    SimTable objects [Ia,II,Ibc]  (i.e. as returned 
    by rdClassSim)

    If provided in the 'snmags' dict, observed SN 
    point(s) are overplotted with errorbars.
    """
    import simplot
    
    # plot color mag diagrams
    if plottype=='colormag':
        if verbose : print('plotting color-magnitude contours for Type II SNe')
        simplot.plotColorMag( simset.II, color=xaxis, mag=yaxis,
                              tobsrange=tobsrange, snmags=snmags, 
                              plotstyle=plotstyle, Nbins=Nbins)

        if verbose : print('plotting color-magnitude contours for Type Ibc SNe')
        simplot.plotColorMag( simset.Ibc, color=xaxis, mag=yaxis,
                              tobsrange=tobsrange, snmags=snmags, 
                              plotstyle=plotstyle, Nbins=Nbins)

        if verbose : print('plotting color-magnitude contours for Type Ia SNe')
        simplot.plotColorMag( simset.Ia, color=xaxis, mag=yaxis,
                              tobsrange=tobsrange, snmags=snmags, 
                              plotstyle=plotstyle, Nbins=Nbins)

    if plottype=='colorcolor':
        if verbose : print('plotting color-magnitude contours for Type II SNe')
        simplot.plotColorColor( simset.II, color1=xaxis, color2=yaxis,
                                plotstyle=plotstyle, Nbins=Nbins,
                                bands=ccbands,histcolor='b')

        if verbose : print('plotting color-magnitude contours for Type Ibc SNe')
        simplot.plotColorColor( simset.Ibc, color1=xaxis, color2=yaxis,
                                tobsrange=tobsrange, snmags=snmags, 
                                plotstyle=plotstyle, Nbins=Nbins,
                                bands=ccbands,histcolor='g')
        
        if verbose : print('plotting color-magnitude contours for Type Ia SNe')
        simplot.plotColorColor( simset.Ia, color1=xaxis, color2=yaxis,
                                tobsrange=tobsrange, snmags=snmags, 
                                plotstyle=plotstyle, Nbins=Nbins,
                                bands=ccbands,histcolor='r')


def classfracz( z, h=0.7,classfrac0 = [0.24, 0.19, 0.57]):
    """calculate SN class fractions as a function of redshift using
    Li et al., Hopkins & Beacom 2006, and a DTD power law"""
    
    R0_1 = 5.0e-5
    R0_2  = 5.44e-4
    beta_1 = 4.5
    beta_2 = 0.0

    if z <= 0.8: rho_dot = R0_1*(1+z)**beta_1
    if z > 0.8: rho_dot = R0_2

    scalefrac = rho_dot/R0_1
    classfrac1,classfrac2 = scalefrac*classfrac0[1],scalefrac*classfrac0[2]

    R0_1 = 2.2e-5
    R0_2 = 3.12e-3
    beta_1 = 2.15
    beta_2 = -5.0

    if z <= 1: rho_Ia = R0_1*(1+z)**beta_1
    if z > 1: rho_Ia = R0_2*(1+z)**beta_2
    scalefrac_Ia = rho_Ia/R0_1
    classfracIa = scalefrac_Ia*classfrac0[0]

    cfrac = np.array([classfracIa,classfrac1,classfrac2])/sum([classfracIa,classfrac1,classfrac2])

    return cfrac

def runReadPlotSim( simroot=None,  zrange=[0.4,2.3], avrange=[0,1], 
                    Nsim=2000, cadence=3, xaxis='W-H', yaxis='H', 
                    tobsrange=[0,0], snmags={}, Nbins=50, 
                    clobber=False, verbose=True, plottype='colormag',
                    ccbands='', dustmodel='mid', ratemodel = 'constant',
                    classfractions = [0.24, 0.19, 0.57],
                    perfect=True, etimelist=''):
    """ run the snana simulator with doClassSim(), 
    read in the SimTable results with rdClassSim(),
    plot the color-mag contours with plotClassSim().
    Returns the list of SimTable results [Ia, II, Ib/c]

    ccbands: four bands used to make color-color plots
             x-axis: band1-band2, yaxis: band3-band4
    ratemodel: 'constant' or 'powerlaw' (see simulate.py)
    classfractions: relative SN rates for Type Ia, Ib/c, II
    perfect: noiseless simulation if true
    model: 'SALT2 or MLCS'
    plottype: 'colormag' or 'colorcolor'
    """
    # define the simulation name if not provided
    if not simroot : 
        simroot = 'HST_classify'
        print( simroot ) 

    # check for existing simulation products
    simset=None
    if not clobber : 
        try : simset = rdClassSim( simroot ) 
        except : pass

    # run the sims if necessary
    if (not simset) or clobber : 
        if clobber<0: 
            # verify with user before running the sim
            userin = raw_input("""
Looks like we don't have an existing SNANA sim product for %s.
Do you want to run the simulations [y/n]""" % simroot) 
            if not userin.lower().startswith('y'):
                print('OK. Bye.')
                return( 0 )

        #redshift-dependent class fractions
        if not classfractions: classfractions = classfracz(np.mean(zrange))

        allbands= ''.join( np.unique( xaxis.strip('-')+yaxis.strip('-')) )
        simroot = doMonteCarloSim( 
            simroot=simroot, bands=allbands, 
            zrange=zrange, avrange=avrange, Nepoch=1,
            Nsim=Nsim, cadence=cadence, clobber=clobber,
            dustmodel=dustmodel, # classfractions=classfractions,
            ratemodel=ratemodel, perfect=perfect,
            etimelist=etimelist)
        simset = rdClassSim( simroot )
    
    # plot the simulation results :
    plotClassSim( simset, xaxis=xaxis, yaxis=yaxis, 
                  tobsrange=tobsrange, snmags=snmags, Nbins=Nbins, 
                  verbose=verbose, plottype=plottype, ccbands=ccbands )

    return( simset ) 




    

def doGridSim( Nsim=1000, simroot='HST_classifyGrid', 
               survey='CANDELS', field='default', bands='VXIZWJH', 
               zrange=[0.4,2.3], avrange=[0,7],
               x1range=[-3.0,3.0], crange=[-0.4,1.0],
               trestrange=[-15.,35.], treststep=1, 
               kcorfile='HST/kcor_HST.fits', 
               nlogz=0, ncolorpar=0, ncolorlaw=0, nlumipar=0, 
               clobber=False, omitTemplateIbc='', omitTemplateII='' ):
    """ master function to run the CC and the Ia simulations
    to produce a comparison sample for classification 
    in the given redshift and Av range.

    OPTIONS: 
    Nsim = total number of simulated SNe in each of the three classes (Ia,Ib/c,II)
    treststep : size of the time sampling step in rest-frame days 
    nlogz,ncolorpar,etc: explicitly specify the dimensions of the grid (overrides Nsim)

    RETURNS: 
     the root name of the resulting simulation set. 
    (use rdClassSim to read in the simulation results)
    """
    import simulate

    # make the shared .simlib file   (need .simlib only to specify the survey)
    simlibfile = 'sim_%s.simlib'%( simroot )
    if ( ( not os.path.exists( os.path.join( SNDATA_ROOT, 'simlib/'+simlibfile) )
           and not os.path.exists( simlibfile ) ) 
         or clobber ):
        simulate.mkgridsimlib( simlibfile, survey=survey, field=field, bands=bands, 
                               clobber=clobber )

    # When user has specified all the grid dimensions,
    # we trust the user to be doing something intelligent
    if np.all( [nlogz, ncolorpar, ncolorlaw, nlumipar] ) : 
        Nsim = nlogz * ncolorpar * ncolorlaw * nlumipar
        GENRANGE_REDSHIFT=['%.3f %.3f'%(tuple(zrange)), '# simulated redshift range']
        NgridZIa = NgridZIbc = NgridZII = nlogz
        NgridCPIa = NgridCPIbc = NgridCPII = ncolorpar
        NgridCLIa = NgridCLIbc = NgridCLII = ncolorlaw
        NgridLPIa = NgridLPIbc = NgridLPII = nlumipar
    # but when the user has just specified Nsim, we 
    # evaluate the redshift range and tune the grid
    # dimensions to avoid creating unnecessary steps
    # in redshift space
    elif abs(zrange[1]-zrange[0]) <= 0.003:
        NgridZIa,NgridZIbc,NgridZII = 1,1,1
        NgridLPIa = int(pow(Nsim,1/2.))
        NgridCPIa = int(pow(Nsim,1/2.))
        NgridLPIbc = 16
        NgridLPII = 27
        if omitTemplateIbc: NgridLPIbc = 15
        if omitTemplateII: NgridLPII = 26
        NgridCPIbc = int(Nsim/float(NgridLPIbc))
        NgridCPII = int(Nsim/float(NgridLPII))
        GENRANGE_REDSHIFT = ['%.3f %.3f'%(np.mean(zrange),np.mean(zrange)), '# simulated redshift range']
    elif abs(zrange[1]-zrange[0]) <= 0.01:
        NgridZIa,NgridZIbc,NgridZII = 3,3,3
        NgridLPIa = int(pow(Nsim/3.,1/2.))
        NgridCPIa = int(pow(Nsim/3.,1/2.))
        NgridLPIbc = 16
        NgridLPII = 27
        if omitTemplateIbc: NgridLPIbc = 15
        if omitTemplateII: NgridLPII = 26
        NgridCPIbc = int(Nsim/3./float(NgridLPIbc))
        NgridCPII = int(Nsim/3./float(NgridLPII))
        GENRANGE_REDSHIFT=['%.3f %.3f'%(tuple(zrange)), '# simulated redshift range']
    else:
        NgridLPIbc = 16
        NgridLPII = 27
        if omitTemplateIbc: NgridLPIbc = 15
        if omitTemplateII: NgridLPII = 26
        NgridZIa,NgridZIbc,NgridZII = int(pow(Nsim,1/3.)),int(pow(Nsim/float(NgridLPIbc),1/2.)),int(pow(Nsim/float(NgridLPII),1/2.))
        NgridLPIa = int(pow(Nsim,1/3.))
        NgridCPIa,NgridCPIbc,NgridCPII = int(pow(Nsim,1/3.)),int(pow(Nsim/float(NgridLPIbc),1/2.)),int(pow(Nsim/float(NgridLPII),1/2.))
        GENRANGE_REDSHIFT=['%.3f %.3f'%(tuple(zrange)), '# simulated redshift range']

    if nlogz: NgridZIa,NgridZIbc,NgridZII = nlogz,nlogz,nlogz

    # make the .input file for the Ia simulation
    inputIa = simulate.mkinput( simname=simroot+'_Ia', simlibfile=simlibfile, 
                                simtype='Ia', Nsim=Nsim, clobber=clobber, 
                                GENSOURCE='GRID',
                                GENFILTERS= [bands,'# List of filters to simulate'],
                                GENRANGE_SALT2x1=['%.1f %.1f'%(tuple(x1range)),'# x1 (stretch) range'],
                                GENRANGE_SALT2c=['%.1f %.1f'%(tuple(crange)),'# color range'],
                                GENRANGE_AV=['%.1f %.1f'%(tuple(avrange)),'# CCM89 extinc param range'],
                                GENRANGE_REDSHIFT=GENRANGE_REDSHIFT,
                                GENRANGE_TREST=['%.1f %.1f'%(tuple(trestrange)), '# range of rest-frame time'],
                                NGRID_TREST= int( (trestrange[1]-trestrange[0]) / float(treststep) ),
                                NGRID_LOGZ = [NgridZIa,'# redshift grid steps'],
                                NGRID_LUMIPAR = [NgridLPIa,'# x1, Delta, stretch, dm15'],
                                NGRID_COLORPAR = [NgridCPIa,'# AV or SALT2 color'],
                                KCOR_FILE = kcorfile,
                                )

    # make the .input file for the Ibc simulation
    inputIbc = simulate.mkinput( simname=simroot+'_Ibc', simlibfile=simlibfile, 
                                 simtype='Ibc', Nsim=Nsim, clobber=clobber, 
                                 GENSOURCE='GRID',
                                 GENFILTERS= [bands,'# List of filters to simulate'],
                                 GENRANGE_SALT2x1=['%.1f %.1f'%(tuple(x1range)),'# x1 (stretch) range'],
                                 GENRANGE_SALT2c=['%.1f %.1f'%(tuple(crange)),'# color range'],
                                 GENRANGE_AV=['%.1f %.1f'%(tuple(avrange)),'# CCM89 extinc param range'],
                                 GENRANGE_REDSHIFT=GENRANGE_REDSHIFT,
                                 GENRANGE_TREST=['%.1f %.1f'%(tuple(trestrange)), '# range of rest-frame time'],
                                 NGRID_TREST= int( (trestrange[1]-trestrange[0]) / float(treststep) ),
                                 NGRID_LOGZ = [NgridZIbc,'# redshift grid steps'],
                                 NGRID_LUMIPAR = [NgridLPIbc,'# x1, Delta, stretch, dm15'],
                                 NGRID_COLORPAR = [NgridCPIbc,'# AV or SALT2 color'],
                                 KCOR_FILE = kcorfile,
                                 )

    # make the .input file for the II simulation
    inputII = simulate.mkinput( simname=simroot+'_II', simlibfile=simlibfile, 
                                simtype='II', Nsim=Nsim, clobber=clobber, 
                                GENSOURCE='GRID',
                                GENFILTERS= [bands,'# List of filters to simulate'],
                                GENRANGE_SALT2x1=['%.1f %.1f'%(tuple(x1range)),'# x1 (stretch) range'],
                                GENRANGE_SALT2c=['%.1f %.1f'%(tuple(crange)),'# color range'],
                                GENRANGE_AV=['%.1f %.1f'%(tuple(avrange)),'# CCM89 extinc param range'],
                                GENRANGE_REDSHIFT=GENRANGE_REDSHIFT,
                                GENRANGE_TREST=['%.1f %.1f'%(tuple(trestrange)), '# range of rest-frame time'],
                                NGRID_TREST= int( (trestrange[1]-trestrange[0]) / float(treststep) ),
                                NGRID_LOGZ = [NgridZII,'# redshift grid steps'],
                                NGRID_LUMIPAR = [NgridLPII,'# x1, Delta, stretch, dm15'],
                                NGRID_COLORPAR = [NgridCPII,'# AV or SALT2 color'],
                                KCOR_FILE = kcorfile,
                                )

    # D. Jones
    # Now look for templates to omit, to avoid
    # self-identification for the SNPhotCC
    if omitTemplateIbc:
        fin = open(inputIbc,'r')
        fout = open(inputIbc+'.mod','w')
        for line in fin:
            line = line.replace('\n','')
            if omitTemplateIbc in line:
                line = '#' + line
            print >> fout, line
        fin.close()
        fout.close()
        os.system('mv %s %s'%(inputIbc+'.mod',inputIbc))
    if omitTemplateII:
        fin = open(inputII,'r')
        fout = open(inputII+'.mod','w')
        for line in fin:
            line = line.replace('\n','')
            if omitTemplateII in line:
                line = '#' + line
            print >> fout, line
        fin.close()
        fout.close()
        os.system('mv %s %s'%(inputII+'.mod',inputII))

    # run the simulations 
    simulate.dosim( inputIa )
    simulate.dosim( inputIbc )
    simulate.dosim( inputII )

    return( simroot )


def colorColorClassify( sn, mjd='peak', classfractions='all', dustmodel='all', 
                        colors='all', Nsim=3000, clobber=False, verbose=False ):
    """ Given an observed SuperNova, for each MJD, and for every
    available optical-IR color compute a P(Ia) value based on the gauss-weighted
    color-color classification metric.  

    sn : either a SuperNova object or a string pointing to a .dat file.
    mjd : 'peak' or 'all'  to use just the epoch nearest to peak or all epochs
    colors : choose which colors to compare 
         'all' : use all available filters
         'ir'  : require at least one IR band in each color (for high-z SNe)
         'opt' : require at least one uv/optical band in each color (for low-z SNe)

    classfractions and dustmodel  : 
      These are two components that we can vary to examine systmatics.
      For each, the user can specify 'high','mid','low', or use 'all' to 
      cycle through all three options, storing each separately.
      classfractions : prior for the fraction of SNe that are Ia
      dustmodel : the dust model to assume 
    """ 
    from __init__ import SuperNova
    IRBANDS = sn.SURVEYDATA.CAMERABANDLIST['IR']
    
    # if first arg is a string, read in the .dat file as a SN
    if type(sn) == str : sn = SuperNova( sn )

    userclobber = clobber
    
    if classfractions == 'all' :  cfraclist = ['high','mid','low']
    else : cfraclist=[classfractions]

    if dustmodel == 'all' :  dustlist = ['high','mid','low']
    else : dustlist=[dustmodel]

    # the colorClassification dictionary will hold the final results
    if 'colorClassification' not in sn.__dict__ : 
        sn.colorClassification = {}
    
    # Constrcut a list of unique colors
    bandlist = sorted( np.unique( sn.FLT ), key=lambda band : sn.SURVEYDATA.BLUEORDER.find( band ) )
    colorlist = []
    for band1 in bandlist[:-1] : 
        ib1 = bandlist.index( band1 )
        for band2 in bandlist[ib1+1:]: 
            ib2 = bandlist.index( band2 )
            if colors == 'ir' and (band1 not in IRBANDS) and (band2 not in IRBANDS):
                continue
            elif colors == 'opt' and (band1 in IRBANDS) and (band2 in IRBANDS):
                continue
            color = band1+'-'+band2
            colorlist.append( color )

    for dust in dustlist : 
        clobber=userclobber
        for cfrac in cfraclist : 
            priorstr = 'dust%s.cfrac%s'%(dust,cfrac)
            if verbose>1 : print(priorstr)
            
            pialist = []
            for color1 in colorlist[:-1] : 
                ic1 = colorlist.index( color1 )
                for color2 in colorlist[ic1+1:] : 
                   # compute the classification probabilities for this setup
                    psetlist = sn.getColorClassification( xaxis=color1, yaxis=color2, mjd=mjd, 
                                                          classfractions=cfrac, dustmodel=dust,
                                                          Nsim=Nsim, clobber=clobber, verbose=max(0,verbose-1) ) 
                    # Store the results into a nested dictionary structure
                    if priorstr not in sn.colorClassification : 
                        sn.colorClassification[ priorstr ] = {}
                    sn.colorClassification[priorstr][color1+'.'+color2] = psetlist
                    if verbose : 
                        # when mjd='all' we have a list of [pia,pibc,pii] sets, one for each mjd;
                        # boil these down to the median P(Ia) value for printing
                        if len(np.shape( psetlist )) == 0 : 
                            continue # failure in getColorClassification
                        elif len(np.shape( psetlist )) > 1 : 
                            pia = np.median(psetlist[:,0])  # mjd='all'
                        elif len(np.shape( psetlist )) == 1 :  
                            pia = psetlist[0]  # single mjd
                        else : 
                            pia = psetlist  # ???
                        pialist.append( pia )
                        print('  %s : %s vs %s  P(Ia)=%.2f'%(priorstr,color1,color2,pia))
                    clobber=False
            if verbose:  
                print(' ---- all colors P(Ia)=%.2f'%(np.median( pialist )) )

    if verbose and classfractions=='all' and dustmodel=='all' :  
        printColorClassification( sn )
    return( sn ) 



def colorMagClassify( sn, mjd='peak', classfractions='all', dustmodel='all', 
                      bands='all', Nsim=3000, modelerror=[0.0,0.0,0.0], 
                      clobber=False, verbose=False ):
    """ Given an observed SuperNova, for each MJD, and for every
    available color compute a P(Ia) value based on the gauss-weighted
    color-mag classification metric.  

    sn : either a SuperNova object or a string pointing to a .dat file.
    mjd : 'peak' or 'all'  to use just the epoch nearest to peak or all epochs

    classfractions and dustmodel  : 
      These are two components that we can vary to examine systmatics.
      For each, the user can specify 'high','mid','low', or use 'all' to 
      cycle through all three options, storing each separately.
      classfractions : prior for the fraction of SNe that are Ia
      dustmodel : the dust model to assume 
    """ 
    from __init__ import SuperNova
    from simplot import BANDORDER

    # if first arg is a string, read in the .dat file as a SN
    if type(sn) == str : sn = SuperNova( sn )

    userclobber = clobber
    
    if classfractions == 'all' :  cfraclist = ['high','mid','low']
    else : cfraclist=[classfractions]

    if dustmodel == 'all' :  dustlist = ['high','mid','low']
    else : dustlist=[dustmodel]

    # the colorClassification dictionary will hold the final results
    if 'colorClassification' not in sn.__dict__ : 
        sn.colorClassification = {}

    if bands=='all': bandlist = np.unique( sn.FLT )
    else : bandlist = bands
    for dust in dustlist : 
        clobber=userclobber
        for cfrac in cfraclist : 
            priorstr = 'dust%s.cfrac%s'%(dust,cfrac)
            if verbose>1 : print(priorstr)

            # compute color classifications for all 
            # possible colors and magnitudes 
            bluest = ''
            pialist = []
            for band2 in BANDORDER : 
                if band2 not in bandlist: continue
                if not bluest : 
                    bluest = band2
                    continue
                for band1 in BANDORDER : 
                    if band1 not in bandlist : continue
                    ib1 = BANDORDER.find( band1 ) 
                    ib2 = BANDORDER.find( band2 ) 
                    if ib2 <= ib1 : continue
                    color = band1+'-'+band2
                    mag = band2

                    if verbose>3 : print( 'getColorClassification %s  %s  %s  %s clobber=%s'%(cfrac, dust, color,mag,clobber) )
                    # compute the classification probabilities for this setup
                    psetlist = sn.getColorClassification( xaxis=color, yaxis=mag, mjd=mjd, 
                                                          classfractions=cfrac, dustmodel=dust,
                                                          Nsim=Nsim, modelerror=modelerror, 
                                                          clobber=clobber, verbose=max(0,verbose-1) ) 
                    clobber=False

                    # Store the results into a nested dictionary structure
                    if priorstr not in sn.colorClassification : 
                        sn.colorClassification[ priorstr ] = {}
                    sn.colorClassification[priorstr][color+'.'+mag] = psetlist

                    # when mjd='all' we have a list of [pia,pibc,pii] sets, one for each mjd;
                    # boil these down to the median P(Ia) value for printing
                    if len(np.shape( psetlist )) == 0 : 
                        continue # failure in getColorClassification
                    elif len(np.shape( psetlist )) > 1 : 
                        pia = np.median(psetlist[:,0])  # mjd='all'
                    elif len(np.shape( psetlist )) == 1 :  
                        pia = psetlist[0]  # single mjd
                    else : 
                        pia = psetlist  # ???
                    pialist.append( pia )
                    if verbose : 
                        print('  %s : %s vs %s  P(Ia)=%.2f'%(priorstr,color,mag,pia))

            sn.PIaColor = np.median( pialist ) 
            if verbose:  
                print(' ---- all colors P(Ia)=%.2f'%(np.median( pialist )) )

    if verbose and classfractions=='all' and dustmodel=='all' :  
        printColorClassification( sn )
    return( sn ) 


def colorClassify( sn, mjd='peak', classfractions='all', dustmodel='all', 
                   Nsim=3000, clobber=False, verbose=False ):
    __doc__ = colorMagClassify.__doc__
    return( colorMagClassify( sn, mjd=mjd, classfractions=classfractions, 
                              dustmodel=dustmodel, Nsim=Nsim, 
                              clobber=clobber, verbose=verbose ) )


def printColorClassification( sn ):
    import numpy as np 
    if 'colorClassification' in sn.__dict__ : 
        allPriors = sn.colorClassification
        pIaArray = []
        print( '          ------ DUST ------')
        print( 'FRACIa |  high    mid    low' )
        for cfrac in ['high','mid','low'] : 
            rowlist = []
            for dust in ['high','mid','low']:
                priorstr = 'dust%s.cfrac%s'%(dust,cfrac) 
                if priorstr not in allPriors.keys() : continue
                allColors = allPriors[priorstr]

                pialist = []
                for colormag in allColors.keys() : 
                    psetlist = allColors[colormag]
                    if len(np.shape( psetlist )) == 0 : 
                        continue # failure in getColorClassification
                    elif len(np.shape( psetlist )) > 1 : 
                        pia = np.median(psetlist[:,0])  # mjd='all'
                    elif len(np.shape( psetlist )) == 1 :  
                        pia = psetlist[0]  # single mjd
                    else : 
                        pia = psetlist  # ???
                    pialist.append( pia )
                if len(pialist):
                    pIaThisPriorSet = np.median( pialist )
                    rowlist.append( pIaThisPriorSet )
            if len(rowlist)==3 : 
                print( " %4s  |  %.2f   %.2f   %.2f  |  %.2f"%( cfrac, rowlist[0], rowlist[1], rowlist[2], np.mean(rowlist) ) )
            pIaArray.append(  rowlist ) 
        print( "        ------------------------------")
        pIaArray = np.array( pIaArray )
        if len(pIaArray.shape)==2 : pIa = pIaArray[1][1]
        else : pIa = np.median( pIaArray )
        syserrplus = np.max(pIaArray) - pIa 
        syserrminus = pIa - np.min(pIaArray)
        bottomrow =  [ pcolmean for pcolmean in np.mean( pIaArray, axis=0 )]
        bottomrow.append( pIa )
        if len(bottomrow)==3 : 
            print( "          %.2f   %.2f   %.2f  |  %.2f"% tuple(bottomrow) )
        print( "\n  %s  P(Ia) = %.2f +%.2f -%.2f"%(sn.name, pIa,syserrplus,syserrminus) )

    elif '_colorClassification' in sn.__dict__ : 
        print( "P(Ia) = %.2f"% sn._colorClassification[0] )
    return( pia ) 
    
