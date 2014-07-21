#! /usr/bin/env python
# S.Rodney
# 2012.03.12
#  Generate the .simlib and .input files needed to run 
#  a SNANA simulation of the CANDELS / CLASH  SN surveys.

# NOTE: This code expects that you are running SNANA at v9_94 or higher,
#   with SNDATA_ROOT updated after April, 2012, so it  includes the new 
#   SNID-based non1a templates from D.Scolnic 


import pyfits
import os 
import exceptions
import numpy as np
import pylab as p
import time
import glob
import constants

# #KCORFILE = 'HST/kcor_HST_VXIZWJH.his' 
# KCORFILE = { 'HST':'HST/kcor_HST_AB.fits',   # no k-corrections. Usable with SALT2 and NON1A only
#              'SNLS':'SNLS/kcor_EFFMEGACAM_BD17.fits' }
IAMODEL = 'SALT2.Guy10_UV2IR'    # or mlsc2k2 or salt2, etc.
# #KCORFILE = 'HST/kcor_HST_AB.fits'   # no k-corrections. Usable with SALT2 and NON1A only
# #IAMODEL = 'SNOOPY'
# 
# # non1a: rest-frame mag + K-correction (requires separate kcor table for each non1a model)
# # NON1A: compute observer-mag from SED (like SALT2) 
NONIAMODEL = 'NON1A'   # NON1A or non1a
# 
# DARKCURRENT={'ACS':0.006,'UVIS':0.0005,'IR':0.05}# e- /pix /sec
# 
# # From instrument handbooks, read noise per pixel:
# # RDNOISE = {'ACS':4.8,'UVIS':3.2,'IR':17}   # e-/pix 
# # Read noise in 0.4" aperture (from ETC)
# RDNOISE04 = {'ACS':60.22,'UVIS':75.96,'IR':114.28}   # e- 
# 
# FilterAlpha = { 'F218W':'D','F225W':'S','F275W':'T','F300X':'E',
#                 'F336W':'U','F390W':'C','F350LP':'W',
#                 'F435W':'B','F475W':'G','F555W':'F','F606W':'V','F625W':'R',
#                 'F775W':'X','F814W':'I','F850LP':'Z',
#                 'F125W':'J','F160W':'H','F125W+F160W':'A',
#                 'F105W':'Y','F110W':'M','F140W':'N',
#                 'F098M':'L','F127M':'O','F139M':'P','F153M':'Q',
#                 'G141':'4','G102':'2','blank':'0'
#                 }
# ACSbandlist =  ['B','G','V','R','X','I','Z']
# IRbandlist =   ['H','J','Y','M','N','L','O','P','Q']
# UVISbandlist = ['D','S','T','U','W','C']
# 
# 
# # PSF FWHM in arcsec
# PSF_FWHM_ARCSEC = { 'ACS': 0.13,'UVIS':0.07, 'IR':0.15 }
# 
# # Native pixel scale in arcsec per pixel
# PIXSCALE = { 'ACS': 0.05, 'UVIS':0.04, 'IR':0.13 }
# 
# 
# # True astronomer's zeropoints, giving the mag of a star
# #   that yields a total flux of 1 e-/sec within the photometric aperture
# # Vega mag zero points for an infinite aperture 
# ZPTINF_VEGA = { 
#     'F105W':25.6236,'F110W':26.0628,'F125W':25.3293,'F140W':25.3761,'F160W':24.6949,
#     'F098M':25.1057,'F127M':23.6799,'F139M':23.4006,'F153M':23.2098, 
#     'F435W':25.76695,'F475W':26.16252,'F555W':25.72747,'F606W':26.40598,
#     'F625W':25.74339,'F775W':25.27728,'F814W':25.51994,'F850LP':24.3230,}
# 
# # 0.4"  aperture, Vega mags
# ZPT04_VEGA = { 
#     'F350LP':26.6852,'F218W':21.0878,'F225W':22.2034,'F275W':22.4757,
#     'F336W':23.3531,'F390W':25.0240,
#     'F105W':25.452,'F110W':25.8829,'F125W':25.1439,'F140W':25.1845,'F160W':24.5037,
#     'F098M':24.9424,'F127M':23.4932,'F139M':23.2093,'F153M':23.0188, 
#     # NOTE: THESE ACS ZEROPOINTS ARE FOR AN INFINITE APERTURE
#     # THEY SHOULD BE CORRECTED BY SUBTRACTING THE 0.4" APERTURE CORRECTION
#     'F435W':25.76695,'F475W':26.16252,'F555W':25.72747,'F606W':26.40598,
#     'F625W':25.74339,'F775W':25.27728,'F814W':25.51994,'F850LP':24.3230, }
# 
# 
# # Infinite aperture, AB mags
# ZPTINF_AB = {
#         # TODO : COLLECT THE UVIS AB ZPTS FOR INFINITE APERTURE
#     'F105W':26.2687,'F110W':26.8223,'F125W':26.2303,'F140W':26.4524,'F160W':25.9463,
#     'F098M':25.6674,'F127M':24.6412,'F139M':24.4793,'F153M':24.4635, 
#     'F435W':25.65777,'F475W':26.05923,'F555W':25.7184 ,'F606W':26.49113,
#     'F625W':25.90667,'F775W':25.66504,'F814W':25.94333,'F850LP':24.84245 }
# 
# # 0.4"  aperture, AB mags
# ZPT04_AB = { 
#     'F350LP':26.8413,'F218W':22.7776,'F225W':23.8629,'F275W':23.9740,
#     'F336W':24.5377,'F390W':25.2389,
#     'F105W':26.0974,'F110W':26.6424,'F125W':26.0449,'F140W':26.2608,'F160W':25.7551,
#     'F098M':25.5041,'F127M':24.4545,'F139M':24.2880,'F153M':24.2725,
#     # NOTE: THESE ACS ZEROPOINTS ARE FOR AN INFINITE APERTURE
#     # THEY SHOULD BE CORRECTED BY SUBTRACTING THE 0.4" APERTURE CORRECTION
#     'F435W':25.65777,'F475W':26.05923,'F555W':25.7184 ,'F606W':26.49113,
#     'F625W':25.90667,'F775W':25.66504,'F814W':25.94333,'F850LP':24.84245 }
# 
# 
# 
# # Detector "Gain" (really Inverse Gain) for each camera, in e- / ADU
# GAIN = {'a':2.0, 'u':1.6, 'i':2.5, 
#         'ACS':2.0, 'UVIS':1.6, 'IR':2.5 }
# 
# # Sky count rates (e-/sec/pix) for Average zodaical light and earthshine
# # For WFC3, extracted from the IHB (includes thermal noise):
# # http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c09_exposuretime04.html
# # FOR ACS, extracted from ETC using 1x1-pix aperture: 
# #  http://etc.stsci.edu/etc/input/acs/imaging/
# SKYCPS = {'F218W':0.0005,'F225W':0.0066,'F275W':0.0037,'F336W':0.0018,
#           'F350LP':0.1077,'F390W':0.0098,
#           'F850LP':0.039,'F775W':0.078,'F625W':0.083,'F606W':0.127,
#           'F555W':0.054,'F475W':0.057,'F435W':0.030,'F814W':0.102,
#           'F098M':0.6106,'F105W':1.0150,'F110W':1.6611,'F125W':1.112,
#           'F127M':0.2697,'F139M':0.2391,'F140W':1.1694,
#           'F153M':0.2361,'F160W':0.943, }

def getNctrl( survey='HST', field='all', dustmodel='mid',
              Nsim=2000, decliners=False, showplots=False, 
              Nbins=5, dz=0.5, clobber=False, verbose=False, **kwargs):
    """ set up and run a simulation to compute the 'control count':
    the number of SNIa that would be detected in the given survey 
    field if the Ia rate were flat at 1 SNuVol = 1 SNIa 10-4 yr-1 Mpc-3 h3

    field : ['all','all1','all2','gsd','gsw','uds','cos','egs','gnd','gnw']
    dustmodel : ['all','high','mid','low']
    decliners : include SNe that peak before the survey starts? 

    To ~match Dahlen+ 2008 use dz=0.4 and Nbins=4 or 5
    """
    from __init__ import SimTable
    from constants import SurveyDataDict
    import os

    sndatadir = os.environ['SNDATA_ROOT']

    # scale the survey area by a large factor so that the control 
    # count (an integer) carries sufficient significant digits
    areascale = 1e6

    # set up the redshift bins for computing rates 
    # and for plotting detection efficiency histograms
    HbinsDetEff = np.arange(20,26.5,0.1)

    if decliners : 
        mjdrangelist = [ [-100,20], [-100,20],  [-70,20],  [-40,20],  [0,0]  ]
    else : 
        # When decliners are excluded, we drop the first epoch of every field,
        # and set the PKMJD range so that we simulate only SNe that are 
        # detectable as positive flux deviations relative to that template epoch
        # (see rates.controlTime for the derivation of these points)
        mjdrangelist = [ [-35,19], [-37,21],  [-19,22],  [-15,15],  [-5,5]  ]


    zrangelist = [ [ round( max(a,0.001),3), round( a+dz,3) ] for a in  \
                       np.linspace( 0, Nbins*dz, Nbins+1 )[:-1]  ]

    if Nbins == 6 : 
        mjdrangelist = [ [-35,19], [-37,21],  [-19,22], [-17,18], [-15,15],  [-5,5]  ]
    elif Nbins == 5 : 
        mjdrangelist = [ [-35,19], [-37,21],  [-19,22],  [-15,15],  [-5,5]  ]
    elif Nbins == 4 :
        mjdrangelist = [ [-35,19], [-37,21],  [-19,22],  [-15,15] ]
    elif Nbins==3 : 
        mjdrangelist = [ [-21,37], [-23,20], [-12,12] ]

    surveydata = SurveyDataDict[survey] 
    if field=='all': fieldlist = ['gsd','gsw','cos','uds','egs','gnd','gnw']
    else : fieldlist = [field]
    if dustmodel=='all': dustlist = ['high','mid','low']
    else : dustlist = [dustmodel]

    Nctrl = {}
    Nctrldz05 = {}
    zdz05 = {}
    DetEffH = {}
    DetEffz = {}
    for field in fieldlist : 
        simroot = '%s_%s_Nctrl'%(survey,field)
        simlibfile = '%s.simlib'%simroot

        for dustmodel in dustlist : 
            for zrange,mjdrange in zip( zrangelist, mjdrangelist ) :
                zbinsDetEff = np.arange(zrange[0],zrange[1]+0.05,0.05)
                zstr = 'z%.2f'%round(np.mean(zrange),2)

                # Set up the simulated PEAKMJD range, extending before/beyond
                # the survey dates according to the redshift-dependent  
                # ranges of detectability, as given in mjdrangelist
                if decliners : mjdlist = sorted( surveydata.EPOCHLIST[field] )
                else : mjdlist = sorted( surveydata.EPOCHLIST[field][1:] )
                mjdpk0 = min(mjdlist) + mjdrange[0]
                mjdpk1 = max(mjdlist) + mjdrange[1]

                # check for existing simulation products
                simname = '%s_dust%s_%s'%(simroot,dustmodel,zstr)
                simdatadir = os.path.join( sndatadir,'SIM/%s'%simname )
                simisdone = np.all([ os.path.isfile( os.path.join(simdatadir,'%s_%s.FITS'%(simname,sfx)) ) 
                                     for sfx in ['PHOT','HEAD'] ] )

                if not simisdone or clobber : 
                    if verbose>2: print(" simsdone=%s  clobber=%s ..."%(simisdone,clobber))
                    if verbose>1: print(" Running SNANA simulation for %s ..."%simname)

                    # make the simlib file
                    mjdlist,bandlist,etimelist = [],[],[]
                    for mjd in surveydata.EPOCHLIST[field] : 
                        for band,etime in zip(surveydata.SEARCHBANDS,surveydata.SEARCHETIMES) :
                            mjdlist.append( mjd ) 
                            bandlist.append( band )
                            etimelist.append( etime )
                            continue
                        continue
                    simlibfile = mksimlib( simlibfile, survey=survey, field=field,
                                           mjdlist=mjdlist, bandlist=bandlist, etimelist=etimelist,
                                           perfect=False, clobber=clobber, verbose=verbose )

                    # make the input file
                    inputfile = '%s.input'%simname 
                    inputfile = mkinput( simname,  inputfile=inputfile, simlibfile=simlibfile,
                                         survey=survey, field=field, simtype='Ia', Nsim=Nsim, 
                                         ratemodel='constant',       # SNR(z) : 'constant','powerlaw','flat'
                                         dustmodel=dustmodel,   # p(Av) or p(c) model : 'high','mid','low'
                                         applySearchEff=0,  # Keep all SNe, raising flags if not detected
                                         smear = True, # smear the magnitudes to reflect intrinsic variations ?
                                         clobber=clobber, verbose=verbose,
                                         GENRANGE_REDSHIFT= [ '%.3f %.3f'%(zrange[0],zrange[1]),'# range of simulated redshifts'],
                                         GENRANGE_PEAKMJD= [ '%.1f %.1f'%(mjdpk0,mjdpk1),'# range of simulated peak MJD dates'],
                                         GENFILTERS=['JH','# List of filters to simulate'],
                                         SOLID_ANGLE=[surveydata.SOLIDANGLE[field]*areascale,'# Field size in (scaled) steradians'],
                                         **kwargs )

                    # run the simulation 
                    dosim( inputfile, simname=simname, perfect=False, simargs='', verbose=verbose )

                # read in the simulation results
                sim = SimTable( simname, verbose=verbose )

                # Compute the Control Count : the number of  simulated detections        
                # First, normalize the distributions and scale by NSNSURVEY,
                # which is the number of SN events in the survey volume/time.
                # Also remove the area scaling factor. 
                idet = sim.DUMP['idet']
                Nctrlz = idet.sum() * float(sim.NSNSURVEY) / Nsim / areascale
                Nctrl['%s.%s.%s'%(field,dustmodel,zstr)] = Nctrlz

                # Compute the detection efficiency vs peak H mag
                NdetH,Hbins = np.histogram( sim.Hpk[idet], bins=HbinsDetEff )
                NsimH,Hbins = np.histogram( sim.Hpk, bins=HbinsDetEff )
                DetEffH['%s.%s.%s'%(field,dustmodel,zstr)] = np.ma.masked_invalid( NdetH / NsimH.astype(float) )
  
                # Compute the detection efficiency vs redshift
                Ndetz,zbins = np.histogram( sim.z[idet], bins=zbinsDetEff )
                Nsimz,zbins = np.histogram( sim.z, bins=zbinsDetEff )
                DetEffz['%s.%s.%s'%(field,dustmodel,zstr)] = np.ma.masked_invalid( Ndetz / Nsimz.astype(float) )

                # Record the ctrl count in bins of dz=0.05, 
                # for computing the rate with a rolling window function. 
                Nctrldz05['%s.%s.%s'%(field,dustmodel,zstr)] = Ndetz * float(sim.NSNSURVEY) / Nsim / areascale
                if verbose : print( "%.2f  %.2f   %.3f"%( zrange[0], zrange[1], Nctrlz ) )
            
                if showplots :
                    from matplotlib import pyplot as p
                    # Distribution of all SNe exploding in this volume
                    # (the total number here depends on the user-defined Nsim=NGENTOT_LC)
                    zbinsplot=np.arange(0,3.,0.1)
                    Nall,zall = np.histogram( sim.DUMP['Z'], bins=zbinsplot )
                    Nallsurvey = Nall / Nsim / areascale
                    idet = sim.DUMP['idet']
                    Ndet,zdet = np.histogram( sim.DUMP['Z'][idet], bins=zbinsplot )
                    Ndetsurvey = Ndet / Nsim / areascale
                    p.clf()
                    p.plot( zbinsplot[1:], Nallsurvey, drawstyle='steps-post', color='b', label=r'%i total SNe in survey volume \& time'%sim.NSNSURVEY )
                    p.plot( zbinsplot[1:], Ndetsurvey, drawstyle='steps-post', color='r', label='%i detectable SNe'%Ndetsurvey.sum() )
                    p.xlabel('redshift')
                    p.ylabel('Number of SNe')
                    p.legend( loc='upper left', frameon=False, numpoints=3, borderpad=0.2)

    # collapse all the fields to get Nctrl for each dust model
    # report everything : 
    NctrlRoll = {}
    for dustmodel in dustlist : 
        print( "\n\n---- Dust = %s ----"%dustmodel )
        header = '   z      ' + '     '.join( fieldlist ) + '      ALL'
        table = ''
        NctrlAllFieldsAllz = 0
        NctrlAllFieldsAllz_dz05 = []
        for zrange in zrangelist : 
            zstr = 'z%.2f'%round(np.mean(zrange),2)
            table += '\n%.1f-%.1f '%(tuple(zrange))
            NctrlAllFields = 0
            #NctrlAllFields_dz05 = []
            for field in fieldlist : 
                key = '%s.%s.%s'%(field,dustmodel,zstr) 
                table += ' %6.3f'%Nctrl[key]
                NctrlAllFields += Nctrl[key]
                #NctrlAllFields_dz05.append( Nctrldz05[key] )
            table += '  %6.3f'%NctrlAllFields
            NctrlAllFieldsAllz += NctrlAllFields
            #NctrlAllFieldsAllz_dz05.append( np.sum( NctrlAllFields_dz05, axis=0 ) )
        table += '\n total: %.3f'%NctrlAllFieldsAllz
        print( header + table )

        ## compute the ctrl count in rolling windows of width dz
        #NctrlAll_dz05 = np.ravel( NctrlAllFieldsAllz_dz05 )
        #zdz05 = np.arange( np.round(zrangelist[0][0],2), zrangelist[-1][-1], 0.05 )+0.025
        #zRoll = np.arange( np.round(np.mean(zrangelist[0]),2), zrangelist[-1][-1]-np.diff(zrangelist[-1])/2.+0.05, 0.05 )
        #NctrlRoll[dustmodel] = []
        #for zR in zRoll : 
        #    iinbin = np.where( (zdz05>=zR-(dz/2.)) & (zdz05<zR+(dz/2.)) )
        #    NctrlRoll[dustmodel].append( np.ravel(NctrlAllFieldsAllz_dz05)[iinbin].sum() )

    return( Nctrl) #, NctrlRoll, zRoll, DetEffH, DetEffz )




def mksimlib( simlibfile, survey='HST', field='default',
              bands='WJH', etimes = [400,1000,1000], 
              mjd0=55050.0, cadence=52, Nepoch=9, 
              mjdlist=[], bandlist=[], etimelist=[],
              perfect=False, clobber=False, verbose=False ) :
    """ Generate a .simlib file.

    There are two ways to provide the sequence of observations: 
    
    A) regularly spaced, identical epochs
       The obs dates are specified with  mjd0, cadence and Nepoch.
       The filter set (same in each epoch) is given as a string in bands. 
       Exposure times (same for each epoch) are in etimes

    B) heterogeneous epochs, explicitly defined
       each observation has a single entry in each of mjdlist, bandlist, etimelist
       (so all three lists must be the same length)

    Option A is defined with generic default values, but if  mjdlist, bandlist 
    and etimelist are provided, then option B supercedes option A. 

    SIMULATED NOISE : 
    Assumes an effective photometry aperture of radius r=0.4 arcsec,
    defined by the Gaussian PSF sigma  sig = r/2, so that the total
    effective area for noise calculation is  
      A = 4*pi*sig^2 = pi*r^2 = 0.503 arcsec^2
    """
    # TODO: define the gaussian psf_sig1 and sig2 more appropriately, 
    #  correctly accounting for non-gaussian psf shape

    # TODO : provide an option to use astronomical units for sky sigma and PSF size:
    # SKYSIG_UNIT: ADU_PER_SQARCSEC  
    # PSF_UNIT: ARCSEC_FWHM

    import exceptions
    from math import pi
    from string import ascii_uppercase as LETTERS
    from constants import SurveyDataDict

    if survey not in SurveyDataDict.keys(): 
        raise exceptions.RuntimeError("No SurveyDataDict entry for %s in constants.py."%survey)
    SURVEYDATA = SurveyDataDict[ survey ]

    # set up the list of observation dates if not explicitly provided
    if (not len(mjdlist) ) and ('EPOCHLIST' in SURVEYDATA.__dict__) :
        if field in SURVEYDATA.EPOCHLIST : 
            if len(SURVEYDATA.EPOCHLIST[ field ]) : 
                mjdlist = SURVEYDATA.EPOCHLIST[ field ]
    
    if isinstance( bandlist, basestring) : 
        bandlist = [b for b in bandlist]

    if not len(mjdlist) : 
        if not len(bandlist) and not len(etimelist): 
            if len(bands) != len(etimes) : 
                raise exceptions.RuntimeError("Error : bandlist / etimelist mismatch.")
        mjds = [ mjd0 + iepoch * cadence for iepoch in range( Nepoch ) ]
        for mjd in mjds : 
            for band, etime in zip(bands,etimes) : 
                mjdlist.append( mjd ) 
                bandlist.append( band ) 
                etimelist.append( etime ) 

    mjdlist = np.array( mjdlist ) 
    bandlist = np.array( bandlist ) 
    etimelist = np.array( etimelist ) 

    if perfect : 
        # turn up the exposure time to ridiculous to ensure high S/N even
        # for very faint simulated observations 
        etimelist = np.array([et*1e12 for et in etimelist])

    if os.path.exists( simlibfile ) and not clobber: 
        if verbose: print("%s exists. Not clobbering."%simlibfile)
        return( simlibfile )

    fout = open( simlibfile, 'w' )
    uniquebands = ''.join(np.unique(''.join( np.ravel(bandlist) )))

    head0 = """
SURVEY: %s     FILTERS: %s   TELESCOPE:  HST
USER: rodney       HOST: lucan.pha.jhu.edu
COMMENT: 'survey simulation generated with snana/simulate.py'

BEGIN LIBGEN  %s
"""%( survey, uniquebands, time.asctime() )
    print >> fout, head0
    libid = 1
    Nobs =  len(mjdlist)

     # header for libid 
    libidhd = """

# -------------------------------------------- 
LIBID: %i 
RA: %.5f   DECL: %.5f   NOBS: %i    MWEBV: 0.08   PIXSIZE: %.2f
FIELD: %s

#                            CCD    CCD         PSF1 PSF2 PSF2/1                    
#     MJD      IDEXPT  FLT  GAIN   NOISE SKYSIG  (pixels) RATIO ZPTAVG ZPTSIG  MAG 
""" %( libid, SURVEYDATA.COORD[field]['RA'], SURVEYDATA.COORD[field]['DEC'], 
       Nobs, SURVEYDATA.PIXSCALE['default'], survey+'-'+field )
    print >> fout, libidhd

    # write out the search epoch conditions, one line for each obs epoch and band
    for band in uniquebands : 
        iband = np.where( bandlist==band )[0]
        camera = SURVEYDATA.band2camera( band ) 
        print >>fout, "TELESCOPE:HST  PIXSIZE:%.2f"%SURVEYDATA.PIXSCALE[camera]

        # Here is where we assume an aperture of radius 0.4"
        # by setting the psf_sig1 radius to half of 0.4"
        psf_sig1 = 0.4 / SURVEYDATA.PIXSCALE[camera] / 2.
        #area = 4 * pi * psf_sig1**2
        gain = SURVEYDATA.GAIN[camera]
        rdnoise = SURVEYDATA.RDNOISE[ camera ]

        iobs = 0 
        for mjd, etime in zip(mjdlist[iband],etimelist[iband]) :
            if perfect : 
                skysig = 1
                zpt = 40
            else : 
                skysig = SURVEYDATA.skynoise( camera, band, etime )
                zpt = SURVEYDATA.zptsnana( camera, band, etime ) 
            iobs+=1
            print >> fout, "S: %.3f   %02i%03i   %s   %3.1f  %6.2f  %4.1f  %4.2f 0.00 0.000 %6.3f 0.008 -99.0" % (
                mjd,  LETTERS.find(band)+1, iobs, band, gain, rdnoise, skysig, psf_sig1, zpt )
            continue
        continue

    # footer for libid
    libidft = "\n\nEND_LIBID: %i" % libid
    print >>fout, libidft

    simlibfoot = "\n\nEND_OF_SIMLIB: %i ENTRIES\nDONE.\n" % Nobs
    print >>fout, simlibfoot
    fout.close()
    return( simlibfile )

def mkgridsimlib(simlibfile, survey='HST', field='default',
                 bands='WJH', verbose=True, clobber=False ):

    if os.path.exists( simlibfile ) and not clobber:
        if verbose: print("%s exists. Not clobbering."%simlibfile)
        return( simlibfile )

    fout = open(simlibfile,'w')

    head0 = """
SURVEY: %s     FILTERS: %s   TELESCOPE:  HST
USER: rodney       HOST: lucan.pha.jhu.edu
COMMENT: 'Survey simulation generated with snana/simulate.py'

BEGIN LIBGEN  Tue Feb  5 16:44:15 2013



# --------------------------------------------
LIBID: 1
RA: 0.0   DECL: 0.0   NOBS: 153    MWEBV: 0.08   PIXSIZE: 0.13
FIELD: %s
END_LIBID: 1
END_OF_SIMLIB: 0 ENTRIES
DONE.
"""%(survey,bands,survey)
    
    print >> fout, head0
    fout.close()

    return( simlibfile )


def mkinputGrid( simname,  inputfile=None, simlibfile=None,
                 survey='HST', simtype='Ia', 
                 ngrid_trest=50, genrange_trest=[-15,35],
                 ngrid_logz=50, genrange_redshift=[0.1,2.8],
                 ngrid_lumipar=20, genrange_salt2x1=[-5.0,3.0],
                 ngrid_colorpar=20, genrange_salt2c=[-0.4,1.0],
                 ngrid_colorlaw=1, genrange_rv=[3.1,3.1],
                 clobber=False, **kwargs ):
    """ convenience function for setting up the .input file
     for a grid simulation """

    snanaparam = {
        'GENSOURCE':'GRID', 'GRID_FORMAT':'FITS',
        'NGRID_TREST':ngrid_trest, 'GENRANGE_TREST':genrange_trest,
        'NGRID_LOGZ':ngrid_logz, 'GENRANGE_REDSHIFT':genrange_redshift,
        'NGRID_LUMIPAR':ngrid_lumipar, 'GENRANGE_SALT2X1':genrange_salt2x1,
        'NGRID_COLORPAR':ngrid_colorpar, 'GENRANGE_SALT2C':genrange_salt2c,
        'NGRID_COLORLAW':ngrid_colorlaw, 'GENRANGE_RV':genrange_rv,
        }
    snanaparam.update( **kwargs )
    return( mkinput( simname, inputfile=inputfile, simlibfile=simlibfile,
                     survey=survey, simtype=simtype, **snanaparam ) )

def mkinput( simname,  inputfile=None, simlibfile=None,
             survey='HST', field='default',
             simtype='Ia', Nsim=100, 
             ratemodel='constant',       # SNR(z) : 'constant','powerlaw','flat'
             dustmodel='mid',   # p(Av) or p(c) model : 'high','mid','low'
             applySearchEff=False,  # reject SNe below mag threshold?
             applyCutWin=False,     # reject SNe using S/N cuts?
             smear = True, # smear the magnitudes to reflect intrinsic variations ?
             clobber=False, Ngrid = '', **kwargs ):
    """ Make a sim-input file to run a SNANA monte carlo 
    simulation.   This more flexible form allows the user to provide 
    any SNANA .input file parameter as a keyword argument. 
    The keyword value may be a scalar element with just the parameter
    value, or a two-element list (or tuple) containing the 
    parameter value and a comment string.
    
    simname = string to define the simulation name. (required)

    Nsim = total number of SNe to simulate (NGENTOT_LC)

    simtype = 'Ia'  'Ibc'  'II'  'SLSN' (sets the powerlaw rate form)
           or specify a particular non1a model by type-code and model ID
           (e.g. simtype='33.218'  is Type Ic (33), model 218, aka SDSS-017548)

    rate : the form of the SNR(z) model ['constant','powerlaw','flat']
        'constant' = constant volumetric rate with redshift
        'powerlaw' = double power law in z. The powerlaw 
           parameters are automatically adjusted to ~match
           the observed SNR(z) according to the simtype. 
         'flat' = dN/dz is flat 

    dustmodel : [ 'high' , 'mid', 'low', 'flat', 'none' ]  
        the model to use for defining the dust distribution: dN/dAv or p(Av)

    smear : add some random magnitude variations to reflect the intrinsic dispersion 
      in luminosity and color ?   Uses the Chotard:2011  model for SNIa, and for 
      CCSNe the mag smearing is set to match the Li et al 2011 tables.
      (Keep this False when you want deterministic models, set to True for a more 
       realistic distribution of colors and magnitudes)
     
    """
    from constants import SurveyDataDict

    if survey not in SurveyDataDict.keys(): 
        raise exceptions.RuntimeError("No SurveyDataDict entry for %s in constants.py."%survey)
    SURVEYDATA = SurveyDataDict[ survey ]

    # TODO: If a simulation of the given name exists, it is not over-written unless 
    #         clobbering is turned on. 

    if not inputfile : inputfile = 'sim_%s.input' %( simname )
    if not simlibfile : simlibfile = 'sim_%s.simlib' %( simname )

    if simtype.lower() in ['ibc','ii','cc','sl','slsn'] : genmodel = SURVEYDATA.NONIAMODEL
    elif simtype.lower() == 'ia' : genmodel = SURVEYDATA.IAMODEL
    else : genmodel=SURVEYDATA.NONIAMODEL # user has provided a specific nonIa model number

    # set up the dust extinction model parameters 
    if dustmodel=='high' :  
        # For SNIa we approximate Neill 2006 with a broad SALT2c dist'n
        RANGE_SALT2c = '-0.4  1.00'
        SIGMA_SALT2c = '0.08  0.55' # bifurcated gaussian
        # For CCSN we approximate Dahlen+ 2012 with a gauss+exponential
        # which closely follows the Neill+2006 gaussian for Av<1.5
        # but then has a more RP05-like tail to high Av.
        # Note: like the N06 model this dist'n underpopulates at Av~0
        RANGE_AV = '0.0 7.0'
        TAU_AV=2.8       # dN/dAV = exp(-AV/tau)
        SIGMA_AV=0.5     #     += GAUSS(Av,sigma)
        RATIO_AV0=3.0    #  Gauss/exp ratio at AV=0
    elif dustmodel=='mid' : # approximate Kessler+ 2009 
        RANGE_SALT2c = '-0.4  1.00'
        SIGMA_SALT2c = '0.08  0.25'
        RANGE_AV = '0.0 7.0'
        TAU_AV=1.7    
        SIGMA_AV=0.6   
        RATIO_AV0=4.0  
    elif dustmodel=='low' :  # minimal dust model
        RANGE_SALT2c = '-0.4  1.00'
        SIGMA_SALT2c = '0.08  0.10'
        RANGE_AV = '0.0 7.0'
        TAU_AV=0.5  
        SIGMA_AV=0.15
        RATIO_AV0=1.0  
    elif dustmodel=='flat' :  # flat dust distribution
        RANGE_SALT2c = '-0.4  1.0'
        SIGMA_SALT2c = '1e12  1e12'
        RANGE_AV = '0.0 7.0'
        TAU_AV=100  
        SIGMA_AV=0
        RATIO_AV0=0  
    elif dustmodel=='none' :  # flat dust distribution
        RANGE_SALT2c = '-0.4  0.4'
        SIGMA_SALT2c = '0.06 0.03'
        RANGE_AV = '0.0 0.0'
        TAU_AV=0  
        SIGMA_AV=0
        RATIO_AV0=0  
    else :  # default SNANA setup
        RANGE_SALT2c = '-0.4   0.6'
        SIGMA_SALT2c =  '0.08  0.14'
        RANGE_AV = '0.0 3.0'
        TAU_AV=0.70
        SIGMA_AV=0
        RATIO_AV0=0  

    # Default : set the number of grid steps equally for the three primary
    # light curve parameters :  redshift, luminosity (x1), color (Av or c)
    # (only used by SNANA when GENSOURCE='GRID')
    if not Ngrid: Ngrid = int(pow(Nsim,1/3.))

    # dictionary of SNANA .input file parameters. Each entry 
    # is keyed by the parameter name and contains a 2-valued list
    # with the parameter value and a comment string
    defaultdict = {
        'GENVERSION':[simname,"# SNDATA version to generate"],
        'CLEARPROMPT':[0,'# 0=>overwrite previous version without prompt'],
        'SIMLIB_FILE':[simlibfile,'# list of observations'],
        'KCOR_FILE':[SURVEYDATA.KCORFILE,'# K corrections and filter definitions'],
        'FORMAT_MASK':[32,'# 2=ascii, observed; 4=ascii, model; 32=fits, obs+model'],
        'CIDOFF': [0,'# Offset for simulated SN index'],
        'GENMODEL':[genmodel,'# SALT2 or NON1A, extrapolated to 300-18000 Angstroms'],
        'GENFILTERS':[SURVEYDATA.ALLBANDS,'# List of filters to simulate'],
        'GENMEAN_SALT2x1':[0.0,'# salt2 stretch '],
        'GENRANGE_SALT2x1':['-5.0  +3.0','# x1 (stretch) range'],
        'GENSIGMA_SALT2x1':['1.5   0.9','# bifurcated sigmas'],
        'GENMEAN_SALT2c':[0.0,'# salt2 color'],
        'GENRANGE_SALT2c':[RANGE_SALT2c,'# color range'],
        'GENSIGMA_SALT2c':[SIGMA_SALT2c,'# bifurcated sigmas'],
        'GENALPHA_SALT2':[0.135, '# salt2 shape-luminosity modifier (OK for MW dust, D.Scolnic 2013.06.12'],
        #'GENBETA_SALT2':[3.19,'# salt2 color-luminosity modifier'],
        'GENBETA_SALT2':[4.1,'# salt2 color-luminosity modifier 4.1 = MW Dust'],
        'GENMEAN_DM15':[1.1,'# SNooPy stretch '],
        'GENRANGE_DM15':['0.7  2.2','# x1 (stretch) range'],
        'GENSIGMA_DM15':['0.3   0.3','# bifurcated sigmas'],
        'GENMEAN_RV':[3.1,'# mean RV to generate'],
        'GENRANGE_RV':['3.1 3.1','# RVrange'],
        'GENSIGMA_RV':['0.0 0.0','# RV sigma'],
        'GENRANGE_AV':[RANGE_AV,'# CCM89 extinc param range'],
        'GENTAU_AV':[TAU_AV,'# dN/dAV = exp(-AV/tau)'],
        'GENSIG_AV':[SIGMA_AV,'#     += GAUSS(Av,sigma)'],
        'GENRATIO_AV0':[RATIO_AV0,'#  Gauss/exp ratio at AV=0'],
        'EXTINC_MILKYWAY':[0,'# 0,1 => MW extinction off,on'],
        #'GENMODEL_ERRSCALE':[0.0,'# MLCS2k2 model ERROR scale (=> Hubble scatter WITH color smearing)'],
        #'GENMAG_OFF_MODEL':[0.0,"# Magnitude offset" ],
        #'GENMAG_SMEAR':[0.0,'# coherent mag-smear in all passbands # (=> Hubble scatter with NO color smearing)'],
        #'GENMAG_SMEAR_FILTER':['UBVRIZWJH 0','# and/or fixed smearing per filter (results in color variations)'],
        #'GENMODEL_ERRSCALE_CORRELATION':[0.0,'# correlation between GENMAG_SMEAR & ERRSCALER'],
        'GENSOURCE':['RANDOM','# RANDOM or GRID'],
        'NGEN_LC':[Nsim,'# stop simulating when you reach this number of SN lightcurves (after cuts)'],
        'NGENTOT_LC':[Nsim,'# total number of SN lightcurves to generate (before cuts)'],
        'GENRANGE_PEAKMJD':['55000 55100','# range of simulated peak MJD dates'],
        'GENSIGMA_SEARCH_PEAKMJD':[1.0,'# sigma-smearing for  SEARCH_PEAKMJD (days)'],
        'GENRANGE_TREST':['-15    35','# rest epoch relative to peak (days)'],
        'GENRANGE_REDSHIFT':['0.1  2.8', '# simulated redshift range'],
        'SOLID_ANGLE':[SURVEYDATA.SOLIDANGLE[field],'# Field size in steradians'],
        'OMEGA_MATTER':constants.OMEGA_MATTER,
        'OMEGA_LAMBDA':constants.OMEGA_LAMBDA,
        'W0_LAMBDA':constants.W0_LAMBDA,
        'H0':constants.H0,
        'NGRID_LOGZ':[Ngrid,'# redshift grid steps'],
        'NGRID_LUMIPAR':[Ngrid,'# x1, Delta, stretch, dm15'],
        'NGRID_COLORPAR':[Ngrid,'# AV or SALT2 color'],
        'NGRID_COLORLAW':[1,'# RV or Beta'], 
        'NGRID_TREST':[100,'# rest-frame epoch'],
        'GRID_FORMAT':['FITS','# TEXT or FITS'],
        }

    # Add in Ia model shift and smearing if requested:
    #if genmodel == IAMODEL : 
    #    # shift the Ia absolute magnitudes to match the center of the 
    #    #  Wang:2006 observed B band luminosity function 
    #    defaultdict['GENMAG_OFF_MODEL']=[0.0,'# Mag offset in all bands' ]
    if smear and genmodel.startswith('SALT2') : 
        # smear the Ia absolute magnitudes to match Wang:2006  and Chotard:2011
        defaultdict['GENMAG_SMEAR_MODELNAME'] = ['C11','# mag smearing follows Chotard:2011']
    elif smear and genmodel.startswith('SNOOPY'):
        defaultdict['GENMAG_SMEAR_FILTER_UV']=['U 0.1', '# mag smearing in rest-frame UV bands']
        defaultdict['GENMAG_SMEAR_FILTER_OPT']=['BVRI 0.08', '# mag smearing in rest-frame optical bands']
        defaultdict['GENMAG_SMEAR_FILTER_IR']=['JHK 0.05', '# mag smearing in rest-frame IR bands']
    else :
        defaultdict['GENMAG_SMEAR_MODELNAME'] = ['NONE','# no model-based mag smearing']

    # overwrite the default  parameters with user specified values
    pardict = dict( defaultdict.items() + kwargs.items() )

    # allow only one of NGEN_LC or NGENTOT_LC
    if 'NGEN_LC' in pardict.keys() and 'NGENTOT_LC' in pardict.keys() :
        if 'NGEN_LC' in kwargs.keys() : 
            pardict['#NGENTOT_LC'] = pardict.pop('NGENTOT_LC')
        else : 
            pardict['#NGEN_LC'] = pardict.pop('NGEN_LC')

    # Type out the .input file text content
    def inputline( key, val ):
        if type(val) in [list,tuple] : 
            if len(val)==2 : 
                if key.startswith('GENMAG_SMEAR_FILTER'):
                    # special handling for multiple GENMAG_SMEAR_FILTER keys
                    return('%-30s %-10s %s\n'%(key[:19]+':',val[0],val[1]))
                else : 
                    return('%-30s %-10s %s\n'%(key+':',val[0],val[1]))
            else : 
                return('%s: %s\n'%(key,val[0]))
        else : 
            return('%s: %s\n'%(key,val))

    inputtext = "#SNANA Simulation input file Generated using snana.simulate.mkinput\n\n"

    # GENSOURCE must be first parameter listed, so grid size paramaters can be set after 
    inputtext += inputline( 'GENSOURCE', pardict['GENSOURCE'])

    for k,v in pardict.iteritems():
        if k == 'GENSOURCE' : continue
        inputtext += inputline( k, v )

    # -------------------------------------------------------------------
    # Add some complex components that don't fit in simple key,val pairs.
    # -------------------------------------------------------------------

    if ratemodel == 'constant' : 
        # for constant rate we use a powerlaw with power=0
        # b/c it give clearer reporting in the .log files
        inputtext += """
#SN Rate vs redshift:
#DNDZ:  HUBBLE             # constant volumetric rate at all z
DNDZ: POWERLAW 1.0E-4 0.0  # Also a constant volumetric rate: SNR ~ (1+z)^0 (scaled to 1 SNuVol)
"""
    elif ratemodel == 'powerlaw' and simtype.lower()=='ia' : 
        # Redshift dependent power law 'fit' to SNIa data
        inputtext += """
#SN Rate vs redshift:
# Here is a two-power-law "fit" to current SNIa rates:
#                R0      Beta  Zmin  Zmax 
DNDZ: POWERLAW2 1.8E-5   2.15  0.0   1.0  # rate = R0(1+z)^Beta for z<1
DNDZ: POWERLAW2 9.5E-5   -0.25 1.0   9.1  # rate = R0(1+z)^Beta for z>1
"""

    elif ratemodel == 'powerlaw' and simtype.lower() in ['ib','ic','ii','ibc','cc'] : 
        # Redshift dependent power law 'fit' to CCSN data
        inputtext += """
# Here is a two-power-law "fit" to the SFR and measured CC SN rates:
#               R0      Beta  Zmin  Zmax 
DNDZ: POWERLAW2 5.0E-5   4.5   0.0   0.7  # rate = R0(1+z)^Beta for z<0.8
DNDZ: POWERLAW2 5.44E-4  0.0   0.7   9.1  # rate = constant for z>0.8
"""
    elif ratemodel == 'powerlaw' and simtype.lower() in ['sl','slsn'] : 
        # The above CCSN rate, reduced by a factor of 5e3 for SLSNe
        inputtext += """
# Here is a two-power-law model that "approximates" the SLSN rate
#               R0      Beta  Zmin  Zmax 
DNDZ: POWERLAW2 1.0E-8   4.5   0.0   0.7  # rate = R0(1+z)^Beta for z<0.8
DNDZ: POWERLAW2 1.089E-7  0.0   0.7   9.1  # rate = constant for z>0.8
"""
    else : 
        # simple flat distribution in redshift
        inputtext += """
#SN Rate vs redshift:
DNDZ: POWERLAW  1.0E-4  0.0      # flat rate at all z
"""

    if type(pardict['GENFILTERS'])==list : 
        searchbands = pardict['GENFILTERS'][0]
    elif type(pardict['GENFILTERS'])==str : 
        searchbands = pardict['GENFILTERS']

    inputtext += """
# ------------------------------------------------------------
#Apply selection cuts to simulate detection efficiency
#Using S/N ratio cuts   
#CUTWIN_OPT = 0 : don't apply  S/N selection cuts
#CUTWIN_OPT = 1 : Apply cuts, .DUMP includes only accepted SNe        
#CUTWIN_OPT = 3 : Apply cuts, .DUMP includes all SNe, even if rejected
APPLY_CUTWIN_OPT: %i 
%sCUTWIN_SNRMAX: 10 %s   1 -20 60  # S/N>10 for at least 1 of %s filters 
%sCUTWIN_SNRMAX:  5 %s   2 -20 60  # At least two bands must have S/N>5
"""%( int(applyCutWin), 
      '' if applyCutWin else '#', searchbands, searchbands, 
      '' if applyCutWin else '#', searchbands )

    inputtext += """
#Option 2 : search efficiency curve by mags
#0 : keep all SNe, raising flags if not detected (use with NGENTOT_LC)
#1 : keep SN if software trigger passes (use with NGEN_LC)
APPLY_SEARCHEFF_OPT: %i
SEARCHEFF_PIPELINE_FILE: %s  # in $SNDATA_ROOT/models/searcheff/
"""%( int(applySearchEff), SURVEYDATA.SEARCHEFF_PIPELINE_FILE )


    inputtext += """
#  ************************************************** 
#    Allowed SIMGEN_DUMP variables 
#   
# From SIM-GENERATION: 
# CID LIBID RA DECL MWEBV MWEBVMAP MWEBVERR ZCMB_SMEAR ZCMB ZHELIO VPEC Z GENZ MU DLMAG GALID GALZTRUE GALZPHOT GALZPH GALZERR GALSNSEP GALSNDM GALWGT PEAKMJD MJD0 MAGT0_H MAGT0_J MAGT0_W AV RV DELTA DM15 S2alpha SALT2alpha S2beta SALT2beta S2x0 SALT2x0 S2x1 SALT2x1 S2c SALT2c S2mb SALT2mb MAGSMEAR_COH VSI COLORSHIFT0 COLORSHIFT1 COLORSHIFT2 COLORSHIFT3 COLORSHIFT4 TRISE_SHIFT TFALL_SHIFT GENTYPE SNTYPE NON1A_INDEX NOBS NOBSDIF NEPOCH
#
# From CUTWIN-ANALYSIS: 
# REDSHIFT_FINAL TRESTMIN TMIN TRESTMAX TMAX SNRMAX SNRMAX1 SNRMAX2 SNRMAX3 SNRMAX4 SNRMAX5 SNRMAX_H SNRMAX_J SNRMAX_W TGAPMAX T0GAPMAX CUTMASK SIMEFMSK SIM_SEARCHEFF_MASK
#
#SNRMAX = max S/N ratio, across all bands
#SNRMAX1 = max S/N ratio, from the band with the highest SNR = SNRMAX
#SNRMAX2 = max S/N ratio, from the band with the second highest SNR
#etc.
  
#Dump out a summary .DUMP text file
SIMGEN_DUMP: %i CID LIBID SIM_EFFMASK Z MU SNTYPE NON1A_INDEX  PEAKMJD AV RV S2x0 S2x1 S2c S2mb %s %s

#  stop sim when effic error is this small (avoid infinite loops)
EFFERR_STOPGEN:	0.001
"""%( 14 + len(searchbands)*2, 'SNRMAX_'+' SNRMAX_'.join( searchbands ), 'MAGT0_'+' MAGT0_'.join( searchbands )  )

    non1aheader = """
# ----------------------------------------- 
# Auto-generated comments: 
#   NON1A keys for sim-input file. 
#   Created 2012-03-23   by  rkessler    HOST=sdssdp62.fnal.gov   
#   SCRIPT     : /home/s1/rkessler/snana/util/sednon1a_prep.pl  Scolnic_v8 
#   DIRECTORY  : /data/dp62.a/data/analysis/sednon1a_prep  
#   NSED(NON1A): 43 
# 
# User-generated comments: 
#   MAGOFF and MAGSMEAR adjusted to match m_R peak and sigma 
#   from Li:2011, Drout:2011, Kiewe:2012
#   Nugent SED templates updated by D.Scolnic as illustrated in 
#   http://kicp-workshops.uchicago.edu/SNphotID_2012/depot/talk-scolnic-daniel.pdf 

#SNTYPE  20  => II+IIP       (WGTSUM= 0.57 * 0.70 = 0.399)
#SNTYPE  21  => IIn+IIN      (WGTSUM= 0.57 * 0.10 = 0.057) 
#SNTYPE  22  => IIL          (WGTSUM= 0.57 * 0.20 = 0.114)

#SNTYPE  32  => Ib           (WGTSUM= 0.19 * 0.32 = 0.061) 
#SNTYPE  33  => Ic           (WGTSUM= 0.19 * 0.58 = 0.110) 

NON1A_KEYS: 5 
         INDEX   WGT    MAGOFF   MAGSMEAR  SNTYPE 
"""

    if smear : 
        IIPsmear = 0.80
        IInsmear = 1.00
        IILsmear = 0.42
        Ibsmear = 0.9
        Icsmear = 0.6
        SLsmear = 1.5
    else : 
        IIPsmear = 0.0
        IInsmear = 0.0
        IILsmear = 0.0
        Ibsmear  = 0.0
        Icsmear  = 0.0
        IcBLsmear = 0.0
        SLsmear = 0.0

    ibctable = """
NON1A:    103   0.0061  -0.280     %.3f     32    # Ib   (CSP-2004gv)
NON1A:    104   0.0061   1.720     %.3f     32    # Ib   (CSP-2006ep)
NON1A:    105   0.0061  -1.480     %.3f     32    # Ib   (CSP-2007Y)
NON1A:    202   0.0061  -0.680     %.3f     32    # Ib   (SDSS-000020)
NON1A:    203   0.0061  -1.080     %.3f     32    # Ib   (SDSS-002744)
NON1A:    212   0.0061   0.020     %.3f     32    # Ib   (SDSS-014492)
NON1A:    234   0.0061  -0.780     %.3f     32    # Ib   (SDSS-019323)

NON1A:    022   0.0122  -1.420     %.3f     33    # Ic   (SNLS-04D4jv)
NON1A:    101   0.0122  -0.600     %.3f     33    # Ic   (CSP-2004fe)
NON1A:    102   0.0122  -1.420     %.3f     33    # Ic   (CSP-2004gq)
NON1A:    205   0.0122  -0.700     %.3f     33    # Ic   (SDSS-004012)
NON1A:    207   0.0122  -0.700     %.3f     33    # Ic   (SDSS-013195)
NON1A:    211   0.0122   0.180     %.3f     33    # Ic   (SDSS-014475)
NON1A:    217   0.0122  -1.200     %.3f     33    # Ic   (SDSS-015475)
NON1A:    218   0.0122  -0.400     %.3f     33    # Ic   (SDSS-017548)
NON1A:    021   0.0122  -1.780     %.3f     33    # Ibc  (SNLS-04D1la)
"""%( tuple( [Ibsmear]*7 + [Icsmear]*9 ) )
       
    iitable = """
NON1A:    201   0.0166   2.400     %.3f     20    # IIP  (SDSS-000018)
NON1A:    204   0.0166   0.700     %.3f     20    # IIP  (SDSS-003818)
NON1A:    208   0.0166   0.900     %.3f     20    # IIP  (SDSS-013376)
NON1A:    210   0.0166   1.400     %.3f     20    # IIP  (SDSS-014450)
NON1A:    213   0.0166   1.500     %.3f     20    # IIP  (SDSS-014599)
NON1A:    214   0.0166   0.700     %.3f     20    # IIP  (SDSS-015031)
NON1A:    215   0.0166   1.200     %.3f     20    # IIP  (SDSS-015320)
NON1A:    216   0.0166   1.800     %.3f     20    # IIP  (SDSS-015339)
NON1A:    219   0.0166   2.400     %.3f     20    # IIP  (SDSS-017564)
NON1A:    220   0.0166   1.000     %.3f     20    # IIP  (SDSS-017862)
NON1A:    221   0.0166   1.300     %.3f     20    # IIP  (SDSS-018109)
NON1A:    222   0.0166   0.800     %.3f     20    # IIP  (SDSS-018297)
NON1A:    223   0.0166   0.800     %.3f     20    # IIP  (SDSS-018408)
NON1A:    224   0.0166   0.700     %.3f     20    # IIP  (SDSS-018441)
NON1A:    225   0.0166   1.400     %.3f     20    # IIP  (SDSS-018457)
NON1A:    226   0.0166   0.100     %.3f     20    # IIP  (SDSS-018590)
NON1A:    227   0.0166   1.000     %.3f     20    # IIP  (SDSS-018596)
NON1A:    228   0.0166   0.200     %.3f     20    # IIP  (SDSS-018700)
NON1A:    229   0.0166   0.450     %.3f     20    # IIP  (SDSS-018713)
NON1A:    230   0.0166   0.050     %.3f     20    # IIP  (SDSS-018734)
NON1A:    231   0.0166   1.700     %.3f     20    # IIP  (SDSS-018793)
NON1A:    232   0.0166   1.000     %.3f     20    # IIP  (SDSS-018834)
NON1A:    233   0.0166   1.100     %.3f     20    # IIP  (SDSS-018892)
NON1A:    235   0.0166   2.000     %.3f     20    # IIP  (SDSS-020038)

NON1A:    206   0.1140   0.450     %.3f     21    # IIN  (SDSS-012842)
NON1A:    209   0.1140  -0.500     %.3f     21    # IIN  (SDSS-013449)

#NON1A:    002   0.0800  -0.900     %.3f     22    # IIL  (Nugent+Scolnic_IIL)
"""%( tuple( [IIPsmear]*24 + [IInsmear]*2 + [IILsmear]*1 ) )

    slsntable = """
NON1A:    206   0.3300  -1.000     %.3f     21    # SLSN proxy [IIn] (SDSS-012842)
NON1A:    209   0.3300  -1.000     %.3f     21    # SLSN proxy [IIn] (SDSS-013449)
NON1A:    002   0.3300  -3.000     %.3f     22    # SLSN proxy [IIL] (Nugent+Scolnic_IIL)
"""%( tuple([SLsmear]*3 ) )


    if simtype.lower() == 'ibc' : 
        inputtext += non1aheader
        inputtext += ibctable
    elif simtype.lower() == 'ii' : 
        inputtext += non1aheader
        inputtext += iitable
    elif simtype.lower() == 'cc' : 
        inputtext += non1aheader
        inputtext += iitable
        inputtext += ibctable
    elif simtype.lower() in ['sl','slsn'] : 
        inputtext += non1aheader
        inputtext += slsntable
    elif simtype.lower().find('.')>0 : 
        # user has specified a single non1a model to simulate
        sntype     = int(simtype.split('.')[0]) 
        non1aindex = int(simtype.split('.')[1]) 
        inputtext += non1aheader
        for line in iitable.split('\n') + ibctable.split('\n') :
            if len(line.strip())==0 : continue
            thismod = int(line.split()[1])
            thistype = int(line.split()[5])
            if sntype != thistype : continue
            if non1aindex != thismod : continue
            inputtext += line
            break

    # generate the input file
    fout = open(inputfile,'w')
    print >>fout, inputtext
    fout.close()
    return( inputfile ) 



def dosim( inputfile, simname='', perfect=False, simargs='', verbose=False ): 
    """ Run the snana light curve simulator with the 
    given .input file """
    SNANA_DIR = os.environ['SNANA_DIR']
    SNDATA_ROOT = os.environ['SNDATA_ROOT']
    simdir = os.path.abspath( os.path.join( SNDATA_ROOT, 'SIM') )
    if not os.path.isdir( simdir ) : 
        print("Initializing an empty directory for SNANA simulation output: %s"%simdir)
        os.makedirs( simdir )

    tstart = time.time()
    if perfect : simargs += 'GENPERFECT 2'   # x10000 exp.time, but allows mag smearing and host Av
    if not simname : 
        simname = os.path.splitext( os.path.basename( inputfile ) )[0]

    if not os.path.isfile( inputfile ) : 
        raise exceptions.RuntimeError("ERROR: missing %s"%inputfile)

    simcmd = '%s  %s  %s > %s.log'%(
        os.path.join(SNANA_DIR,'bin/snlc_sim.exe'), inputfile, simargs, simname)
    if verbose>0 : print( 'Running SNANA simulation for %s:'%inputfile)
    if verbose>1 : print( '   ' + simcmd )           
    os.system( simcmd )
    if verbose>1 : print( '    Finished simulation in %i sec'%(
            int(time.time()-tstart ) ))
    return( )


       
    

def doSingleSim( simname=None, 
                 z=1, pkmjd=55500, model='Ia',  
                 trestrange=[-15,30], Av=0, 
                 mB=0, x1=0, c=0, dustmodel='mid',
                 survey='HST', field='default', bands='JHW',
                 Nobs='max', cadence='min', mjdlist=[], bandlist=[], 
                 perfect=True, verbose=False, clobber=False, debug=False ):
    """ set up and run a SNANA simulation to generate a single
    synthetic SN with the given parameters. 

    To simulate a type Ia with SALT2, set model='Ia', and set 
    the shape, color and luminosity  with  x1, c and mB.

    To simulate a non-Ia, the 'model' parameter must specify
    the particular non-Ia light curve template of interest,
    by type-code and model ID   (e.g. model='33.218'  is 
    Type Ic (33), model 218, aka SDSS-017548)
    
    You must also provide z and Av.  The luminosity may be 
    set with 

    Parameters required for both Ia and non-Ia simulations:
    z : redshift 
    pkmjd : date of peak brightness (for rest frame B band)
    trestrange : the rest-frame range of dates, 
    Nobs : the number of observation epochs per filter. 
           Provide an integer number, or use Nobs='max' 
           to fix this to the maximum number that SNANA 
           can simulate (as of c. v20,  SNANA has a cap 
           of 600 total observing epochs across all bands)

    cadence : the simulated observing cadence in all bands.
              Either provide the number of observer-frame days, 
              or use cadence='min' to fix this to the minimum 
              cadence that SNANA can simulate 
    
    Returns a SuperNova object. 
    """
    import os
    from __init__ import SuperNova
    if debug: import pdb; pdb.set_trace()

    # Av = 2.2*c   # Av ~ (Beta - 1 ) * c
    if simname==None and model=='Ia' : 
        simname = 'sim_Ia_mB%.1f_x1%.1f_c%.1f_z%.1f'%(mB,x1,c,z)
    elif simname==None : 
        simname = 'sim_%s_Av%.1f_z%.1f'%(model,Av,z)

    # generate the .input and .simlib files
    inputfile = 'single_%s.input'%(simname)        
    simlibfile = 'single_%s.simlib'%(simname)
    if not os.path.isfile( inputfile ) or clobber: 
        inputfile = mkinput( 
            simname, inputfile=inputfile, simlibfile=simlibfile,
            simtype=model, Nsim=1, ratemodel='constant', dustmodel=dustmodel,
            smear=False, clobber=clobber, 
            GENFILTERS=bands, GENRANGE_AV='%.4f  %.4f'%(Av,Av),
            GENMEAN_SALT2x1=x1, GENRANGE_SALT2x1='%s  %s'%(x1,x1), 
            GENMEAN_SALT2c=c, GENRANGE_SALT2c='%s  %s'%(c,c), 
            GENSOURCE='RANDOM',
            GENRANGE_PEAKMJD='%f %f'%(pkmjd,pkmjd),
            GENSIGMA_SEARCH_PEAKMJD=0.0,
            GENRANGE_TREST='%.3f  %.3f'%tuple(trestrange),
            GENRANGE_REDSHIFT='%s  %s'%(z,z),
            GENSIGMA_REDSHIFT=0, NGRID_LUMIPAR=1,
            )
    elif verbose : 
        print( "%s exists. Not clobbering."%inputfile )

    if not os.path.isfile( simlibfile ) or clobber: 
        # set the simulated observation dates based on pkmjd, z and trestrange
        # (note:if the user provided an explicit mjdlist+bandlist, it will override this)
        mjd0=pkmjd+(1+z)*(trestrange[0])
        mjd1=pkmjd+(1+z)*(trestrange[1])
        if cadence in [0,'min','best'] : 
            if bandlist : Nbands = len(bandlist)
            else : Nbands = len(bands)
            Nepoch= min(150,int(595/Nbands) - 1)
            cadence = (mjd1 - mjd0) / float(Nepoch)
        else : 
            Nepoch= min(150,int((mjd1 - mjd0) / cadence)+1)
        mksimlib( simlibfile, survey=survey, field=field, 
                  bands=bands,  etimes =[ 10000]*len(bands), 
                  mjd0=mjd0, Nepoch=Nepoch, cadence=cadence, 
                  mjdlist=mjdlist, bandlist=bandlist, 
                  etimelist=np.ones(len(mjdlist)).tolist(),
                  perfect=perfect, clobber=clobber )
    elif verbose : 
        print( "%s exists. Not clobbering."%simlibfile )

    # check for existing simulation products
    sndatadir = os.environ['SNDATA_ROOT']
    simdir = os.path.abspath( os.path.join( sndatadir, 'SIM') )
    if not os.path.isdir( simdir ) : 
        print("Initializing an empty directory for SNANA simulation output: %s"%simdir)
        os.makedirs( simdir )
    simdatadir = os.path.join( sndatadir,'SIM/%s'%simname )
    photfits = os.path.join( simdatadir, '%s_PHOT.FITS'%simname)
    headfits = os.path.join( simdatadir, '%s_HEAD.FITS'%simname)
    if not ( os.path.isfile(photfits) and os.path.isfile(headfits) ) or clobber : 
        # run the simulation 
        dosim( inputfile, simname=simname, perfect=perfect )
    elif verbose : 
        print( "%s simulation exists. Not clobbering."%simname )

    # read in  and return the simulated SN light curve as a SuperNova object
    simsn = SuperNova( simname=simname, snid=1, verbose=verbose )

    if model=='Ia' and mB>0:
        # The SNANA simulator fixes the simulated x0 value (and therefore 
        # the mB value, and the flux scaling in all bands) with this formula :
        #   SIMx0 = 1e12  * 10**( -0.4 * ( mu(z) - alpha*x1 + beta*c ) )
        # Here the alpha, beta parameters and the mu(z) calculation rely
        #  on the cosmological parameters specified in the .siminput file, 
        # We therefore need to rescale the flux of the simulated light curve in all
        # bands so that it uses the correct value of x0 (ie mB), as determined 
        # by the salt2 fitting of the input (actual, observed) SN light curve.
        mBsim = simsn.SIM_SALT2mB
        deltamB = mB - mBsim
        fluxfactor = 10**(-0.4*(deltamB))
        
        simsn.FLUXCAL *= fluxfactor
        simsn.FLUXCALERR *= fluxfactor
        simsn.MAG -= deltamB * (simsn.MAG<99)
    return( simsn )



def dozGrid( name ):
    """ make a simulation grid over redshift
    generate plots
    """

    mksimlib('zGrid', survey='HST', field='default', bands='GVIZYJNH', etimes=[10000]*8, mjd0=55050.0, cadence=2, Nepoch=9, mjdlist=[], bandlist=[], etimelist=[], perfect=False, clobber=False, verbose=False)

    mkinputGrid('snIa_zGrid',  GENFILTERS='BGVIZYJNH', simlibfile='zGrid.simlib', survey='HST', simtype='Ia', ngrid_trest=20, genrange_trest=[-15, 60], ngrid_logz=50, genrange_redshift=[0.1, 2.8], ngrid_lumipar=5, genrange_salt2x1=[-5.0, 3.0], ngrid_colorpar=5, genrange_salt2c=[-0.4, 1.0], ngrid_colorlaw=1, genrange_rv=[3.1, 3.1], clobber=False )
    mkinputGrid('snII_zGrid',  GENFILTERS='BGVIZYJNH', simlibfile='zGrid.simlib', survey='HST', simtype='II', ngrid_trest=20, genrange_trest=[-15, 60], ngrid_logz=50, genrange_redshift=[0.1, 2.8], ngrid_lumipar=5, genrange_salt2x1=[-5.0, 3.0], ngrid_colorpar=5, genrange_salt2c=[-0.4, 1.0], ngrid_colorlaw=1, genrange_rv=[3.1, 3.1], clobber=False )
    mkinputGrid('snIbc_zGrid',  GENFILTERS='BGVIZYJNH', simlibfile='zGrid.simlib', survey='HST', simtype='Ibc', ngrid_trest=20, genrange_trest=[-15, 60], ngrid_logz=50, genrange_redshift=[0.1, 2.8], ngrid_lumipar=5, genrange_salt2x1=[-5.0, 3.0], ngrid_colorpar=5, genrange_salt2c=[-0.4, 1.0], ngrid_colorlaw=1, genrange_rv=[3.1, 3.1], clobber=False )

    dosim('sim_snIa_zGrid.input', perfect=True )
    dosim('sim_snII_zGrid.input', perfect=True )
    dosim('sim_snIbc_zGrid.input', perfect=True )


def plotzGrid( showerr=False ):
    from __init__ import SimTable
    from matplotlib import pyplot as pl

    simIa = SimTable('snIa_zGrid')
    if simIa.MAGREF=='AB' or simIa.SURVEYDATA.KCORFILE.endswith('AB.fits') :
        magsystem = 'AB'
    else :
        magsystem = 'Vega'
    z = simIa.z
    I = np.ma.masked_greater_equal( simIa.LCMATRIX[:,:,:,:,3,:], 32 )
    V = np.ma.masked_greater_equal( simIa.LCMATRIX[:,:,:,:,2,:], 32 )
    VI = V-I
    VIzmean = np.array( [ np.ma.mean(VI[:,:,:,iz,:]) for iz in range(50) ] )
    VIzerr = np.array( [ np.ma.std(VI[:,:,:,iz,:]) for iz in range(50) ] )
    VIzmedian = np.array( [ np.ma.median(VI[:,:,:,iz,:]) for iz in range(50) ] )
    pl.plot( z, VIzmedian, color='r', ls='-' )
    if showerr : 
        pl.plot( z, VIzmean, color='r', ls='--' )
        pl.fill_between( z, VIzmedian-VIzerr, VIzmedian+VIzerr, color='r', alpha=0.3 )

    simII = SimTable('snII_zGrid')
    z = simII.z
    I = np.ma.masked_greater_equal( simII.LCMATRIX[:,:,:,:,3,:], 32 )
    V = np.ma.masked_greater_equal( simII.LCMATRIX[:,:,:,:,2,:], 32 )
    VI = V-I
    VIzmean = np.array( [ np.ma.mean(VI[:,:,:,iz,:]) for iz in range(50) ] )
    VIzerr = np.array( [ np.ma.std(VI[:,:,:,iz,:]) for iz in range(50) ] )
    VIzmedian = np.array( [ np.ma.median(VI[:,:,:,iz,:]) for iz in range(50) ] )
    pl.plot( z, VIzmedian, color='b', ls='-' )
    if showerr : 
        pl.plot( z, VIzmean, color='b', ls='--' )
        pl.fill_between( z, VIzmedian-VIzerr, VIzmedian+VIzerr, color='b', alpha=0.3 )

    simIbc = SimTable('snIbc_zGrid')
    z = simIbc.z
    I = np.ma.masked_greater_equal( simIbc.LCMATRIX[:,:,:,:,3,:], 32 )
    V = np.ma.masked_greater_equal( simIbc.LCMATRIX[:,:,:,:,2,:], 32 )
    VI = V-I
    VIzmean = np.array( [ np.ma.mean(VI[:,:,:,iz,:]) for iz in range(50) ] )
    VIzerr = np.array( [ np.ma.std(VI[:,:,:,iz,:]) for iz in range(50) ] )
    VIzmedian = np.array( [ np.ma.median(VI[:,:,:,iz,:]) for iz in range(50) ] )
    pl.plot( z, VIzmedian, color='g', ls='-' )
    if showerr : 
        pl.plot( z, VIzmean, color='g', ls='--' )
        pl.fill_between( z, VIzmedian-VIzerr, VIzmedian+VIzerr, color='g', alpha=0.3 )

    ax = pl.gca()
    pl.xlabel('Redshift')
    pl.ylabel( 'V-I color (%s)'%magsystem)
    pl.text( 0.95, 0.95, 'Type Ia', color='r',  fontsize='large', fontweight='bold', ha='right', va='top', transform=ax.transAxes )
    pl.text( 0.95, 0.9, 'Type Ib/c', color='g', fontsize='large',  fontweight='bold', ha='right', va='top', transform=ax.transAxes )
    pl.text( 0.95, 0.85, 'Type II', color='b',  fontsize='large', fontweight='bold', ha='right', va='top', transform=ax.transAxes )
