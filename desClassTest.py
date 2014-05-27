import numpy as np
import glob
import os
import pyfits
import time
from colors import *

# modelindex:[Type,ModelName,IAUname]
modelIndexDict = {-1:['Ia','SALT2','SALT2'],
                   0:['Ia','mlcs2k2','MLCS-U2'],
                   10:['Ibc','SNLS-04D1la','SNLS-04D1la'],
                   13:['Ib'  ,'SDSS-000020','SN2004ib'],
                   14:['Ib'  ,'SDSS-002744','SN2005hm'],
                   23:['Ib'  ,'SDSS-014492','SN2006jo'],
                   45:['Ib'  ,'SDSS-019323','SN2007nc'],
                   16:['Ic'  ,'SDSS-004012','SDSS-004012'],
                   18:['Ic'  ,'SDSS-013195','SN2006fo'],
                   22:['Ic'  ,'SDSS-014475','SDSS-014475'],
                   28:['Ic'  ,'SDSS-015475','SN2006lc'],
                   29:['Ic'  ,'SDSS-017548','SN2007ms'],
                   1:['Ic','NUGENT1bc','Nugent Ibc' ],
                   5:['Ic','CSP-2004fe' ,'CSP-2004fe' ],
                   6:['Ic','CSP-2004gq' ,'CSP-2004gq' ],
                   7:['Ic','CSP-2004gv' ,'CSP-2004gv' ],
                   8:['Ic','CSP-2006ep' ,'CSP-2006ep' ],
                   9:['Ic','CSP-2007Y'  ,'CSP-2007Y'  ],
                   11:['Ic','SNLS-04D4jv','SNLS-04D4jv'],
                   4:['II-L','NUGENT2L','NugentScolnic-IIL'],
                   2:['IIn','NUGENT2n','Nugent-IIn'],
                   17:['IIn' ,'SDSS-012842','SN2006ez'],
                   20:['IIn' ,'SDSS-013449','SN2006ix'],
                   3:['II-P','NUGENT2P','Nugent-IIP'],
                   12:['II-P','SDSS-000018','SN2004hx'],
                   15:['II-P','SDSS-003818','SN2005gi'],
                   19:['II-P','SDSS-013376','SN2006gq'],
                   21:['II-P','SDSS-014450','SN2006kn'],
                   24:['II-P','SDSS-014599','SN2006jl'],
                   25:['II-P','SDSS-015031','SN2006iw'],
                   26:['II-P','SDSS-015320','SN2006kv'],
                   27:['II-P','SDSS-015339','SN2006ns'],
                   30:['II-P','SDSS-017564','SN2007iz'],
                   31:['II-P','SDSS-017862','SN2007nr'],
                   32:['II-P','SDSS-018109','SN2007kw'],
                   33:['II-P','SDSS-018297','SN2007ky'],
                   34:['II-P','SDSS-018408','SN2007lj'],
                   35:['II-P','SDSS-018441','SN2007lb'],
                   36:['II-P','SDSS-018457','SN2007ll'],
                   37:['II-P','SDSS-018590','SN2007nw'],
                   38:['II-P','SDSS-018596','SN2007ld'],
                   39:['II-P','SDSS-018700','SN2007md'],
                   40:['II-P','SDSS-018713','SN2007lz'],
                   41:['II-P','SDSS-018734','SN2007lx'],
                   42:['II-P','SDSS-018793','SN2007og'],
                   43:['II-P','SDSS-018834','SN2007ny'],
                   44:['II-P','SDSS-018892','SN2007nv'],
                   }


def desClassify( istart=0, iend=-1, desdatdir='SIMGEN_PUBLIC_DES', useLuminosityPrior=False, modelerror=[0.05,0.07, 0.07],clobber=False, testrun=False ):

    import exceptions
    import sys
    import os

    # Define where the .dat files are
    if not os.path.isdir( desdatdir ) : 
        thisfile = sys.argv[0]
        if 'ipython' in thisfile : thisfile = __file__
        thispath = os.path.abspath( os.path.dirname( thisfile ) )
        desdatdir = os.path.join( thispath, desdatdir ) 
    if not os.path.isdir( desdatdir ) : 
        raise exceptions.RuntimeError( 'No directory %s'%os.path.basename( desdatdir ) )

    desdatfilelist = glob.glob( '%s/*.DAT'%desdatdir )

    if istart<0 : istart = len(desdatfilelist) +1 - istart
    if iend<0 : iend = len(desdatfilelist) +1 - iend
    iSubset = np.arange( istart, iend )

    outfile = 'DESclassTest_%05i-%05i_ulp%i.out'%(istart,iend,int(useLuminosityPrior))
    if os.path.isfile( outfile ) and not clobber : 
        print("Output file %s exists. Not clobbering."%outfile )
        return(0)

    print( "Classifying %i SN light curves. Writing results to %s"%( len( iSubset ), outfile ) )

    fout = open(outfile,'w')
    print >> fout,'#        DatFile Type zsim        Model   P_Ia  P_Ibc   P_II   chi2_Ia  chi2_Ibc  chi2_II   Ndof  time[sec]   bestIbcMod  bestIIMod'
    fout.close()
   
    for isn  in iSubset  :
        datfile = desdatfilelist[ isn ]
        print( 'Classifying SN %i (%i of %i) : %s'%( isn, isn+1 - iSubset[0], len(iSubset), datfile ) )
        outstr = doOneSN( os.path.abspath( datfile ), 
                          testrun=testrun, returnsn=False, 
                          useLuminosityPrior=useLuminosityPrior, modelerror=modelerror,
                          clobber=max(0,clobber-1) )
        fout = open(outfile,'a')
        print >> fout, outstr 
        fout.close()

    return


def doOneSN( datfile, useLuminosityPrior=False, modelerror=[0.05,0.07, 0.07],
             clobber=False, testrun=False, returnsn=False, debug=False ):

    if debug: import pdb; pdb.set_trace()

    from hstsnpipe.tools import snana
    start = time.time()
    try:
        sn = snana.SuperNova( datfile )
        sn.PEAKMJD = sn.MJD[ sn.signoise.argmax() ]

        thistypeindex = int(sn.SIM_NON1a.split()[0])
        thistype = modelIndexDict[ thistypeindex ][0]
        thismodel = sn.SIM_COMMENT.split('=')[-1].split('.')[0].strip()
        if thistype=='Ia' : omittemp = None
        else : omittemp=thismodel

        if testrun : 
            pIa, pIbc, pII = 0,0,0
            chi2Ia, chi2Ibc, chi2II = 0,0,0
            Ndof = 0
            bestIbc='None'
            bestII ='None'
        else : 
            sn.doGridClassify(clobber=clobber, useLuminosityPrior=useLuminosityPrior,
                              kcorfile='DES/kcor_DES_grizY.fits', modelerror=modelerror,
                              omitTemplateIbc=omittemp, omitTemplateII=omittemp,
                              nlogz=1, nlumipar=20, ncolorpar=20, ncolorlaw=1, npkmjd=20)
            pIa, pIbc, pII = sn.PIa,sn.PIbc,sn.PII
            chi2Ia, chi2Ibc, chi2II = min(sn.chi2Ia)/sn.Ndof,min(sn.chi2Ibc)/sn.Ndof,min(sn.chi2II)/sn.Ndof
            Ndof = sn.Ndof
            bestIbc = snana.constants.IBCMODELS[ '%03i'%sn.maxLikeIbcModel.LUMIPAR ][1] 
            bestII = snana.constants.IIMODELS[ '%03i'%sn.maxLikeIIModel.LUMIPAR ][1] 

        end = time.time()

        outstr='%15s %4s %5.3f %12s  %6.3f %6.3f %6.3f   %7.3f  %7.3f  %7.3f    %3i   %5i  %12s  %12s'%(
            os.path.basename(datfile), thistype, sn.z, thismodel, pIa, pIbc, pII,
            chi2Ia, chi2Ibc, chi2II, Ndof, int(end-start), bestIbc, bestII )

    except Exception as exc: 
        print('classify error : %s'%exc )
        end = time.time()
        bestIbc, bestII = 'None', 'None'
        outstr='%15s %4s %5.3f %12s  %6.3f %6.3f %6.3f   %7.3f  %7.3f  %7.3f    %3i   %5i   %12s  %12s  %s'%(
            os.path.basename(datfile), thistype, sn.z, thismodel, -9, -9, -9, -9, -9, -9, -9,  int(end-start), bestIbc, bestII, exc)

    if returnsn: 
        print( outstr ) 
        return( sn )
    return( outstr )



def plotOutput( outfile0='desClassTest_ulp0.out',
                outfile2='desClassTest_ulp2.out'):
    """ read in the results of the DES classification run 
    and plot results """
    from matplotlib import pyplot as pl

    pl.clf()

    for ulp,outfile in zip( [0,2], [outfile0, outfile2] ) :
        datfiles, types, models = np.loadtxt(
            outfile, usecols=[0,1,3], skiprows=1, unpack=True, dtype=str )
        z, pIa, pIbc, pII, chi2Ia, chi2Ibc, chi2II, Ndof, xtime  = np.loadtxt(
            outfile, usecols=[2,4,5,6,7,8,9,10,11], skiprows=1, unpack=True, dtype=float )

        pCC = pIbc + pII

        i50 = np.where( ( (types=='Ia') & (Ndof>0) & (pIa>0.5) ) | 
                        ( (types!='Ia') & (Ndof>0) & (pIa<0.5) ) )[0]
        i95 = np.where( ( (types=='Ia') & (Ndof>0) & (pIa>0.95) ) | 
                        ( (types!='Ia') & (Ndof>0) & (pIa<0.95) ) )[0]
    
        print( "UseLumPrior=%i : %.1f pct of input SN (%i of %i) are classified correctly (i.e. P(true|D)>50 pct )"%(
                ulp, 100*float(len(i50))/len(z), len(i50), len(z) ) )
        print( "UseLumPrior=%i : %.1f pct of input SN (%i of %i) are correct with high confidence. (i.e. P(true|D)>95 pct )"%(
                ulp, 100*float(len(i95))/len(z), len(i95), len(z) ) )

        chi2 = np.array( [chi2Ia,chi2Ibc,chi2II] )
        chi2min = chi2.min( axis=0 )

        iIa = np.where( (types=='Ia') & (Ndof>0) )
        iCC = np.where( (types!='Ia') & (Ndof>0) )
        iII = np.where( ( (types=='II') | (types=='II-P') | (types=='IIn') | (types=='II-L') ) & (Ndof>0) )
        iIbc= np.where( ( (types=='Ib') | (types=='Ic') | (types=='Ibc') ) & (Ndof>0) )

        ax1 = pl.subplot(2,2,ulp+1)
        for isubset,color,label in zip( [ iIa, iIbc,iII ], [red,green,blue], ['Ia','Ibc','II'] ) : 
            chi2hist, chi2histbins = np.histogram( chi2min[isubset], bins = np.arange(0,5,0.1) )
            ax1.plot( chi2histbins[:-1], chi2hist, marker=' ', ls='-', color=color, 
                      drawstyle='steps-mid', label=label )
        ax1.axvline( 1, ls='--', lw=2, color='0.3')
        ax1.set_xlabel( r'min $\chi^2/\nu$' )
        ax1.set_ylabel( r'N$_{\rm SN}$' )
        ax1.legend(loc='upper right', numpoints=2, frameon=False )

        ax2 = pl.subplot(2,2,ulp+2)

        x = 1 
        for color, sntypelist, pthistype in zip( 
            [red, green, blue],
            [['Ia',], ['Ibc','Ib','Ic'], ['II','II-P','II-L','IIn'] ], 
            [ pIa, pCC, pCC ] ):
            
            modIDlist = [ imod for imod in modelIndexDict.keys()
                          if np.any( [ modelIndexDict[imod][0] == thistype 
                                       for thistype in sntypelist ])  ]
            snanaModNamelist = [ modelIndexDict[imod][1] for imod in modIDlist ] 
            iauModNamelist = [ modelIndexDict[imod][2] for imod in modIDlist ] 

            xlabels = []
            for imod, snanamodname, iaumodname in zip( modIDlist, snanaModNamelist, iauModNamelist) : 
                n95 = len(np.where( (models==snanamodname) & (Ndof>0) & (pthistype>0.95) )[0])
                n50 = len(np.where( (models==snanamodname) & (Ndof>0) & (pthistype>0.5)  )[0])
                nall = float( len(np.where( (models==snanamodname) & (Ndof>0) )[0] ))
                if nall : 
                    x += 1
                    xlabels.append( iaumodname )
                    pl.bar( [ x ] , [ n50/nall ] , color=color, alpha=0.3, lw=1, width=[1,] )
                    pl.bar( [ x ] , [ n95/nall ] , color=color, alpha=0.6, lw=2, width=[1,] )
                continue
        ax2.set_xlabel( 'Model ID' )
        ax2.set_ylabel( 'Fraction Correct' )
        pl.draw()


