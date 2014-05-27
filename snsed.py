#! /usr/bin/env python
#S.rodney
# 2011.05.04
"""
Extrapolate the Hsiao SED and the non1a SEDs 
down to 300 angstroms to allow the W filter to 
reach out to z=2.5 smoothly in the k-correction 
tables and model light curves.
"""

import os
from numpy import *
from pylab import * 

try : sndataroot = os.environ['SNDATA_ROOT']
except KeyError: 
    sndataroot = os.path.abspath( '.' )

MINWAVE = 300     # min wavelength for extrapolation (Angstroms)
MAXWAVE = 20000   # max wavelength for extrapolation (Angstroms)

def extendEverything( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                      doCC=True, doIa=True ) :
    """ Run all the extrapolation functions, for both the SALT2 model 
    components and all the non1a SEDs """

    if doCC : 
        extendNon1a()

    if doIa : 
        salt2dir = os.path.join( sndataroot, salt2dir ) 
        if not os.path.isdir( salt2dir ) : 
            os.makedirs( salt2dir )

        # make the Hsiao07.extrap.dat template SED
        hsiao07 = os.path.join( sndataroot, 'snsed/Hsiao07.dat')
        hsiao07ext = os.path.join( sndataroot, 'snsed/Hsiao07.extrap.dat')
        extrapolatesed_linfit( hsiao07, hsiao07ext, Npt=10, Nsmooth=0, forceslope=True )

        extendSALT2_temp0()
        extendSALT2_temp1()
        extendSALT2_flatline()    


def getsed( sedfile = os.path.join( sndataroot, 'snsed/Hsiao07.dat'), day='all'  ) : 
    d,w,f = loadtxt( sedfile, unpack=True ) 

    #d = d.astype(int)
    days = unique( d ) 

    if day == 'all' : 
        dlist = [ d[ where( d == day ) ] for day in days ]
        wlist = [ w[ where( d == day ) ] for day in days ]
        flist = [ f[ where( d == day ) ] for day in days ]
        return( dlist, wlist, flist )
    else : 
        return( w[ where( d == day ) ], f[ where( d == day ) ] )


def plotsed( sedfile= os.path.join( sndataroot, 'snsed/Hsiao07.dat'), 
             day='all', normalize=False, **kwarg): 
    dlist,wlist,flist = getsed( sedfile ) 
    #days = unique( dlist ) 
    for i in range( len(wlist) ) : 
        thisday = dlist[i][0]
        
        #defaults = { 'label':str(thisday) } 
        #plotarg = dict( kwarg.items() + defaults.items() )
        if day!='all' : 
            if abs(thisday-day)>0.6 : continue
        offset, normfactor = 0, 1
        if normalize : 
            if normalize > 1 : 
                normfactor = (flist[i][:-1] * (wlist[i][1:]-wlist[i][:-1])).sum()
            else : 
                normfactor = flist[i].max()
                offset = thisday
        plot( wlist[i], flist[i]/normfactor + offset, **kwarg )
        # user_in=raw_input('%i : return to continue'%i)


def extrapolatesed( sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, 
                    Bzptwave=900, Bextwave=2000, Bfrac=0.2, Brefwave=2800, 
                    Nred = 10, Nsmooth=0, verbose=False ):
    """
    Set bluewave and bluefrac to define an extrapolation point
    blueward of the low-wavelength edge of the SED: 
      Bextwave = wavelength of the extrapolation point
      Brefwave = reference wavelength for setting extrapolated flux 
      Bfrac = fraction of the flux at Brefwave to set as the 
          flux value at Bextwave
    The linear extrapolation  uses only this fixed point and
    the input SED point at Brefwave. 

    On the red side (where the SED tail is more well behaved)
    we perform a linear fit to the last Nred points to define 
    the extrapolation.  Set Nsmooth to a positive integer to set 
    the window size for median smoothing of the input spectrum 
    before the linear fitting. 

    e.g.  Bzptwave, Bextwave, Bfrac, Brefwave = 900, 2000, 0.2, 2800 
     fix the flux at 900 angstroms to be 0. 
     fix the flux at 2000 angstroms to be 20% of the input SED's
     flux at 2800 angstrom, then define a piecewise linear extrapolation
     connecting the three anchor points at 900, 2000, and the minimum 
     wavelength of the input SED.

    Note: For CCSN SEDs, we use 
       Bzptwave=900, Bextwave=2000, Bfrac=0.2, Brefwave=2800, 
      
    and for a SNIa SED, we might use 
       Bzptwave=600, Bextwave=800, Bfrac=0.1, Brefwave=1000
     
    """
    from scipy import interpolate as scint
    from scipy import stats
    import shutil

    medsmooth = lambda f,N : array( [ median( f[max(0,i-N):min(len(f),max(0,i-N)+2*N)]) for i in range(len(f)) ] )

    dlist,wlist,flist = getsed( sedfile )  # these are 2-D lists of SEDs at different phases
    dlistnew, wlistnew, flistnew = [],[],[]

    fout = open( newsedfile, 'w' )
    for i in range( len(dlist) ) : 
        d,w,f = dlist[i],wlist[i],flist[i]

        if minwave  < w[0] : 
            wavestep = w[1] - w[0]
            irefpt = abs( w - Brefwave ).argmin()
            wrefpt = w[irefpt]
            frefpt = f[irefpt]
            wextpt = Bextwave
            fextpt = Bfrac * frefpt

            if wextpt < w[0] : 
                # Blueward extrapolation from the minimum wavelength of the input 
                # SED down to the first extrapolation point
                a = (f[0]-fextpt)/(w[0]-wextpt)  # slope of extrapolation line
                b = f[0] - a*w[0] # intercept
                wextBlue = arange( wextpt, w[0], wavestep )
                fextBlue = array( [ max( 0, a * wave + b ) for wave in wextBlue ] ) 
                w = append( wextBlue, w )
                f = append( fextBlue, f )
            if minwave < w[0] : 
                # Blueward extrapolation to the min wavelength (fixed to zero flux at Bzptwave)
                wextpt = Bzptwave
                fextpt = 0
                a = (f[0]-fextpt)/(w[0]-wextpt)  # slope of extrapolation line
                b = f[0] - a*w[0] # intercept
                wextBlue = arange( minwave, w[0], wavestep )
                fextBlue = array( [ max( 0, a * wave + b ) for wave in wextBlue ] ) 
                w = append( wextBlue, w )
                f = append( fextBlue, f )
                
        if maxwave > w[-1] : 
            # Redward linear extrapolation from last Nred points
            wavestep = w[-1] - w[-2]
            wN = w[-Nred:]

            if Nsmooth : fN = medsmooth( f, Nsmooth )[-Nred:]
            else : fN = f[-Nred:]
            (a,b,rval,pval,stderr)=stats.linregress(wN,fN)

            if a > 0 : 
                # re-do redward  linear extrapolation using the peak
                # in order to guarantee a downward slope
                if verbose : print( "  re-fitting %s day %i to ensure downward slope to the red"%(sedfile, d[0]) )
                wN = [ w[ f.argmax() ], w[-1] ]
                fN = [ f.max(), f[-1],  ]
                (a,b,rval,pval,stderr)=stats.linregress(wN,fN)

            wextRed = arange( w[-1]+wavestep, maxwave, wavestep )
            fextRed = array( [ max( 0, a * wave + b ) for wave in wextRed ] )

            w = append( w, wextRed )
            f = append( f, fextRed )
        
        for i in range( len( w ) ) :
            print >> fout, "%7.2f  %10.2f  %12.7e"%( d[0], w[i], f[i] )
    fout.close() 

    return( newsedfile )



def extrapolatesed_linfit(sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, 
                          Npt=2, Nsmooth=0, forceslope=True, verbose=False ):
    """ use a linear fit of the first/last Npt  points on the SED
    to extrapolate to the blue and red.  

    Set Nsmooth to a positive integer to set the window size for median 
    smoothing of the input spectrum before the linear fitting. 
    
    With forceslope=True, check if the linear fit returns a slope that would 
    send the tail upward (i.e. a positive derivative on the red side, negative 
    on the  blue side).  If so, then re-do the fit using just the first/last 
    point and the peak flux point, in order to guarantee a downward sloping tail.
    """

    from scipy import interpolate as scint
    from scipy import stats
    import shutil

    medsmooth = lambda f,N : array( [ median( f[max(0,i-N):min(len(f),max(0,i-N)+2*N)]) for i in range(len(f)) ] )
    
    dlist,wlist,flist = getsed( sedfile ) 
    dlistnew, wlistnew, flistnew = [],[],[]

    fout = open( newsedfile, 'w' )
    for i in range( len(dlist) ) : 
        d,w,f = dlist[i],wlist[i],flist[i]

        if Nsmooth : ffit = medsmooth( f, Nsmooth )
        else : ffit = f
        wavestep = w[1] - w[0]

        # blueward linear extrapolation from first N points
        wN = w[:Npt]
        fN = ffit[:Npt]
        (a,b,rval,pval,stderr)=stats.linregress(wN,fN)
        if forceslope and a < 0 : 
            # re-do blueward linear extrapolation using the peak
            # in order to guarantee a downward slope
            print( "  re-fitting %s day %i to ensure downward slope to the blue"%(sedfile,d[0]))
            wN = [ w[0], w[ ffit.argmax() ] ]
            fN = [ f[0], ffit.max() ]
            (a,b,rval,pval,stderr)=stats.linregress(wN,fN)
        Nbluestep = len( arange( minwave, w[0], wavestep ) )
        wextBlue = sorted( [ w[0] -(i+1)*wavestep for i in range(Nbluestep) ] )
        fextBlue = array( [ max( 0, a * wave + b ) for wave in wextBlue ] )

        # redward linear extrapolation from last N points
        wN = w[-Npt:]
        fN = ffit[-Npt:]
        (a,b,rval,pval,stderr)=stats.linregress(wN,fN)
        if forceslope and a > 0 : 
            # re-do redward  linear extrapolation using the peak
            # in order to guarantee a downward slope
            if verbose : print( "  re-fitting %s day %i to ensure downward slope to the red"%(sedfile, d[0]) )
            wN = [ w[ ffit.argmax() ], w[-1] ]
            fN = [ ffit.max(), ffit[-1],  ]
            (a,b,rval,pval,stderr)=stats.linregress(wN,fN)
        Nredstep = len( arange( w[-1], maxwave,  wavestep ) )
        wextRed =  sorted( [ w[-1] + (i+1)*wavestep for i in range(Nredstep) ] )
        fextRed = array( [ max( 0, a * wave + b ) for wave in wextRed ] )

        wnew = append( append( wextBlue, w ), wextRed )
        fnew = append( append( fextBlue, f ), fextRed )
        # dnew = zeros( len(wnew) ) + d[0]
        
        for i in range( len( wnew ) ) :
            print >> fout, "%7.2f  %10.2f  %12.7e"%( d[0], wnew[i], fnew[i] )
    fout.close() 

    return( newsedfile )


def extrapolatesed_flatline(sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE ):
    """ extrapolate to the red and the blue using a flatline. i.e. the 
    extrapolated flux is fixed to the endpoint values
    """
    from scipy import interpolate as scint
    from scipy import stats
    import shutil
    
    dlist,wlist,flist = getsed( sedfile ) 
    dlistnew, wlistnew, flistnew = [],[],[]

    fout = open( newsedfile, 'w' )
    for i in range( len(dlist) ) : 
        d,w,f = dlist[i],wlist[i],flist[i]

        wavestep = w[1] - w[0]
        # blueward flatline extrapolation from first point
        Nbluestep = len( arange( minwave, w[0], wavestep ) )
        wextBlue = sorted( [ w[0] -(i+1)*wavestep for i in range(Nbluestep) ] )
        fextBlue = array( [ f[0] for wave in wextBlue ] )

        # redward flatline extrapolation from last point
        Nredstep = len( arange( w[-1], maxwave,  wavestep ) )
        wextRed =  sorted( [ w[-1] + (i+1)*wavestep for i in range(Nredstep) ] )
        fextRed = array( [ f[-1]  for wave in wextRed ] )

        wnew = append( append( wextBlue, w ), wextRed )
        fnew = append( append( fextBlue, f ), fextRed )
        # dnew = zeros( len(wnew) ) + d[0]
        
        for i in range( len( wnew ) ) :
            print >> fout, "%5.1f  %10i  %12.7e"%( d[0], wnew[i], fnew[i] )
    fout.close() 

    return( newsedfile )


def extendNon1a(origdir = "snsed/non1a.ORIG", verbose=True ):
    """ Extrapolate each of the Non-Ia template sed files,
    preserving a pristine copy in $SNDATA_ROOT/snsed/non1a.ORIG
    """
    import glob
    import shutil

    origdir = os.path.join(sndataroot, origdir)
    non1adir = os.path.join(sndataroot, "snsed/non1a")
    if not os.path.isdir( origdir ) :
        os.rename( non1adir, origdir ) 
    if not os.path.isdir( non1adir ) :
        os.mkdir( non1adir )

    otherstufflist = glob.glob("%s/*.DAT"%(origdir)) + glob.glob("%s/*.LIST"%(origdir)) + glob.glob("%s/*.INPUT"%(origdir))
    for otherfile in otherstufflist : 
        newfile =  os.path.join( non1adir, os.path.basename( otherfile ) )
        shutil.copy( otherfile, newfile )
        
    sedlist = glob.glob("%s/*.SED"%origdir)
    for sedfile in sedlist : 
        newsedfile =  os.path.join( non1adir, os.path.basename( sedfile ) )

        if 'SDSS-012842' in sedfile or 'SDSS-013449' in sedfile : 
            # These are already super-smooth blackbodies for the IIn templates
            # 2-pt linear extrapolation is best.
            Npt = 2
            Nsmooth=0
            print("EXTRAPOLATING %s\n    ==> %s"%(sedfile, newsedfile) )
            extrapolatesed_linfit(sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, 
                                  Npt=Npt, Nsmooth=Nsmooth, forceslope=True  )
            print("     Done with %s.\a\a\a"%newsedfile)


        else : 
            # for all other templates, we use a fixed blue anchor point 
            # and on the red side we fit a lot of points so we don't 
            # get thrown around by sharp spectral features
            Nred = 100
            Nsmooth = 20

            print("EXTRAPOLATING %s\n    ==> %s"%(sedfile, newsedfile) )
            extrapolatesed( sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, 
                            Bextwave=2000, Bfrac=0.2, Brefwave=2800, 
                            Nred = 100, Nsmooth=20, verbose=verbose )
            print("     Done with %s.\a\a\a"%newsedfile)



def extendSALT2_temp0( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                       salt2srcdir = 'models/SALT2/SALT2.Guy10_LAMOPEN', 
                       tailsedfile = 'snsed/Hsiao07.extrap.dat',
                       wjoinblue = 2800, wjoinred = 8500 ,
                       wmin = MINWAVE, wmax = MAXWAVE ):
    """ extend the salt2 Template_0 model component 
    by adopting the UV and IR tails from another SED model. 
    The default is to use SR's extrapolated modification 
    of the Hsiao 2007 sed model, scaled and joined at the 
    wjoin wavelengths, and extrapolated out to wmin and wmax. 
    """
    import shutil
    sndataroot = os.environ['SNDATA_ROOT']
 
    salt2dir = os.path.join( sndataroot, salt2dir ) 
    salt2srcdir = os.path.join( sndataroot, salt2srcdir ) 
    
    temp0fileIN = os.path.join( salt2srcdir, 'salt2_template_0.dat' ) 
    temp0fileOUT = os.path.join( salt2dir, 'salt2_template_0.dat' ) 
    temp0dat = getsed( sedfile=temp0fileIN ) 

    tailsedfile = os.path.join( sndataroot, tailsedfile ) 

    taildat = getsed( sedfile=tailsedfile ) 
    
    dt,wt,ft = loadtxt( tailsedfile, unpack=True ) 
    taildays = unique( dt ) 

    # build up modified template from day -20 to +50
    outlines = []
    for i in range( 71 ) : 
        thisday = i - 20

        # get the tail SED for this day
        it = where( taildays == thisday )[0]
        dt = taildat[0][it]
        wt = taildat[1][it]
        ft = taildat[2][it]

        #if thisday > 50 : 
        #    d0new = dt
        #    w0new = wt
        #    f0new = ft * (bluescale+redscale)/2.
        #else : 

        # get the SALT2 template SED for this day
        d0 = temp0dat[0][i]
        w0 = temp0dat[1][i]
        f0 = temp0dat[2][i]
        print( 'splicing tail onto template for day : %i'%thisday )

        i0blue = argmin(  abs(w0-wjoinblue) )
        itblue = argmin( abs( wt-wjoinblue))

        i0red = argmin(  abs(w0-wjoinred) )
        itred = argmin( abs( wt-wjoinred))

        itmin = argmin( abs( wt-wmin))
        itmax = argmin( abs( wt-wmax))

        bluescale = f0[i0blue]/ft[itblue] 
        redscale = f0[i0red]/ft[itred] 

        d0new = dt.tolist()[itmin:itblue] + d0.tolist()[i0blue:i0red-1] + dt.tolist()[itred:itmax+1]
        w0new = wt.tolist()[itmin:itblue] + w0.tolist()[i0blue:i0red-1] + wt.tolist()[itred:itmax+1]
        f0new = (bluescale*ft).tolist()[itmin:itblue] + f0.tolist()[i0blue:i0red-1] + (redscale*ft).tolist()[itred:itmax+1]

        # plot it
        clf()
        plot( w0, f0, ls='-',color='b', lw=1, label='Input SALT2 Model')
        plot( wt, (bluescale+redscale)/2. * ft, ls=':',color='r', lw=1, label='SED model')
        plot( w0new, f0new, ls='--',color='k', lw=2, label='Extrapolated SALT2 model')
        legend()
        draw()
        #raw_input('return to continue')

        # append to the list of output data lines
        for j in range( len( d0new ) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    d0new[j], w0new[j], f0new[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp0fileOUT, 'w' ) 
    fout.writelines( outlines ) 
    fout.close() 
        


def extendSALT2_temp1( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                       salt2srcdir = 'models/SALT2/SALT2.Guy10_LAMOPEN', 
                       # wjoinblue = 2000, wjoinred = 8500 ,
                       wjoinblue = None, wjoinred = None,
                       wmin = MINWAVE, wmax = MAXWAVE,
                       wstep = 10 ):
    """ extend the salt2 Template_1 model component 
    with a flat line at 0 to the blue and to the red.
    If join wavelengths are not provided, uses the 
    end points of the input spectrum
    """
    import shutil
    sndataroot = os.environ['SNDATA_ROOT']
 
    salt2dir = os.path.join( sndataroot, salt2dir ) 
    salt2srcdir = os.path.join( sndataroot, salt2srcdir ) 
    
    temp1fileIN = os.path.join( salt2srcdir, 'salt2_template_1.dat' ) 
    temp1fileOUT = os.path.join( salt2dir, 'salt2_template_1.dat' ) 
    temp1dat = getsed( sedfile=temp1fileIN ) 

    # build up modified template from day -20 to +50
    outlines = []
    for i in range( 71 ) : 
        thisday = i - 20

        # get the SALT2 template SED for this day
        d1 = temp1dat[0][i]
        w1 = temp1dat[1][i]
        f1 = temp1dat[2][i]
        print( 'extrapolating with flatline onto template for day : %i'%thisday )
        
        if wjoinblue==None : wjoinblue=w1[0]
        if wjoinred==None : wjoinred=w1[-1]

        i1blue = argmin(  abs(w1-wjoinblue) )
        i1red = argmin(  abs(w1-wjoinred) )
        
        Nblue = int((wjoinblue-wmin )/wstep + 1)
        Nred = int((wmax -wjoinred )/wstep + 1)

        d1new =  (ones(Nblue)*thisday).tolist() + d1.tolist()[i1blue+1:i1red-1] + (ones(Nred)*thisday).tolist()
        w1new = arange(wmin,wmin+Nblue*wstep,wstep).tolist() + w1.tolist()[i1blue+1:i1red-1] + arange(wjoinred,wjoinred+Nred*wstep,wstep).tolist()
        f1new = zeros(Nblue).tolist() + f1.tolist()[i1blue+1:i1red-1] + zeros(Nred).tolist()

        # plot it
        clf()
        plot( w1, f1, ls='-',color='r', lw=1)
        plot( w1new, f1new, ls='--',color='k', lw=2)
        draw()
        #raw_input('return to continue')

        # append to the list of output data lines
        for j in range( len( d1new ) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    d1new[j], w1new[j], f1new[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp1fileOUT, 'w' ) 
    fout.writelines( outlines ) 
    fout.close() 
        


def extendSALT2_flatline( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                          salt2srcdir = 'models/SALT2/SALT2.Guy10_LAMOPEN', 
                          wjoinblue = 2000, wjoinred = 8500 ,
                          wmin = MINWAVE, wmax = MAXWAVE,
                          wstep = 10, showplots=False ):
    """ extrapolate the *lc* and *spec* .dat files for SALT2
    using a flatline to the blue and red """

    sndataroot = os.environ['SNDATA_ROOT']
    salt2dir = os.path.join( sndataroot, salt2dir ) 
    salt2srcdir = os.path.join( sndataroot, salt2srcdir ) 
    
    filelist = ['salt2_lc_dispersion_scaling.dat',
                'salt2_lc_relative_covariance_01.dat',
                'salt2_lc_relative_variance_0.dat',
                'salt2_lc_relative_variance_1.dat',
                'salt2_spec_covariance_01.dat',
                'salt2_spec_variance_0.dat',
                'salt2_spec_variance_1.dat']
    
    #for filename in  ['salt2_lc_dispersion_scaling.dat']: 
    #for filename in  ['salt2_lc_relative_covariance_01.dat']:
    for filename in filelist : 
        infile = os.path.join( salt2srcdir, filename )
        outfile = os.path.join( salt2dir, filename )

        newsedfile = extrapolatesed_flatline( infile, outfile, minwave=wmin, maxwave=wmax,
                                              wjoinblue=wjoinblue, wjoinred=wjoinred )
        
        # plot it
        if showplots: 
            #for d in range(-20,50) : 
            for d in [-10,-5,0,5,10,15,20,25,30,35,40,45,50] : 
                clf()
                plotsed( infile, day=d, ls='-',color='r', lw=1) 
                plotsed( outfile,day=d, ls='--',color='k', lw=2)
                print( '%s : day %i'%(filename,d) )
                draw()
                # raw_input('%s : day %i.  return to continue'%(filename,d)) 

    

def plotAll(): 
    """ 
    read in and plot all the non1a template SEDs at peak, 
    in three panels, with the Ia template for comparison,
    """
    ioff()
    ax1 = subplot( 131 )
    # plot the type II-L in purple
    plotsed( os.path.join(sndataroot,'snsed/non1a/Nugent+Scolnic_IIL.SED'), day=0, normalize=2, color='b', lw=0.5, label='II-L' )
    # plot the type IIn's in orange 
    for sed in ['SDSS-012842','SDSS-013449'] : 
        plotsed( os.path.join(sndataroot,'snsed/non1a/%s.SED'%sed), day=0.5, normalize=2, color='b', lw=0.5, label=sed )
    # Ia in red
    plotsed( os.path.join(sndataroot,'snsed/Hsiao07.extrap.dat'), day=0, normalize=2, color='r', label='Ia (Hsiao)', lw=2 )
    plotsed( os.path.join(sndataroot,'snsed/Hsiao07.dat'), day=0, normalize=2, color='k', label='Ia (Hsiao)', lw=1, ls=':' )
    title('Type II-L and IIn Templates')
    #legend()
    draw()

    subplot( 132, sharex=ax1, sharey=ax1 )
    # plot the type IIP's in blue 
    IIpsedlist = [ 'SDSS-000018','SDSS-003818','SDSS-013376','SDSS-014450','SDSS-014599','SDSS-015031','SDSS-015320',
                   'SDSS-015339','SDSS-017564','SDSS-017862','SDSS-018109','SDSS-018297','SDSS-018408','SDSS-018441',
                   'SDSS-018457','SDSS-018590','SDSS-018596','SDSS-018700','SDSS-018713','SDSS-018734','SDSS-018793',
                   'SDSS-018834','SDSS-018892','SDSS-020038' ]
    for sed in IIpsedlist :
        plotsed( os.path.join(sndataroot,'snsed/non1a/%s.SED'%sed), day=0.5, normalize=2, color='b', lw=0.5, label=sed )
    # Ia in red
    plotsed( os.path.join(sndataroot,'snsed/Hsiao07.extrap.dat'), day=0, normalize=2, color='r', label='Ia (Hsiao)', lw=2 )
    plotsed( os.path.join(sndataroot,'snsed/Hsiao07.dat'), day=0, normalize=2, color='k', label='Ia (Hsiao)', lw=1, ls=':' )
    #legend()
    title('Type II-P Templates')
    draw()

    subplot( 133, sharex=ax1, sharey=ax1  )
    Ibclist =  [ 'CSP-2004gv','CSP-2006ep','CSP-2007Y','SDSS-000020','SDSS-002744','SDSS-014492',
                 'SDSS-019323','SNLS-04D1la','SNLS-04D4jv','CSP-2004fe','CSP-2004gq','SDSS-004012',
                 'SDSS-013195','SDSS-014475','SDSS-015475','SDSS-017548' ]
    for sed in Ibclist :
        plotsed( os.path.join(sndataroot,'snsed/non1a/%s.SED'%sed), day=0.5, normalize=2, color='g', lw=0.5, label=sed )
    # Ia in red
    plotsed( os.path.join(sndataroot,'snsed/Hsiao07.extrap.dat'), day=0, normalize=2, color='r', label='Ia (Hsiao)', lw=2 )
    plotsed( os.path.join(sndataroot,'snsed/Hsiao07.dat'), day=0, normalize=2, color='k', label='Ia (Hsiao)', lw=1, ls=':' )
    title('Type Ib and Ic Templates', color='g')
    #legend()
    ion()

    suptitle( 'Comparing SNANA CC and Ia templates at peak' )
    draw()




def extratest( sed, **kwarg ) :
    sedorig = '/usr/local/SNDATA_ROOT/snsed/non1a.ORIG/'+sed+'.SED'
    sednew  = '/usr/local/SNDATA_ROOT/snsed/non1a/'+sed+'test.SED'
    extrapolatesed_linfit( sedorig, sednew, **kwarg )
    clf()
    plotsed( sednew, day=0.5, color='g', ls='-', lw=1, normalize=2, label='new'  )
    plotsed( sedorig, day=0.5, color='b', ls='-', lw=3, normalize=2, label='orig'  )

    plotsed( '/usr/local/SNDATA_ROOT/snsed/Hsiao07.extrap.dat', day=0, normalize=2, color='r', label='Ia (Hsiao)', lw=2 )

    ax = gca()
    text( 0.1,0.95, sed, ha='left',va='top', transform=ax.transAxes )
    legend( )

    
