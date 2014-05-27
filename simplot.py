#! /usr/bin/env python
# S.Rodney
# 2012.05.03
#  plotting magnitudes, colors, redshifts, etc. from a SNANA simulation table.


import pyfits
import os 
import exceptions
import numpy as np
import pylab as p
import time
import glob
from constants import SNTYPEDICT


BANDORDER = 'STUCBGVRXIZWLYMJOPNQH'
BANDORDER_RED2BLUE = 'HQNPOJMYLWZIXRVGBCUTS'




def timestr( time ) : 
    """ convert an int time into a string suitable for use 
    as an object parameter, replacing + with p and - with m
    """
    if time>=0 : pm = 'p'
    else : pm = 'm'
    return( '%s%02i'%(pm,abs(time) ) )


def plotNsim( sim, zbinsplot=np.arange(0,3.,0.1),
              zbinscount=[0,0.5,1.0,1.5,2.0,2.5,3.0] ) :
    """ from the .DUMP output, read in the distribution of simulated
    SNe vs z (before detection cuts) and the distribution of simulated
    detections vs z (after cuts).  From the .README file, get the
    total number of SNe "per season" (based on the survey volume,
    redshift range, etc).

    Normalize the redshift distributions to integrate to unity, then
    multiply by the total number of SNe to get the number of SNe in
    each redshift bin.

    If the simulation used a constant volumetric rate, then this
    number of SNe vs z can be used as the denominator for a rates
    calculation.
    """

    # Distribution of all SNe exploding in this volume
    # (the total number here depends on the user-defined NGEN_LC
    #  or NGENTOT_LC values)
    Nall,zall = np.histogram( sim.DUMP['REDSHIFT'], bins=zbinsplot )

    # Distribution of all detected SNe
    idet = sim.DUMP['idet']
    Ndet,zdet = np.histogram( sim.DUMP['REDSHIFT'][idet], bins=zbinsplot )

    # normalize the distributions and scale by NSNSURVEY,
    # which is the number of SN events in the survey volume/time
    normfactor = float(sim.NSNSURVEY) / Nall.sum()
    Nallsurvey = Nall * normfactor
    Ndetsurvey = Ndet * normfactor

    p.clf()
    p.plot( zbinsplot[1:], Nallsurvey, drawstyle='steps-post', color='b', label=r'%i total SNe in survey volume \& time'%sim.NSNSURVEY )
    p.plot( zbinsplot[1:], Ndetsurvey, drawstyle='steps-post', color='r', label='%i detectable SNe'%Ndetsurvey.sum() )
    p.xlabel('redshift')
    p.ylabel('Number of SNe')
    p.legend( loc='upper left', frameon=False, numpoints=3, borderpad=0.2)

    # ------------------------------------------------------------
    # Count up simulated detection counts 
    Ndet,zdet = np.histogram( sim.DUMP['REDSHIFT'][idet], bins=zbinscount )
    Ndetcount = Ndet * normfactor

    print( "   zrange   NSIM_DET" )
    for i in range(len(zbinscount)-1) : 
        print( "%.2f  %.2f   %.3f"%( zbinscount[i], zbinscount[i+1], Ndetcount[i] ) )
    return( None )


def plotColorMag( sim, color='W-H', mag='H', mjdrange=None, 
                  plotstyle='contour', tsample=5.0, Nbins=None, 
                  binrange = None, linelevels = [ 0.68, 0 ], 
                  snmags={}, classfractions = [0.24, 0.19, 0.57],
                  histbinwidth = 0.2, verbose=False, **kwargs ):
    return( plotSimClouds( sim, xaxis=color, yaxis=mag, mjdrange=mjdrange, 
                           plotstyle=plotstyle, tsample=tsample, Nbins=Nbins, 
                           binrange = binrange, linelevels = linelevels,
                           snmags=snmags, classfractions = classfractions,
                           histbinwidth = histbinwidth, verbose=verbose, 
                           **kwargs )    )

def plotColorColor( sim, color1='W-H', color2='J-H', mjdrange=None, 
                    plotstyle='contour', tsample=5.0, Nbins=None, 
                    binrange = None, linelevels = [ 0.68, 0 ], 
                    snmags={}, classfractions = [0.24, 0.19, 0.57],
                    histbinwidth = 0.2, verbose=False, **kwargs ):
    return( plotSimClouds( sim, xaxis=color1, yaxis=color2, mjdrange=mjdrange, 
                           plotstyle=plotstyle, tsample=tsample, Nbins=Nbins, 
                           binrange = binrange, linelevels = linelevels,
                           snmags=snmags, classfractions = classfractions,
                           histbinwidth = histbinwidth, verbose=verbose, 
                           **kwargs )    )

def plotSimClouds( sim, xaxis='W-H', yaxis='H', mjdrange=None, tsample=5.0, 
                   plotstyle='contourf', Nbins=None, binrange = None, linelevels=[0.95, 0.50, 0], 
                   sidehist=True, snmags={}, classfractions=[0.24,0.19,0.57],
                   histbinwidth = 0.2, verbose=False, debug=False, **kwargs ):
    """ construct a mag-mag, color-color or color-mag plot. 

    xaxis : the quantity to plot along the x axis. 
    yaxis : ditto for the y axis. 
      These are strings, and either may be a color ('W-H') or a magnitude ('H')
    mjdrange : range of MJD dates to plot  [defaults to the mean peak mjd -30 and +60]]
    tsample : spacing between simulated light curve sample points (obs-frame days)
    plotstyle : 'contourf' ,  'contour', 'points', 'contourp' (contour lines + points)
    Nbins : number of bins (along each axis) for 2-D histograms in contour plots
            if Nbins==None, then it is set automatically based on the number of 
            SNe in the simulation.
    linelevels :fraction of the population to enclose in the solid line contours
    sidehist : show 1-d histograms along the top and right side  
      (when True, we use a whole Figure instance. If False, we use only the current axis,
      allowing the calling program to embed the main plot into a subplot axes instance)
    """ 
    from math import sqrt, pi
    from matplotlib import cm
    from matplotlib.patches import FancyArrowPatch
    from hstsnpipe.tools.figs import colors

    if mjdrange==None : 
        mjdpkmean = np.mean( sim.SIM_PEAKMJD ) 
        zmean = np.mean( sim.SIM_REDSHIFT ) 
        mjdrange = [ mjdpkmean - 30*(1+zmean), mjdpkmean + 60*(1+zmean) ]

    if sidehist : 
        ax1 = p.axes( [0.12,0.12,0.68,0.68])
        ax2 = p.axes( [0.12,0.8,0.68,0.13], sharex=ax1 )
        ax3 = p.axes( [0.8,0.12,0.13,0.68], sharey=ax1 )
    else : 
        ax1 = p.gca()

    # For now, assume that all SNe in the sim are of the same type
    sntype = SNTYPEDICT[ sim.SNTYPE[0] ]

    # Set up the default plot colors based on SN type
    plotdefaults = {'ls':' ','marker':'o','mew':0.2,'ms':5,'alpha':0.1 } 
    if sntype in ['II','IIn','IIP','IIL'] : 
        plotdefaults['mfc'] = colors.lightblue
        plotdefaults['mec'] = colors.darkblue
        plotdefaults['color'] = colors.darkblue
        histcolor=colors.darkblue
        cmap = cm.Blues_r
        cfrac=classfractions[2]
    elif sntype in ['Ib','Ic','Ibc'] : 
        plotdefaults['mfc'] = colors.khaki
        plotdefaults['mec'] = colors.olivegreen
        plotdefaults['color'] = colors.olivegreen
        histcolor=colors.green
        cmap = cm.Greens_r
        cfrac=classfractions[1]
    elif sntype == 'Ia': 
        plotdefaults['mfc'] = colors.pink
        plotdefaults['mec'] = colors.maroon
        plotdefaults['color'] = colors.maroon
        histcolor=colors.maroon
        cmap = cm.Reds_r
        cfrac=classfractions[0]
    else :
        plotdefaults['mfc'] = 'k'
        plotdefaults['mec'] = 'k'
        plotdefaults['color'] = 'black'
        histcolor='k'
        cmap = cm.Greys
        cfrac=1.0
    plotargs = dict( plotdefaults.items() + kwargs.items() )
  
    # sample magnitudes at intervals across the range of observation
    # days (mjdrange) using the given sampling spacing (tsample) 
    if xaxis.find('-')>0: 
        band1, band2 = xaxis.split('-') 
    else : 
        band1, band2 = xaxis,xaxis
    if yaxis.find('-')>0:
        band3, band4 = yaxis.split('-') 
    else : 
        band3, band4 = yaxis,yaxis
    mag1, mag2, mag3, mag4  = [], [], [], []

    for mjd in np.arange( mjdrange[0], mjdrange[1]+tsample, tsample ): 
        # sample the light curves at the given MJD(s)
        sim.samplephot( mjd, tmatch=tsample )
        m1 = sim.__dict__['%s%i'%(band1, int(mjd))]
        m2 = sim.__dict__['%s%i'%(band2, int(mjd))]
        m3 = sim.__dict__['%s%i'%(band3, int(mjd))]
        m4 = sim.__dict__['%s%i'%(band4, int(mjd))]
        # limit to observations with legit data
        igood = np.where( (m1<90) & (m1>-90) & 
                          (m2<90) & (m2>-90) &
                          (m3<90) & (m3>-90) &
                          (m4<90) & (m4>-90) )[0]
        mag1 += m1[igood].tolist()
        mag2 += m2[igood].tolist()
        mag3 += m3[igood].tolist()
        mag4 += m4[igood].tolist()

    if not len(mag1) : 
        print( "ERROR: no good mags for one of  %s"%(''.join(np.unique([band1,band2,band3,band4]))))
        if debug : import pdb; pdb.set_trace()
        return( None ) 
    mag1 = np.array( mag1 ) 
    mag2 = np.array( mag2 )
    mag3 = np.array( mag3 )
    mag4 = np.array( mag4 )
    if band1==band2 : xarray = mag1
    else : xarray = mag1-mag2
    if band3==band4 : yarray = mag3
    else : yarray = mag3-mag4

    if plotstyle == 'points' or plotstyle == 'contourp':
        ax1.plot( xarray, yarray, **plotargs )
        if verbose : 
            print '%.f Type %s SNe Simulated'%(len(xarray),sntype)
            print 'Sampled every %.f days (observed frame)'%tsample

    if not binrange : 
        # Set the range for binning (to make contours) 
        # ensuring that all relevant SNe are included            
        if band1==band2 : 
            xbinlowlim, xbinhighlim = 18, 34
            if 'SNLS' in sim.simname : xbinlowlim, xbinhighlim = 15, 28
        else : xbinlowlim, xbinhighlim = -10, 10
        if band3==band4 : 
            ybinlowlim,ybinhighlim = 18, 34
            if 'SNLS' in sim.simname : ybinlowlim,ybinhighlim = 15, 28
        else : ybinlowlim,ybinhighlim = -10, 10
        xbinlow = max(xbinlowlim, min(xarray)-0.5) 
        xbinhigh = min(xbinhighlim, max(xarray)+0.5)
        ybinlow = max(ybinlowlim,min(yarray)-0.5)
        ybinhigh = min(ybinhighlim,max(yarray)+0.5)
        binrange = [[xbinlow,xbinhigh],[ybinlow,ybinhigh]]

    # Plot filled contours, showing  the full extent of the population,
    # and contour lines containing 68% of the population.
    # First, bin the points into a 2-d histogram:
    # (Note that we reverse the x-y order here to get the binned arrays 
    #  plotted in the correct direction )
    if not Nbins : Nbins = int( sqrt( sim.nsim  )/2 ) 
    count,y,x = p.histogram2d( yarray, xarray, bins=Nbins, range=[binrange[1],binrange[0]] )

    # Renormalize relative to the sum of all SNe in this class : 
    count /= count.sum()
        
    # Now set up an array 'cabove' such that  the cell value in cabove[i,j] 
    # is equal to the sum of all cells that have a value higher than c[i,j]  
    cabove = scumsum( count )
        
    if plotstyle.startswith('contour') : 
        # solid lines give probability contours at specified levels
        # (defaults to 0.68 for "1-sigma contours")
        ax1.contour( x[:-1], y[:-1], cabove, linelevels, colors=[plotargs['color'],plotargs['color']], ls='-' )

    if plotstyle=='contourf' :
        #flevels = [ 1e-30, 0 ]
        # filled contours show full extent of the population
        #ax1.contourf( x[:-1], y[:-1], count, flevels, cmap=cmap, alpha=0.5 )
        ax1.contourf( x[:-1], y[:-1], cabove, levels=linelevels, colors=[plotargs['mec'],plotargs['mfc']], alpha=0.5, extend='neither' )

    filt1 = sim.SURVEYDATA.band2filter(band1)
    filt2 = sim.SURVEYDATA.band2filter(band2)
    filt3 = sim.SURVEYDATA.band2filter(band3)
    filt4 = sim.SURVEYDATA.band2filter(band4)
    if band1==band2 : ax1.set_xlabel('%s'%filt1)
    else : ax1.set_xlabel('%s - %s'%(filt1,filt2))
    if band3==band4 : ax1.set_ylabel('%s'%(filt3))
    else : ax1.set_ylabel('%s - %s'%(filt3,filt4))

    if sidehist : 
        # construct the 1-d histograms along the edges
        histbinsX =  np.arange(binrange[0][0]-histbinwidth,binrange[0][1]+histbinwidth, histbinwidth)
        histbinsY = np.arange(binrange[1][0]-histbinwidth,binrange[1][1]+histbinwidth, histbinwidth)
        histbincentersY = histbinsY[:-1] + (histbinsY[1]-histbinsY[0])/2.
        histbincentersX = histbinsX[:-1] + (histbinsX[1]-histbinsX[0])/2.

        histY, edge = p.histogram( yarray, bins=histbinsY )
        histX, edge = p.histogram( xarray, bins=histbinsX )

        Nsample = len(mag1)
        ax2.plot( histbincentersX,  cfrac*histX/Nsample, color=histcolor, ls='-', drawstyle='steps-mid' ) 
        ax2.xaxis.set_ticks_position('top')
        ymin2,ymax2 = ax2.get_ylim()
        ax2.set_yticks( np.round( np.linspace( ymin2, ymax2, 4), 2 )[1:] )

        ax3.plot( cfrac*histY/Nsample, histbincentersY, color=histcolor,ls='-', drawstyle='steps-mid' )
        ax3.yaxis.set_ticks_position('right')
        xmin3,xmax3 = ax3.get_xlim()
        ax3.set_xticks( np.round( np.linspace( xmin3, xmax3, 4), 2 )[1:] )

    # If SN magnitudes were provided, then plot the observations with error bars
    xmin,xmax = xarray.min()-1.0,xarray.max()+1.0
    ymin,ymax = yarray.max()+3.0,yarray.min()-1.0
    likelihood = 0.0
    if ( band1 in snmags and band2 in snmags and 
         band3 in snmags and band4 in snmags ) : 
        if band1==band2 : snx = abs(snmags[band1])
        else : snx = abs(snmags[band1])-abs(snmags[band2])
        if band3==band4 : sny = abs(snmags[band3])
        else : sny = abs(snmags[band3])-abs(snmags[band4])

        # compute the likelihood value of the position where the observed 
        # SN magnitudes land: the likelihood that the SN belongs to the 
        # simulated class, based on the observed data alone
        isnx = np.argmin( np.abs( x-snx ) )
        isny = np.argmin( np.abs( y-sny ) )
        try: 
            likelihood = 1 - cabove[ isnx ][ isny ]
        except : 
            likelihood = 0.0

        if ( 'd'+band1 in snmags.keys() and 'd'+band2 in snmags.keys() and
             'd'+band3 in snmags.keys() and 'd'+band4 in snmags.keys() ) :
            dsnx1,dsnx2 = snmags['d'+band1], snmags['d'+band2]
            dsny3,dsny4 = snmags['d'+band3], snmags['d'+band4]
            if band1==band2 : dsnx = dsnx1
            else : dsnx = np.sqrt( dsnx1**2 + dsnx2**2 )
            if band3==band4 : dsny = dsny3
            else : dsny = np.sqrt( dsny3**2 + dsny4**2 )
            
            # plot upper-limit arrow(s) as needed
            if band1==band2 and dsnx < 0 : 
                arr = FancyArrowPatch( [snx,sny], [snx+1.5,sny], arrowstyle='-|>', mutation_scale=25, fc='k', ls='dashed' )
                ax1.add_patch( arr )
            if band1!=band2 and dsnx1 < 0 : 
                arr = FancyArrowPatch( [snx,sny], [snx+1.5,sny], arrowstyle='-|>', mutation_scale=25, fc='k', ls='dashed' )
                ax1.add_patch( arr )
            if band1!=band2 and dsnx2 < 0 : 
                arr = FancyArrowPatch( [snx,sny], [snx-1.5,sny], arrowstyle='-|>', mutation_scale=25, fc='k', ls='dashed' )
                ax1.add_patch( arr )
            if band3==band4 and dsny < 0 : 
                arr = FancyArrowPatch( [snx,sny], [snx,sny+1.5], arrowstyle='-|>', mutation_scale=25, fc='k', ls='dashed' )
                ax1.add_patch( arr )
            if band3!=band4 and dsny3 < 0 : 
                arr = FancyArrowPatch( [snx,sny], [snx,sny+1.5], arrowstyle='-|>', mutation_scale=25, fc='k', ls='dashed' )
                ax1.add_patch( arr )
            if band3!=band4 and dsny4 < 0 : 
                arr = FancyArrowPatch( [snx,sny], [snx,sny-1.5], arrowstyle='-|>', mutation_scale=25, fc='k', ls='dashed' )
                ax1.add_patch( arr )
                
            # plot the point and error
            if dsnx1>0 and dsnx2>0 : 
                ax1.errorbar(  snx, sny, 0, abs(dsnx), color='k', marker='o', mec='k',mfc='w', mew=1.5, elinewidth=1.5, alpha=1.0, capsize=0, zorder=10  )
            if dsny3>0 and dsny4>0 : 
                ax1.errorbar(  snx, sny, abs(dsny), 0, color='k', marker='o', mec='k',mfc='w', mew=1.5, elinewidth=1.5, alpha=1.0, capsize=0, zorder=10  )
               
        else : 
            ax1.plot( snx, sny, color='k', marker='o', zorder=10  )
        if sidehist: 
            ax3.axhline( sny, color='0.5', lw=1, ls='-', zorder=10)
            ax2.axvline( snx, color='0.5', lw=1, ls='-', zorder=10)
            
        # ensure that the axes ranges include our SN observation
        if sny > ymin: ymin = sny + 1
        if sny < ymax: ymax = sny - 1
        if snx < xmin: xmin = snx - 1
        if snx > xmax: xmax = snx + 1

    ax1.set_xlim(binrange[0])
    ax1.set_ylim(binrange[1])
    if band1==band2 : 
        if not ax1.xaxis_inverted() : ax1.invert_xaxis()
        if sidehist:
            if not ax2.xaxis_inverted() : ax2.invert_xaxis()
    if band3==band4 : 
        if not ax1.yaxis_inverted() : ax1.invert_yaxis()
        if sidehist : 
            if not ax3.yaxis_inverted() : ax3.invert_yaxis()
    return( ax1, likelihood )


def plot_mag_z( sim, band='H', mjd='peak', plotstyle='median',
                restbands=False, detlim=False, **kwargs ):
    """ plot the magnitudes against redshift for the given MJD
    mjd='peak' is a special case that samples all simulated SNe
    at their respective peaks.  Otherwise we sample all at the 
    same MJD, which probably means they are at different LC ages.

    If restbands == True, show the rest-frame 
    band-pass contribution fractions at each 
    redshift
    detlim : plot a dashed line at the detection limit ~25.5
    """ 
    z = sim.z
    if mjd in [ None, 0, 'pk','peak'] : 
        # read in the peak mags
        mag = sim.__dict__['SIM_PEAKMAG_'+band]
    else  : 
        # sample the light curves at the given MJD date
        sim.samplephot( mjd )
        mag = sim.__dict__['%s%i'%(band, int(mjd))]

    # limit to observations with legit data
    igood = np.where( (mag<99) & (mag>-99) )[0]
    if not len(igood) : 
        print( "ERROR: no good mags for %s vs z"%(band))
        return( None ) 

    mag = mag[igood] 
    z = z[igood]

    # Plot it
    if band in BANDCOLOR.keys(): color = BANDCOLOR[band] 
    else : color = 'k'
    plotdefaults={'ls':' ','marker':'o',
                  'mew':0.2,'ms':5,'alpha':0.4, 'mfc':color,'mec':color,}
    plotargs = dict( plotdefaults.items() + kwargs.items() )

    ax = p.gca()

    if plotstyle == 'points' :
        # Plot a point for every simulated SN
        if band1 in BANDCOLOR.keys(): color1 = BANDCOLOR[band1] 
        else : color1 = 'k'
        if band2 in BANDCOLOR.keys(): color2 = BANDCOLOR[band2] 
        else : color2 = 'k'
        kwargs['mfc'] = color1
        kwargs['mec'] = color2
        p.plot( z, mag, **kwargs )
    elif plotstyle == 'median' :
        # Plot a rolling median at each redshift.
        # We use the 3-sigma-clipped mean and associated robust sigma
        # using astrolib-ported python functions defined below.

        # sort the mag and z arrays by redshift
        zsortidx = z.argsort()
        zsorted = z[zsortidx]
        magbyz = mag[zsortidx]

        # compute the sigma-clipped mean and associated robust sigma 
        # over bins containing 5% of the simulated SNe
        from numpy import array
        Nsim = len(sim.z)
        Nmed = int(0.05*Nsim)
        magmed,magmederr = [],[]
        magmax, magmin = [], []
        for imag in range( len(mag) ) : 
            magsample = magbyz[ max(0,imag-Nmed/2) : min(len(magbyz),max(0,imag-Nmed/2)+Nmed) ]
            mean, sigma = meanclip( magsample, clipsig=3, maxiter=3, converge_num=0.1 )
            magmed.append( mean ) 
            magmederr.append( sigma )
            magmax.append( max(magsample) )
            magmin.append( min(magsample) )
        magmed, magmederr = array(magmed),array(magmederr)
        magmax, magmin = array(magmax),array(magmin)
       
        ax = p.gca()

        plotdefaults1={'alpha':0.3}
        plotargs1 = dict( plotdefaults1.items() + kwargs.items() )
        fill_between( ax, zsorted, magmin, magmax, **plotargs1 )

        plotdefaults2={'alpha':0.6}
        plotargs2 = dict( plotdefaults2.items() + kwargs.items() )
        fill_between( ax, zsorted, magmed-magmederr, magmed+magmederr, **plotargs2 )

    ax.set_xlim(z.min()-0.2,z.max()+0.2)
    ax.set_ylim(mag.max()+0.2,mag.min()-0.2)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Vega Magnitude')
    if detlim : 
        ax.axhline( 25.5, ls='--', color='0.4')
        ax.text(0.25,25.3,r'3-$\sigma$ Detection Limit', ha='left',va='bottom',color='0.4')
    

    if restbands : 
        ax.set_ylim(mag.max()+2,mag.min()+0.4)
        # plotting bandpass cross-correlations
        sim.readfilters()
        zrange = np.arange( z.min(), z.max(), 0.02 )
        ax2 = ax.twinx()
        w2 = sim.FILTERS[band][:,0]
        f2 = sim.FILTERS[band][:,1]
        restbanddat = getrestbands( )
        for otherband in 'KHJYIRVBU' : 
            if otherband not in restbanddat.keys() :  continue
            if otherband in BANDCOLOR.keys(): 
                otherbandcolor = BANDCOLOR[otherband] 
            else : otherbandcolor = 'k'
            w1 = restbanddat[otherband][:,0]
            f1 = restbanddat[otherband][:,1]
            xcor = xcorz( w1, f1, w2, f2, zrange, normalize=True )
            if xcor.max() == 0 : continue
            ax2.plot( zrange, xcor, marker=' ', color=otherbandcolor, ls='-' )
            ax2.set_ylim( -0.02, 8 )

            # Label the filter xcor line, but 
            # don't over-write labels on right side:
            if xcor.argmax()== len(xcor)-1: 
                if zrange[-1] == z.max : continue
            zmax = zrange[ xcor.argmax() ]
            ax2.text( zmax, xcor.max() - 0.1, otherband, 
                      color=otherbandcolor, backgroundcolor='w')
            ax2.set_yticks( [] )
    p.draw()
    return( ax )

    
                             
def multiplot_mag_z( sim, bands='GRXIZMH', mjd='peak', sndat={}, restbands=True, **kwargs ):
    """ multi-panel plot showing peak mag vs z diagrams.

    mjd='peak' is a special case that samples all simulated SNe
    at their respective peaks.  Otherwise we sample all at the 
    same MJD, which probably means they are at different LC ages.
    """
    fig = p.gcf()
    Nax = len(bands)

    if Nax > 25 : Nrow, Ncol = 5,6
    elif Nax > 20 : Nrow, Ncol = 5,5 
    elif Nax > 16 : Nrow, Ncol = 4,5 
    elif Nax > 12 : Nrow, Ncol = 4,4 
    elif Nax > 9 : Nrow, Ncol = 3,4 
    elif Nax > 6 : Nrow, Ncol = 3,3
    elif Nax > 4 : Nrow, Ncol = 2,3
    elif Nax > 3 : Nrow, Ncol = 2,2
    elif Nax > 2 : Nrow, Ncol = 1,3
    elif Nax > 1 : Nrow, Ncol = 1,2
    else: Nrow,Ncol = 1, 1

    iax = 0
    for band in bands : 
        iax += 1
        ax = fig.add_subplot( Nrow, Ncol, iax )
        plot_mag_z( sim, band, mjd=mjd, restbands=restbands, **kwargs )
        if band in sndat.keys() : 
            ax.axhline( sndat[band], color='k',ls='-',lw=2 )
        if 'z' in sndat.keys() : 
            ax.axvline( sndat['z'], color='k',ls='-',lw=2 )



def plot_obscolor_z( sim, bands='WH', mjd='peak', clobber=False, **kwargs ):
    """ plot the observed color at the given MJD against redshift.
    mjd='peak' is a special case that samples all simulated SNe
    at their respective peaks.  Otherwise we sample all at the 
    same MJD, which probably means they are at different LC ages.
     """ 
    z = sim.z
    band1 = bands[0]
    band2 = bands[1]

    if mjd in [ None, 0, 'pk','peak'] : 
        # read in the peak mags
        obsmag1 = sim.__dict__['SIM_PEAKMAG_'+band1]
        obsmag2 = sim.__dict__['SIM_PEAKMAG_'+band2]
    else  : 
        # sample the photometry for all SNe at the given mjd, with an enormous match window
        # so that we always sample the observation nearest to mjd, regardless of 
        # how far from mjd it actually is.
        sim.samplephot( mjd=mjd, tmatch=1000, clobber=clobber )
        obsmag1 = sim.__dict__['%s%i'%(band1,int(mjd))]
        obsmag2 = sim.__dict__['%s%i'%(band2,int(mjd))]

    # limit to observations with legit data
    igood = np.where( (obsmag1<99) & (obsmag1>-99) &
                      (obsmag2<99) & (obsmag2>-99) )[0]
    if not len(igood) : 
        print( "ERROR: no good mags for %s vs z"%(bands))
        return( None ) 
    obscolor = obsmag1[igood] - obsmag2[igood]
    z = z[igood]

    # Plot it
    if band1 in BANDCOLOR.keys(): color = BANDCOLOR[band1] 
    else : color = 'k'
    plotdefaults={'ls':' ','mew':0.2,'ms':5,'alpha':0.4, 'mfc':color,'mec':color,
                  'marker':'o'}
    plotargs = dict( plotdefaults.items() + kwargs.items() )

    ax = p.gca()
    ax.plot( z, obscolor, **plotargs)
    ax.text( 0.9,0.9, '%s - %s vs z'%(band1,band2), ha='right', va='top', 
             color=color, backgroundcolor='w', transform=ax.transAxes) 
    return( 1 )


def plot_color_z( sim, band1='W', band2='H', mjd='peak', 
                  plotstyle='median', snmags={}, **kwargs ):
    """ 
    plot the colors against redshift: band1-band2 vs z 
    at the given mjd day or list of days

    mjd='peak' is a special case that samples all simulated SNe
    at their respective peaks.  Otherwise we sample all at the 
    same MJD, which probably means they are at different LC ages.

       plotstyle = 'median' or 'points'
    """ 
    from matplotlib import cm

    # For now, assume that all SNe in the sim are of the same type
    sntype = SNTYPEDICT[ sim.SNTYPE[0] ]

    plotdefaults = {'ls':' ','marker':'o','mew':0.2,'ms':5,'alpha':0.4 } 
    if sntype in ['II','IIn','IIP','IIL'] : 
        plotdefaults['mfc'] = 'b'
        plotdefaults['mec'] = 'b'
        plotdefaults['color'] = 'b'
        cmap = cm.Blues
    elif sntype in ['Ib','Ic','Ibc'] : 
        plotdefaults['mfc'] = 'g'
        plotdefaults['mec'] = 'g'
        plotdefaults['color'] = 'g'
        cmap = cm.Greens
    elif sntype == 'Ia': 
        plotdefaults['mfc'] = 'r'
        plotdefaults['mec'] = 'r'
        plotdefaults['color'] = 'r'
        cmap = cm.Reds
    plotargs = dict( plotdefaults.items() + kwargs.items() )

    if mjd in [ 0, 'pk','peak'] : 
        # read in the peak mags
        mag1 = sim.__dict__['SIM_PEAKMAG_'+band1]
        mag2 = sim.__dict__['SIM_PEAKMAG_'+band2]
    else  : 
        # sample the light curves at the given obs-frame age (rel. to peak)
        sim.samplephot( mjd )
        mag1 = sim.__dict__['%s%i'%(band1, int(mjd))]
        mag2 = sim.__dict__['%s%i'%(band2, int(mjd))]

    # limit to observations with legit data
    igood = np.where( (mag1<99) & (mag1>-99) & 
                      (mag2<99) & (mag2>-99)  )[0]
    if not len(igood) : 
        print( "ERROR: no good mags for %s-%s vs %s"%(band1,band2,band2))
        return( None ) 

    mag1 = mag1[igood] 
    mag2 = mag2[igood] 
    color = mag1-mag2
    z = sim.z[igood]

    ax = p.gca()
    if plotstyle == 'points' :
        # Plot a point for every simulated SN
        if band1 in BANDCOLOR.keys(): color1 = BANDCOLOR[band1] 
        else : color1 = 'k'
        if band2 in BANDCOLOR.keys(): color2 = BANDCOLOR[band2] 
        else : color2 = 'k'
        kwargs['mfc'] = color1
        kwargs['mec'] = color2
        p.plot( z, color, **kwargs )
    elif plotstyle == 'median' :
        # Plot a rolling median at each redshift.
        # We use the 3-sigma-clipped mean and associated robust sigma
        # using astrolib-ported python functions defined below.

        # sort the color and z arrays by redshift
        zsortidx = z.argsort()
        zsorted = z[zsortidx]
        colorbyz = color[zsortidx]

        # compute the sigma-clipped mean and associated robust sigma 
        # over bins containing 5% of the simulated SNe
        from numpy import array
        Nsim = len(sim.z)
        Nmed = int(0.05*Nsim)
        cmed,cmederr = [],[]
        for icolor in range( len(color) ) : 
            colorsample = colorbyz[ max(0,icolor-Nmed/2) : min(len(colorbyz),max(0,icolor-Nmed/2)+Nmed) ]
            mean, sigma = meanclip( colorsample, clipsig=3, maxiter=3, converge_num=0.1 )
            cmed.append( mean ) 
            cmederr.append( sigma )
        cmed, cmederr = array(cmed),array(cmederr)
       
        ax = p.gca()
        fill_between( ax, zsorted, cmed-cmederr, cmed+cmederr, **kwargs )
        #p.plot(  zsorted, cmed, ls='-', color=kwargs['color'], lw=2 )

    if band1 in snmags.keys() and band2 in snmags.keys() and 'z' in snmags.keys() : 
        sncolor = snmags[band1]-snmags[band2]
        snmag = snmags[band2]
        snz = snmags['z']
        if 'd'+band1 in snmags.keys() and 'd'+band2 in snmags.keys() and 'dz' in snmags.keys(): 
            dsncolor = np.sqrt( snmags['d'+band1]**2 + snmags['d'+band2]**2 )
            dsnz = snmags['dz']
            p.errorbar(  snz, sncolor, dsncolor, dsnz, color='k', marker='o', capsize=0, elinewidth=2, ecolor='k'  )
        p.plot( snz, snmags[band1]-snmags[band2],  color='k', marker='o'  )

    ax = p.gca()
    ax.set_ylabel('%s-%s'%(band1,band2) )
    ax.set_xlabel('Redshift')
    ax.set_xlim( sim.z.min(), sim.z.max() )
    ax.set_ylim( color.min(), color.max() )
    return(1)


def multiplot_color_z( sim, mjd='peak', bluebands='GRXIZMH', redbands='XH', 
                       tobs=0, snmags={}, **kwargs ):
    """ multi-panel plot showing color-mag diagrams.
    mjd='peak' is a special case that samples all simulated SNe
    at their respective peaks.  Otherwise we sample all at the 
    same MJD, which probably means they are at different LC ages.
    """
    fig = p.gcf()
    Nax = 0

    if len(bluebands)==1 : bluebands=[bluebands]
    if len(redbands)==1 : redbands=[redbands]
    for bband in bluebands : 
        ibband = BANDORDER.find( bband )
        for rband in redbands : 
            irband = BANDORDER.find( rband )
            if irband <= ibband : continue
            Nax += 1
            break

    Nrow = 1
    Ncol = 1
    if Nax > 25 : Nrow, Ncol = 5,6
    elif Nax > 20 : Nrow, Ncol = 5,5 
    elif Nax > 16 : Nrow, Ncol = 4,5 
    elif Nax > 12 : Nrow, Ncol = 4,4 
    elif Nax > 9 : Nrow, Ncol = 3,4 
    elif Nax > 6 : Nrow, Ncol = 3,3
    elif Nax > 4 : Nrow, Ncol = 2,3
    elif Nax > 3 : Nrow, Ncol = 2,2
    elif Nax > 2 : Nrow, Ncol = 1,3
    elif Nax > 1 : Nrow, Ncol = 1,2
    else: Nrow,Ncol = 1, 1

    iax = 0
    for bband in bluebands : 
        ibband = BANDORDER.find( bband )
        for rband in redbands : 
            irband = BANDORDER.find( rband )
            if irband <= ibband : continue
            iax += 1
            ax = fig.add_subplot( Nrow, Ncol, iax )
            plot_color_z( sim, mjd=mjd, band1=bband, band2=rband, tobs=tobs, **kwargs )
            if bband in snmags.keys() and rband in snmags.keys() : 
                p.plot( snmags['z'], snmags[bband]-snmags[rband], marker='D',
                      mec='w', mfc='k',mew=1.5,ms=12 )
            break



def plotSALT2par(sim ) :
    """ plot histograms showing the range of light curve 
    shapes and colors (assumes a SALT2 simulation)""" 

    fig = p.figure(1) 
    p.clf()

    idet = sim.DUMP['idet']

    # Color distribution
    ax1 = fig.add_subplot(2,2,1)
    c = sim.DUMP['S2c']
    cbin, cedge = np.histogram( c, bins=30 ) 
    cdetbin, cdetedge = np.histogram( c[idet], bins=30 ) 
    p.plot( cedge[:-1], cbin, drawstyle='steps-post',color='r', label='simulated')
    p.plot( cdetedge[:-1], cdetbin, drawstyle='steps-post',color='g', label='detected')
    ax1.set_ylabel('Number of SNe')
    ax1.text(0.05,0.95, 'SALT2 Color: c', transform=ax1.transAxes, ha='left',va='top')

    # Stretch distribution
    ax2 = fig.add_subplot(2,2,2)
    x1 = sim.DUMP['S2x1']
    x1bin, x1edge = np.histogram( x1, bins=30 ) 
    x1detbin, x1detedge = np.histogram( x1[idet], bins=30 ) 
    p.plot( x1edge[:-1], x1bin, drawstyle='steps-post', color='r', label='sim' )
    p.plot( x1detedge[:-1], x1detbin, drawstyle='steps-post', color='g', label='det' )
    p.legend( loc='upper right', frameon=False, numpoints=1, handlelen=0.1, borderpad=0.2)
    ax2.text(0.05,0.95, 'SALT2 Stretch: x1', transform=ax2.transAxes, ha='left',va='top')

    # Redshift distribution
    ax3 = fig.add_subplot(2,2,3)
    z = sim.DUMP['REDSHIFT']
    zbin, zedge = np.histogram( z, bins=30 ) 
    zdetbin, zdetedge = np.histogram( z[idet], bins=30 ) 
    p.plot( zedge[:-1], zbin, drawstyle='steps-post', color='r', label='z' )
    p.plot( zdetedge[:-1], zdetbin, drawstyle='steps-post', color='g', label='z' )
    ax3.text(0.05,0.95, 'Redshift: z', transform=ax3.transAxes, ha='left',va='top')
    ax3.set_ylabel('# simulated SNe')



ETCdatH_1ksec = {
    'mag':np.array( [ 20, 21, 22, 22.5, 23, 23.2,  23.3,  23.4,  23.6,  23.8, 24.0, 24.2, 24.4,    24.6,  25, 25.5, 26, ]),
    'snr':np.array( [ 188, 93, 41.9, 27.36, 17.7, 14.81, 13.55, 12.39, 10.35, 8.65, 7.22,  6.0,  5.0, 4.18, 2.9, 1.84, 1.16 ]), 
    }

def plotSNR( sim, obsdat=ETCdatH_1ksec, **kwargs ):
    """ multi-panel plot showing mag vs z and/or S/N Ratio vs z diagrams
    Can provide a dictionary 'obsdat' with observed data as lists in items
    with keys 'mag' and 'snr'.

    NOTE: For the moment this is only for H band. 
    """
    idet = sim.DUMP['idet']

    # # S/N ratio histogram
    # ax1 = fig.add_subplot(2,2,1)
    # snr = sim.DUMP['SNRMAX_H']
    # snrbin, snredge = np.histogram( snr, bins=30 ) 
    # snrdetbin, snrdetedge = np.histogram( snr[idet], bins=30 ) 
    # p.plot( snredge[:-1], snrbin, drawstyle='steps-post',color='r', label='simulated')
    # p.plot( snrdetedge[:-1], snrdetbin, drawstyle='steps-post',color='g', label='detected')
    # ax1.set_ylabel('Number of SNe')
    # ax1.text(0.05,0.95, 'S/N Ratio', transform=ax1.transAxes, ha='left',va='top')

    # # S/N ratio vs mag
    # ax2 = fig.add_subplot(2,2,2)
    ax2 = p.gca()

    snr = sim.DUMP['SNRMAX_H']
    mag = sim.DUMP['MAGT0_H']

    defaultargs = {'marker':'o','color':'r','ls':' '}
    plotargs = dict( defaultargs.items() + kwargs.items() ) 

    ax2.plot( mag, snr, **plotargs )
    ax2.set_ylabel('S/N Ratio')
    ax2.set_xlabel('Vega Mag')
    #ax2.text(0.95,0.95, 'S/N Ratio', transform=ax2.transAxes, ha='right',va='top')

    if 'mag' in obsdat.keys() : 
        if 'snr' in obsdat.keys(): 
            ax2.plot( obsdat['mag'], obsdat['snr'], marker='s', ls=' ', color='r' )
        #if 'snropt' in obsdat.keys(): 
        #    ax2.plot( obsdat['mag'], obsdat['snropt'], marker='d', ls=' ', color='c' )

def plotDumpHist( sim, dumpvar, bins=30, showlegend=False, **kwargs ):
    """ 
    Plot a histogram for a single variable from the .DUMP file 
    Plots the full sample in black and the sub-sample of 
    detected SNe in red.
    """
    idet = sim.DUMP['idet']
    ax1 = p.gca()
    dumpdat = sim.DUMP[dumpvar]
    ct, edge = np.histogram( dumpdat, bins=bins ) 
    detct, detedge = np.histogram( dumpdat[idet], bins=bins ) 
    p.plot( edge[:-1], ct, drawstyle='steps-post',color='k', label='sim', **kwargs)
    p.plot( detedge[:-1], detct, drawstyle='steps-post',color='r', label='det', **kwargs)
    ax1.set_ylabel('Number of SNe')
    ax1.set_xlabel(dumpvar)
    ax1.text(0.05,0.95, dumpvar, transform=ax1.transAxes, ha='left',va='top')
    if showlegend: 
        p.legend( loc='best', frameon=False, numpoints=1, handlelen=0.1, borderpad=0.2)



def plotDump2( sim, dumpvarx, dumpvary, showlegend=False, **kwargs ):
    """ 
    Plot two columns from the .DUMP file against each other.
    Plots the full sample in black and the sub-sample of 
    detected SNe in red.
    """
    idet = sim.DUMP['idet']
    ax1 = p.gca()
    x = sim.DUMP[dumpvarx]
    y = sim.DUMP[dumpvary]
    p.plot( x, y, ls=' ', marker='o', color='k', label='sim', **kwargs)
    p.plot( x[idet], y[idet], ls=' ', marker='s', color='r', label='det', **kwargs)
    ax1.set_xlabel(dumpvarx)
    ax1.set_ylabel(dumpvary)
    if showlegend: 
        ax1.text(0.05,0.95, dumpvary+' vs '+dumpvarx, transform=ax1.transAxes, ha='left',va='top')
        p.legend( loc='best', frameon=True, numpoints=1, handlelen=0.05, borderpad=0.2,
                  handletextpad=0.1)

def plotDetEff( sim, band='H', magbinwidth=0.2, zbinwidth=0.2, 
                clobber=False, **kwargs ):
    """ 
    Plot the detection efficiency vs mag curve 
    """
    idet = sim.DUMP['idet']
    ciddet = sim.DUMP['CID'][idet]

    # sample the photometry for all SNe at t=0, with an enormous match window
    # so that we always sample the observation nearest to peak, regardless of 
    # how far from peak it actually is.
    sim.samplephot( tobs=0, tmatch=1000, clobber=clobber )
    mags = sim.__dict__['mag%sp00'%band]
    z = sim.z

    igood = np.where( (mags>0) & (mags<99) )[0]
    magmin = np.min( mags[igood] )
    magmax = np.max( mags[igood] )
    mbinlist = np.arange( magmin, magmax+magbinwidth/2., magbinwidth ) 

    zmin = np.min( z[igood] )
    zmax = np.max( z[igood] )
    zbinlist = np.arange( zmin, zmax+zbinwidth/2., zbinwidth ) 

    mdetefflist = []
    for binmag0 in mbinlist: 
        inbin = np.where((mags>binmag0) & (mags<binmag0+magbinwidth) )[0]
        if len(inbin) : 
            Ninbin = len(inbin) 
            Ndetinbin = len( [ cid for cid in sim.SNID[inbin] if int(cid) in ciddet ] )
            deteff = float(Ndetinbin) / Ninbin 
        elif binmag0 > 25.5 : deteff=0
        else  : deteff=1
        mdetefflist.append( deteff ) 

    zdetefflist = []
    for binz0 in zbinlist: 
        inbin = np.where((z>binz0) & (z<binz0+zbinwidth) )[0]
        Ninbin = len(inbin) 
        if Ninbin : 
            Ndetinbin = len( [ cid for cid in sim.SNID[inbin] if int(cid) in ciddet ] )
            deteff = float(Ndetinbin) / Ninbin 
        elif binz0 > 2.5 : deteff=0
        else  : deteff=1
        zdetefflist.append( deteff ) 
    
    plotdefaults = {'color':'r', 'ls':'-', 'drawstyle':'steps-pre'}
    plotargs = dict( plotdefaults.items() + kwargs.items() )

    ax1 = p.subplot(211) 
    ax1.plot( mbinlist, mdetefflist,  **plotargs )
    ax1.set_xlabel('brightest observed mag')
    ax1.set_ylabel('detection efficiency')
    ax1.set_ylim( -0.05,1.1 )

    ax1 = p.subplot(212) 
    ax1.plot( zbinlist, zdetefflist,  **plotargs )
    ax1.set_xlabel('redshift')
    ax1.set_ylabel('detection efficiency')
    ax1.set_ylim( -0.05,1.1 )



def plot_color_curve( sim, band1='W', band2='H', 
                      plotstyle='median', 
                      mjdpk=None, Nmjd=50, mjdstep=2,
                      snmags={}, **kwargs ):
    """ 
    plot the colors against redshift: band1-band2 vs time 
    plotstyle : 'median' or 'lines'
    mjdpk, Nmjd, mjdstep : defines the MJD range to sample
    """ 
    from matplotlib import cm

    if band1 not in sim.bands : 
        print("No %s band available."%band1)
        return(None)
    if band2 not in sim.bands : 
        print("No %s band available."%band2)
        return(None)

    # For now, assume that all SNe in the sim are of the same type
    sntype = SNTYPEDICT[ sim.SNTYPE[0] ]

    plotdefaults = {'ls':' ','marker':'o','mew':0.2,'ms':5,'alpha':0.4 } 
    if sntype in ['II','IIn','IIP','IIL'] : 
        plotdefaults['mfc'] = 'b'
        plotdefaults['mec'] = 'b'
        plotdefaults['color'] = 'b'
        cmap = cm.Blues
    elif sntype in ['Ib','Ic','Ibc'] : 
        plotdefaults['mfc'] = 'g'
        plotdefaults['mec'] = 'g'
        plotdefaults['color'] = 'g'
        cmap = cm.Greens
    elif sntype == 'Ia': 
        plotdefaults['mfc'] = 'r'
        plotdefaults['mec'] = 'r'
        plotdefaults['color'] = 'r'
        cmap = cm.Reds
    plotargs = dict( plotdefaults.items() + kwargs.items() )

    # set up the range of dates for LC sampling
    if not mjdpk : mjdpk = np.median( sim.mjdpk )
    mjd0 = mjdpk-mjdstep*int(Nmjd/3.)
    tobs = np.arange( mjd0, mjd0+mjdstep*Nmjd,mjdstep)

    # Build up the color curves : 
    medcolor, topcolor, bottomcolor = [],[],[]
    for t in tobs : 
        # get mags for all SNe on this date
        sim.samplephot( t, bandlist=[band1,band2] )
        mag1t = sim.__dict__['mag%s%s'%(band1, timestr(t)-mjdpk)]
        mag2t = sim.__dict__['mag%s%s'%(band2, timestr(t)-mjdpk)]
    
        # limit to observations with legit data
        igood = np.where( (mag1t<99) & (mag1t>-99) & 
                          (mag2t<99) & (mag2t>-99)  )[0]
        if not len(igood) : 
            import pdb; pdb.set_trace()
            continue
        mag1t = mag1t[igood] 
        mag2t = mag2t[igood] 

        # measure colors for all simulated SNe
        colort = mag1t-mag2t
        return( colort )
       
        # Now compute the median and find the range about 
        # the median containing 68% of the simulated SNe
        medcolor.append( np.median( colort ) )
        topcolor.append( np.stdev( colort ) )
        topcolor.append( np.stdev( colort ) )
        

    ax = p.gca()
    if plotstyle == 'lines' :
        # Plot a line for every simulated SN
        if band1 in BANDCOLOR.keys(): color1 = BANDCOLOR[band1] 
        else : color1 = 'k'
        if band2 in BANDCOLOR.keys(): color2 = BANDCOLOR[band2] 
        else : color2 = 'k'
        kwargs['mfc'] = color1
        kwargs['mec'] = color2
        p.plot( z, color, **kwargs )
    elif plotstyle == 'median' :
        # Plot filled contours enclosing 68% of the sample
        # centered around a rolling median color at each MJD.
        # We use the 3-sigma-clipped mean.

        # sort the color and z arrays by redshift
        zsortidx = z.argsort()
        zsorted = z[zsortidx]
        colorbyz = color[zsortidx]

        # compute the sigma-clipped mean and associated robust sigma 
        # over bins containing 5% of the simulated SNe
        from numpy import array
        Nsim = len(sim.z)
        Nmed = int(0.05*Nsim)
        cmed,cmederr = [],[]
        for icolor in range( len(color) ) : 
            colorsample = colorbyz[ max(0,icolor-Nmed/2) : min(len(colorbyz),max(0,icolor-Nmed/2)+Nmed) ]
            mean, sigma = meanclip( colorsample, clipsig=3, maxiter=3, converge_num=0.1 )
            cmed.append( mean ) 
            cmederr.append( sigma )
        cmed, cmederr = array(cmed),array(cmederr)
       
        ax = p.gca()
        fill_between( ax, zsorted, cmed-cmederr, cmed+cmederr, **kwargs )
        #p.plot(  zsorted, cmed, ls='-', color=kwargs['color'], lw=2 )

    if band1 in snmags.keys() and band2 in snmags.keys() and 'z' in snmags.keys() : 
        sncolor = snmags[band1]-snmags[band2]
        snmag = snmags[band2]
        snz = snmags['z']
        if 'd'+band1 in snmags.keys() and 'd'+band2 in snmags.keys() and 'dz' in snmags.keys(): 
            dsncolor = np.sqrt( snmags['d'+band1]**2 + snmags['d'+band2]**2 )
            dsnz = snmags['dz']
            p.errorbar(  snz, sncolor, dsncolor, dsnz, color='k', marker='o', capsize=0, elinewidth=2, ecolor='k'  )
        p.plot( snz, snmags[band1]-snmags[band2],  color='k', marker='o'  )

    ax = p.gca()
    ax.set_ylabel('%s-%s'%(band1,band2) )
    ax.set_xlabel('Redshift')
    ax.set_xlim( sim.z.min(), sim.z.max() )
    ax.set_ylim( color.min(), color.max() )
    return(1)




def xcorz( w1, f1, w2, f2, zlist, normalize=True ) : 
    """
    Given a pair of 'flux' vectors f1(w1) and f2(w2), 
    compute the "redshift-cross-correlation" as follows:
    For each redshift value z in zlist, 
    apply the redshift to w1 and interpolate onto w2
    ( scipy.interp defaults to extrapolating with end values ).
    Then multiply the interpolated vector f1' with the static
    vector f2 and integrate the result to get the cross-correlation
    value for that z.
    Returns an array with the cross-correlation value at each z,
    normalized to set the peak value at unity by default.
    """
    from scipy import interp
    xcor = np.array( [ ( interp( w2, w1*(1+z), f1 ) * f2 ).sum() for z in zlist ] )
    if normalize : 
        if xcor.max() > 0 : xcmax = xcor.max()
        else : xcmax = 1
        return( xcor / xcmax ) 
    return( xcor ) 
    
    


def plotSearchEffIn( ) :

    detEffH = np.array([0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.921,0.920,0.915,0.894,0.811,0.573,0.248,0.070,0.017,0.004,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000])
    
    magH = np.array([15.000,15.300,15.600,15.900,16.200,16.500,16.800,17.100,17.400,17.700,18.000,18.300,18.600,18.900,19.200,19.500,19.800,20.100,20.400,20.700,21.000,21.300,21.600,21.900,22.200,22.500,22.800,23.100,23.400,23.700,24.000,24.300,24.600,24.900,25.200,25.500,25.800,26.100,26.400,26.700,27.000,27.300,27.600,27.900,28.200,28.500,28.800,29.100,29.400,29.700,30.000])

    magJ = np.array([15.000,15.300,15.600,15.900,16.200,16.500,16.800,17.100,17.400,17.700,18.000,18.300,18.600,18.900,19.200,19.500,19.800,20.100,20.400,20.700,21.000,21.300,21.600,21.900,22.200,22.500,22.800,23.100,23.400,23.700,24.000,24.300,24.600,24.900,25.200,25.500,25.800,26.100,26.400,26.700,27.000,27.300,27.600,27.900,28.200,28.500,28.800,29.100,29.400,29.700,30.000])

    detEffJ = np.array([0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.940,0.939,0.939,0.937,0.934,0.926,0.910,0.877,0.811,0.697,0.531,0.349,0.199,0.102,0.049,0.023,0.011,0.005,0.002,0.001,0.000,0.000,0.000,0.000,0.000])

    p.plot( magH, detEffH, color='r', ls='--')
    p.plot( magJ, detEffJ, color='b', ls='--')


def scumsum( a ): 
    """ 
    Sorted Cumulative Sum function : 
    Construct an array "sumabove" such that the cell at index i in sumabove 
    is equal to the sum of all cells from the input array "a" that have a
    cell value higher than a[i]  
    """    
    # Collapse the array into 1 dimension
    sumabove = a.ravel()
    
    # Sort the raveled array by descending cell value
    iravelsorted = sumabove.argsort( axis=0 )[::-1]

    # Reassign each cell to be the cumulative sum of all 
    # input array cells with a higher value :  
    sumabove[iravelsorted] = sumabove[iravelsorted].cumsum() 

    # Now unravel back into shape of original array and return
    return( sumabove.reshape( a.shape ) )


def getrestbands( verbose=False, ) :
    """ read in the filter definition  .dat files for the 
    standard Bessell(90)  UBVRI filters, extending to the IR
    with YJHK"""
    sndataroot = os.environ['SNDATA_ROOT']
    filtdatadir = os.path.join( sndataroot, 'filters/MLCS.IR' )

    FILTERS = {}
    datfiles = glob.glob( os.path.join( filtdatadir, '*.dat' ) )
    if verbose : print( "Reading %i filter curves"%len(datfiles) )
    for band in 'UBVRIYJHK' : 
        for datfile in datfiles : 
            if not datfile.lower().endswith( '%s.dat'%band.lower() ) : continue
            FILTERS[ band ] = np.loadtxt( datfile ) 
    return(FILTERS) 


def fill_between(ax, x, y1, y2, **kwargs):
    """
    fill the space between y1(x) and y2(x) on axis ax
    with a matplotlib.patch using **kwargs for the color
    and such
    """
    # add x,y2 in reverse order for proper polygon filling
    from matplotlib.patches import Polygon
    verts = zip(x,y1) + [(x[i], y2[i]) for i in range(len(x)-1,-1,-1)]
    poly = Polygon(verts, **kwargs)
    ax.add_patch(poly)
    ax.autoscale_view()
    return poly



def meanclip(indata, clipsig=3.0, maxiter=5, converge_num=0.02, verbose=0):
   """
   from jiffyclyb:  https://gist.github.com/1310947

   Computes an iteratively sigma-clipped mean on a
   data set. Clipping is done about median, but mean
   is returned.

   .. note:: MYMEANCLIP routine from ACS library.

   :History:
       * 21/10/1998 Written by RSH, RITSS
       * 20/01/1999 Added SUBS, fixed misplaced paren on float call, improved doc. RSH
       * 24/11/2009 Converted to Python. PLL.

   Examples
   --------
   >>> mean, sigma = meanclip(indata)

   Parameters
   ----------
   indata: array_like
       Input data.

   clipsig: float
       Number of sigma at which to clip.

   maxiter: int
       Ceiling on number of clipping iterations.

   converge_num: float
       If the proportion of rejected pixels is less than
       this fraction, the iterations stop.

   verbose: {0, 1}
       Print messages to screen?

   Returns
   -------
   mean: float
       N-sigma clipped mean.

   sigma: float
       Standard deviation of remaining pixels.

   """
   import numpy

   # Flatten array
   skpix = indata.reshape( indata.size, )

   ct = indata.size
   iter = 0; c1 = 1.0 ; c2 = 0.0

   while (c1 >= c2) and (iter < maxiter):
       lastct = ct
       medval = numpy.median(skpix)
       sig = numpy.std(skpix)
       wsm = numpy.where( abs(skpix-medval) < clipsig*sig )
       ct = len(wsm[0])
       if ct > 0:
           skpix = skpix[wsm]

       c1 = abs(ct - lastct)
       c2 = converge_num * lastct
       iter += 1
   # End of while loop

   mean  = numpy.mean( skpix )
   sigma = robust_sigma( skpix )

   if verbose:
       prf = 'MEANCLIP:'
       print '%s %.1f-sigma clipped mean' % (prf, clipsig)
       print '%s Mean computed in %i iterations' % (prf, iter)
       print '%s Mean = %.6f, sigma = %.6f' % (prf, mean, sigma)

   return mean, sigma


def robust_sigma(in_y, zero=0): 
    """ 
    Calculate a resistant estimate of the dispersion of 
    a distribution. For an uncontaminated distribution, 
    this is identical to the standard deviation. 

    Use the median absolute deviation as the initial 
    estimate, then weight points using Tukey Biweight. 
    See, for example, Understanding Robust and 
    Exploratory Data Analysis, by Hoaglin, Mosteller 
    and Tukey, John Wiley and Sons, 1983. 

    .. note:: ROBUST_SIGMA routine from IDL ASTROLIB. 

    :History: 
        * H Freudenreich, STX, 8/90 
        * Replace MED call with MEDIAN(/EVEN), W. Landsman, December 
2001 
        * Converted to Python by P. L. Lim, 11/2009 

    Examples 
    -------- 
    >>> result = robust_sigma(in_y, zero=1) 

    Parameters 
    ---------- 
    in_y: array_like 
        Vector of quantity for which the dispersion is 
        to be calculated 

    zero: int 
        If set, the dispersion is calculated w.r.t. 0.0 
        rather than the central value of the vector. If 
        Y is a vector of residuals, this should be set. 

    Returns 
    ------- 
    out_val: float 
        Dispersion value. If failed, returns -1. 

    """ 
    import numpy
    # Flatten array 
    y = in_y.reshape(in_y.size, ) 

    eps = 1.0E-20 
    c1 = 0.6745 
    c2 = 0.80 
    c3 = 6.0 
    c4 = 5.0 
    c_err = -1.0 
    min_points = 3 

    if zero: 
        y0 = 0.0 
    else: 
        y0 = numpy.median(y) 

    dy    = y - y0 
    del_y = abs( dy ) 

    # First, the median absolute deviation MAD about the median: 

    mad = numpy.median( del_y ) / c1 

    # If the MAD=0, try the MEAN absolute deviation: 
    if mad < eps: 
        mad = numpy.mean( del_y ) / c2 
    if mad < eps: 
        return 0.0 

    # Now the biweighted value: 
    u  = dy / (c3 * mad) 
    uu = u*u 
    q  = numpy.where(uu <= 1.0) 
    count = len(q[0]) 
    if count < min_points: 
        print 'ROBUST_SIGMA: This distribution is TOO WEIRD! Returning', c_err 
        return c_err 

    numerator = numpy.sum( (y[q]-y0)**2.0 * (1.0-uu[q])**4.0 ) 
    n    = y.size 
    den1 = numpy.sum( (1.0-uu[q]) * (1.0-c4*uu[q]) ) 
    siggma = n * numerator / ( den1 * (den1 - 1.0) ) 

    if siggma > 0: 
        out_val = numpy.sqrt( siggma ) 
    else: 
        out_val = 0.0 

    return out_val
