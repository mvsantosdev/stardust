#! /usr/bin/env python2.5
"""
120101
S.Rodney

Classes and functions for handling SNANA light curves.
Read, write, modify, plot, measure and fit.


Classes: 

* LightCurve : a single filter light curve. Primarily just flux
  vs. time, along with flux uncertainties, but can also include mags.

*  Supernova  : a full-fledged transient object, complete with a
  multi-color light curve, observational metadata, and functional 
  attributes.

TODO: 
  write out A/(B-V) for each filter into each spline file
  so that lcmatch.c can read it instead of having it hard-coded

"""
import os
import sys
import exceptions

# Physical Constants
c = 299792.458
# Lambda-CDM cosmological parameters (Komatsu et al. 2009)
H0 = 70.5
h = H0 / 100.
Om = 0.1358 / h**2
Ol = 0.726
Ode = 0.726
w  = -1.


class SuperNova(object):
    """
    A supernova object.
    """
    def __init__(self, datfile=None, clobber=False, debug=False):
        """
        Initialize an empty SN object.
        If datfile argument is provided, load info from the datfile.
        """
        if debug: import pdb;  pdb.set_trace()

        # metadata : measured externally (e.g. from spectrum of SN or host)
        self.name = ''           # SN is nameless by default.
        self.RA = '-'            # Right Ascension
        self.DEC = '-'           # Declination
        # self.z = None            # redshift
        # self.zerr = None         # uncertainty on z
        # self.zfrom = ''          # source of z
        self.zsn = None          # redshift of SN
        self.zsnerr = None       # uncertainty on zsn
        self.zsnfrom=''          # source of zsn     
        self.zhost = None        # redshift of host galaxy
        self.zhosterr = None     # uncertainty on zhost
        self.zhostfrom=''        # source of zhost      
        self.type = ''           # SN class, determined externally (e.g. from spectrum)
        self.EBVmw = 0.0         # milky way E(B-V) 
        self.tref  = 0.0         # reference date for plotting (MJD)
        self.EBVhost = None      # host galaxy E(B-V) (measured externally)
        self.survey = ''         # survey that generated SN light curve
        self.instrument = ''     # instrument used for collecting SN light curve
        
        # Load light curves and metadata from .dat file if available
        if datfile: self.getdata(datfile)   



    def getdata(self, datfile, debug=False ):
        """
        Read the light curve and metadata for a SN (candidate)
        from a SNANA-style .dat file: 
        Comment lines  beginning with '#' and any lines  after the 'END:' 
        keyword are ignored.  
        Metadata values are provided as
          KEY:  value # optional comment

        Light curve data are provided as 
          OBS:  val1  val2  val3 ...

        where the number of variables per obs line and the 
        number of obs lines  are given by the required keywords
        NVAR and NOBS, respectively.
        """
        from numpy import array
        if debug : import pdb; pdb.set_trace()

        if not os.path.isfile(datfile):
            raise exceptions.RuntimeError(
                "%s does not exist."%datfile) 
        self.datfile = os.path.abspath(datfile)
        fin = open(datfile,'r')
        data = fin.readlines()
        fin.close()
        filt,date=[],[]
        flux,dflux=[],[]
        mag,dmag,zpt=[],[],[]
        comments = []
        obslist = []

        # read light curve data
        nobs,nvar,ended = None,None,False
        for i in range(len(data)):
            line = data[i].strip()
            if(len(line.strip())==0) : continue
            if line.startswith("#") : comments.append(line)  
            if ':' in line : 
                keyval = line.split(':')
                key = keyval[0].strip()
                val = keyval[1].strip()
                if '#' in val : val = val.split('#')[0].strip()
                val = recast( val )
                if key == 'END' : 
                    ended = True 
                    break 
                if key=='OBS':
                    obslist.append( [ recast( v ) for v in val.split() ] )
                else : 
                    if key=='VARLIST': 
                        val = val.split()
                        varlist = val
                    self.__dict__[ key ] = val
                    if key=='NOBS': nobs  = val
                    elif key=='NVAR': nvar  = val

        # check for file completeness
        if not ended : 
            print("WARNING: no 'END:' line in %s!  SNANA will not be happy."%datfile)
        if nobs==None or nvar==None : 
            raise exceptions.RuntimeError(
                "ERROR: no 'NOBS:' or 'NVAR:'  in %s! SNANA can't read this."%datfile)
        if( not ( ('MJD' in varlist) and ('FLT' in varlist) and
                  ( ( ('FLUXCAL' in varlist) and ('FLUXCALERR' in varlist) ) 
                    or
                    ( ('MAG' in varlist) and ('MAGERR' in varlist) ) ) ) ) :
            raise exceptions.RuntimeError(
                "Missing required data column among MJD,FLT,FLUX(ERR)/MAG(ERR) in %s"%datfile)

        # unpack obs list into separate arrays for each variable
        for ivar in range(nvar) : 
            var = varlist[ivar]
            datalist = []
            for iobs in range(nobs) : 
                datalist.append( obslist[iobs][ivar] )
            self.__dict__[var] = array( datalist )

        # convert the obs list into separate single-band lightcurves
        # for flt in self.FILTERS : 
        return(None)


    def plotband( self, band, axes=None, debug=False, **kwargs ) : 
        """ 
        Plot a single-band lightcurve. Use the given pylab axes instance
        if provided. 
        **kwargs  are passed to the matplotlib errorbar command
        Returns the axes instances used.

        TODO: handle upper limits a la Primo
        """
        from pylab import gca 
        from numpy import where
        if debug : import pdb; pdb.set_trace()
        if not axes : axes = gca()
        iband = where( self.FLT == band ) 
        t = self.MJD[ iband ] 
        m = self.MAG[ iband ] 
        merr = self.MAGERR[ iband ] 
        axes.errorbar( t, m, merr, **kwargs )
        if not axes.yaxis_inverted() : axes.invert_yaxis()
        return( axes )


    def plotJH( self ) :
        """ convenience function for nicely formatted plot 
        of J and H light curves, with twin y axes and 
        shared x axis.
        """ 
        from pylab import clf
        clf()
        ax1 = self.plotband( 'J', color='b', marker='o', ls=' ', capsize=0 )
        ax1.set_xlabel( 'observer time (MJD)')
        ax1.text( 0.95,0.95, self.SNID, transform=ax1.transAxes,
                  ha='right',va='top', size='x-large' )
        ax1.set_ylabel( 'magnitude (Vega)' )
        ax2 = self.plotband( 'H', axes=ax1, color='r', marker='s', ls=' ', capsize=0 )

        # Settings for Blue F125W axis on the left and red F160W on the right:
        # ax2 = self.plotband( 'H', axes=ax1.twinx(), color='r', marker='s', ls=' ', capsize=0 )
        #ax1.set_ylabel( 'F125W magnitude (Vega)', color='b')
        #ax2.set_ylabel( 'F160W magnitude (Vega)', rotation=-90, color='r')
        #for t in ax1.yaxis.get_ticklabels():  t.set_color('b')
        #for t in ax2.yaxis.get_ticklabels():  t.set_color('r')

    def writedata(self,datfile=None,clobber=False, verbose=False):
        """
        Write out a .dat file with all available info
        """
        # FOR WRITING : 
        # extract the list of filter names by weeding out all duplicates
        # self.FILTERS  = ''.join( unique( [ f for f in self.FLT ] )

        if not datfile : datfile = self.datfile
        if os.path.isfile(datfile) and not clobber: 
            print("%s exists. Not clobbering."%datfile)
            return(None)

        # make needed directories
        outdir = os.path.dirname( datfile ) 
        if outdir and not os.path.isdir( outdir ) : 
            os.makedirs( outdir )

        # write out header info
        fout = open(datfile,'w')
        for key in self.__dict__.keys() : 
            if( self.__dict__[key] and 
                isinstance(self.__dict__[key], (str,float,int)) ): 
                print >>fout, '#%-12s: %-12s'%(key,self.__dict__[key])
            
        # write out Light Curve Data
        print >>fout,"#columns= filt  date  flux     dflux       mag      dmag   zpt"
        mlc = self.mlc
        for filt in self.filters:
            lc = mlc[filt] 
            if 'mag' not in lc.__dict__.keys(): lc.flux2mag()
            elif 'flux' not in lc.__dict__.keys(): lc.mag2flux()
            for i in range(len(lc.date)):
                print >>fout,"%8s %12.3f %12.5f %12.5f %12.5f %12.5f %12.5f"%(
                    filt, lc.date[i], lc.flux[i], lc.dflux[i],
                    lc.mag[i], lc.dmag[i], lc.zpt[i] )
                continue
            continue
        fout.close()
        if verbose : print("Wrote %s"%datfile)

    def writesalt(self):
        """ 
        write out the light curve formatted for SALT2 processing
        """
        # the lightfile holds basic SN metadata
        if not os.path.isdir(self.saltdir):
            os.makedirs(self.saltdir)
        fout = open("%s/lightfile"%self.saltdir,'w')
        print >>fout,"NAME %s"%self.name
        print >>fout,"RA %s"%self.RA
        print >>fout,"Decl %s"%self.DEC
        print >>fout,"Redshift %s"%self.z
        print >>fout,"MWEBV %s"%self.EBVmw
        fout.close()

        # each filter gets its own light curve file
        for filt in self.filters:
            fout = open("%s/lc2fit_%s.dat"%(self.saltdir,filt.strip("'")),'w')
            print >>fout, "@BAND %s"%filt.strip("'")
            print >>fout, "@INSTRUMENT %s"%self.instrument
            print >>fout, "@MAGSYS %s"%'VEGA'  
            print >>fout, "@SURVEY %s"%self.survey 
            print >>fout, "# Date :"
            print >>fout, "# Flux :"
            print >>fout, "# Fluxerr :"
            print >>fout, "# ZP :"
            print >>fout, "# end"
            for i in range(len(self.mlc[filt].flux)) : 
                print >>fout,"%12.3f %12.3f %12.3f %12.3f"%(
                    self.mlc[filt].date[i], max(0.0,self.mlc[filt].flux[i]), 
                    self.mlc[filt].dflux[i], self.mlc[filt].zpt[i] )
            fout.close()

    def flux2mag(self):
        """ convert flux to magnitudes in all filters """
        for lc in self.mlc.values() : lc.flux2mag()

    def mag2flux(self):
        """ convert flux to magnitudes in all filters """
        for lc in self.mlc.values() : lc.mag2flux()

    @property
    def peak(self):
        """ 
        estimate the location of the light curve peak 
        and report some info about that point.
        """
        from numpy import argmin,argmax, sqrt, log10

        signoise, filtername, flux, date = [],[],[],[]
        for filt in self.filters:
            flux +=  self.mlc[filt].flux.tolist()
            signoise += self.mlc[filt].signoise.tolist()
            date +=  self.mlc[filt].date.tolist()
            filtername += [ filt for i in self.mlc[filt].date ]
        # If we've got SOFT results, extract the peak location 
        # from the best fitting (max likelihood) model
        # and use it to count Npre and Npost
        # (not for determining peak observed S/N or flux)
        Npre, Npost = 0,0
        try: 
            if 'model' not in self.__dict__.keys() : 
                self.getlcmatch()
                self.getmodel()
                # find the point nearest to the model peak
                tpk = self.model.tpk
                ipk = argmin( [ abs(sorted(date)[i]-tpk) 
                                for i in range(len(date)) ] )
                # find the "tpk+10 days" point
                z = self.model.z
                ipk10 = argmin( [ abs( sorted(date)[i] - 
                                       (tpk+10*(1+z)) ) 
                                  for i in range(len(date)) ] )
                Npre = ipk
                Npost = len(date) - ipk10
        except : pass

        # Now determine the peak observed flux and S/N :
        # first guess is the point with the highest S/N
        imaxSN = argmax(signoise)

        # second guess is the point with the highest flux
        # (but disregard if it has a low S/N)
        imaxflux = argmax(flux)
        if signoise[imaxflux] < 10 : imaxflux=imaxSN

        # hole-in-one shot : if the max flux has 
        # a S/N better than 50, then use that point 
        if signoise[imaxflux] > 50 : 
            tpk = date[imaxflux]
            fpk = flux[imaxflux]
            snpk = signoise[imaxflux]
            mpk = -2.5*log10(fpk) + 25
            filtpk = filtername[imaxflux]
            dtpk = 10

        else : 
            # report the average of the two
            tpk = (date[imaxflux] + date[imaxSN]) / 2.
            fpk = (flux[imaxflux] + flux[imaxSN]) / 2.
            snpk = (signoise[imaxflux] + signoise[imaxSN]) / 2.
            mpk = -2.5*log10(fpk) + 25

            # and an 'average' filter
            ifilt = int( 
                round(( self.filters.index(filtername[imaxflux]) + 
                        self.filters.index(filtername[imaxSN]))/2.)
                )
            if ifilt>=len(self.filters) : filtpk = self.filters[-1]
            else : filtpk = self.filters[ifilt]

            # and the uncertainty
            dtpk = abs( (date[imaxflux] - date[imaxSN]) / 2. )

        # Make a (wild) redshift guess based on the peak flux
        zMin = H0 / (c * sqrt(fpk) ) * 10**(-0.2*-16) 
        zMid = H0 / (c * sqrt(fpk) ) * 10**(-0.2*-19) 
        zMax = H0 / (c * sqrt(fpk) ) * 10**(-0.2*-21) 
        
        # Fill in Npre and Npost based on this guesstimate
        # even if there was no SOFT fit available
        if not Npre and not Npost : 
            Npre = int( (imaxflux + imaxSN ) / 2.)
            Npost = len(date) - Npre - 1

        peak = {'filter':filtpk, 'date':tpk, 
                'mag':mpk, 'flux':fpk, 'S/N':snpk, 
                'dateerr':dtpk, 
                'zguessMax':zMax,'zguess':zMid,'zguessMin':zMin,
                'Npre':Npre, 'Npost':Npost}
        return( peak )

    @property
    def z(self):
        """ best available redshift estimate """
        zlist    = [self.zsn, self.zhost, self.peak['zguess'] ]
        zerrlist = [self.zsnerr, self.zhosterr, 
                    self.peak['zguessMax']-self.peak['zguessMin']/2.]
        zerrmin =  min([ zerr for zerr in zerrlist if zerr])
        return( zlist[ zerrlist.index(zerrmin) ] )

    @property
    def zerr(self):
        """ best available redshift error estimate """
        zlist    = [self.zsn, self.zhost, self.peak['zguess'] ]
        zerrlist = [self.zsnerr, self.zhosterr, 
                    self.peak['zguessMax']-self.peak['zguessMin']/2.]
        zerrmin =  min([ zerr for zerr in zerrlist if zerr])
        return( zerrmin )

    @property
    def zfrom(self):
        """ best available redshift source """
        zlist    = [self.zsn, self.zhost, self.peak['zguess'] ]
        zerrlist = [self.zsnerr, self.zhosterr, 
                    self.peak['zguessMax']-self.peak['zguessMin']/2.]
        zerrmin =  min([ zerr for zerr in zerrlist if zerr])
        iz = zerrlist.index(zerrmin)
        if iz==0 : return(self.zsnfrom)
        if iz==1 : return(self.zhostfrom)
        if iz==2 : return('pk-guess')

    @property
    def filters(self):
        """
        if the filters used are from one of the known 
        filter sequences, then we return them as a sorted list
        """
        def sortfunc(f):
            filtnum =  { 'U':0,'B':1,'V':2,'R':3,'I':4,'Z':5,
                         'J':6,'H':7,'K':8,
                         'u':100,'g':110,'r':120,'i':130,'z':140,'y':150,
                         "u'":101,"g'":111,"r'":121,"i'":131,"z'":141,"y'":151,
                         'F606W':24,'F775W':25,'F850LP':26, 
                         'F125W':34,'F160W':35 }
            if f in filtnum : return(filtnum[f])
            else : return(99)
        inorder = sorted( self.mlc.keys(), key=sortfunc)
        return(inorder)

    @property
    def Nepochs(self):
        """ total number of epochs across all filters """
        return( sum( [len(self.mlc[filt].date) for filt in self.filters]) )

    @property
    def Nobs(self):
        """ total number of epochs across all filters """
        return( sum( [len(self.mlc[filt].date) for filt in self.filters]) )

    @property
    def modlike( self ):
        """ compute the likelihood for the current model"""
        from numpy import sqrt, pi, exp
        #import pdb; pdb.set_trace()

        if 'model' not in self.__dict__.keys() :
            print("No model available. Use self.loadmodel()")
            return(-1)
        if type(self.model)!=Model :
            print("Incorrect type for self.model.")
            return(-1)
        chi2 = 0 
        wgt = 1
        for filt in self.filters :
            lc = self.mlc[filt]
            tknot0 = self.model.tknot[filt][0]
            for i in range(len(lc.date)) :
                fmod = self.model.flux(lc.date[i], filt)
                dfmod = fmod * self.model.fxsoft

                # truncate the spline at the earlier date of
                # 5 days before the first knot or t=-20
                # (but note that the uncertainty uses the actual
                #  extrapolated spline values)
                tobs = lc.date[i]-self.model.tpk 
                if tobs < tknot0-5 and tobs < -20 : 
                    fmod = 0
                fdiff = fmod  - lc.flux[i]
                dfobs = lc.dflux[i]
                sig2 = dfmod*dfmod + dfobs*dfobs
                if sig2==0: sig2=1e-6
                chi2 += fdiff*fdiff / sig2
                wgt *= 1 / sqrt(2*pi*sig2)
        print( chi2, wgt, wgt * exp( -0.5 * chi2 ) )
        return( wgt * exp( -0.5 * chi2 ) )

    @property
    def modchi2( self ):
        """ compute the chi2 statistic for the current model"""
        if 'model' not in self.__dict__.keys() :
            print("No model available. Use self.loadmodel()")
            return(-1)
        if type(self.model)!=Model :
            print("Incorrect type for self.model.")
            return(-1)
        chi2 = 0 
        wgt = 1
        for filt in self.filters :
            lc = self.mlc[filt]
            for i in range(len(lc.date)) :
                fmod = self.model.flux(lc.date[i], filt)
                fdiff = fmod  - lc.flux[i]
                dfmod = fmod * self.model.fxsoft
                dfobs = lc.dflux[i]
                sig2 = dfmod*dfmod + dfobs*dfobs
                if sig2==0: sig2=1e-6
                chi2 += fdiff*fdiff / sig2
        return( chi2 )


        


    def tpkpolyfit(self, tpk=None, ntpk=20, dtpk=1, 
                 interact=False, clobber=False, debug=False):
           """ fit a 3rd or 5th order polynomial around the 
           highest S/N points to get a useful tpk range
                      _
                    /   \__
          ___|_____/  |     \___|____
            tpk0     tpk       tpk1
             |----ntpk*dtpk-----|

             Returns (tpk0,ntpk,dtpk) in MJD
           """
           from numpy import array, append, arange, median, std
           from scipy import polyfit, polyval
           import spline
           if debug: import pdb;  pdb.set_trace()

           if interact : 
               from pylab import plot,draw,show,gca,clf,\
                   figure,ioff,ion,draw,show
               clf()
               ioff()
               
           tpklist = []
           for filt in self.filters  :
               lc = self.mlc[filt]
               tfit = [ lc.date[i] for i in range(len(lc.date)) 
                        if  lc.signoise[i]>5 ]
               mfit = [ lc.mag[i] for i in range(len(lc.date)) 
                        if lc.signoise[i]>5 ]
               if len(tfit) < 3 : continue
               if len(tfit) < 5 : deg=2
               if len(mfit) < 10 or lc.date[-1]-lc.date[0]<20: deg=3
               else : deg=5

               # Option 1: using scipy and polyfit when available
               if USESCIPY :
                   coeff = polyfit( tfit, mfit, deg )
                   xfit = arange(tfit[0],tfit[-1],0.1)
                   yfit = polyval( coeff, xfit )
                   tpklist.append( xfit[ yfit.argmin() ] )
               
               # option 2: simple spline interpolation for bad scipy installs
               else :
                   spl = spline.Spline( tfit, mfit )
                   xfit = arange(tfit[0],tfit[-1],0.1)
                   yfit = spl( xfit )
                   tpklist.append( xfit[ yfit.argmin() ] )

               if interact : 
                   colorlist = ['purple','b','g','darkorange','r'] * 20
                   plot( xfit, yfit , colorlist[self.filters.index(filt)])

           tpk = median( tpklist ) 
           tpkerr = std( tpklist ) 

           if tpkerr > dtpk*ntpk/2. : 
               tpk0 = tpk - tpkerr
               tpk1 = tpk + tpkerr
           else : 
               tpk0 = tpk - dtpk*ntpk/2.
               tpk1 = tpk + dtpk*ntpk/2.

           # make an ordered array of all observations
           tmlc = array([])
           for filt in self.filters:
               tmlc = append(tmlc, self.mlc[filt].date )
           tmlc.sort()

           # if the peak is near the beginning/end 
           # of the light curve, then make sure we search 
           # over at least 20*(1+z) days before/after the peak
           z = self.z
           if ( tpk < (tmlc[0] + 10*(1+z)) and 
                tpk0 > (tmlc[0] - 20*(1+z)) ) :
               tpk0 = tmlc[0]-20*(1+z)
           if ( tpk > (tmlc[-1] - 10*(1+z)) and 
                tpk1 < tmlc[-1]+10*(1+z) ) :
               tpk1 = tmlc[-1]+10*(1+z)

           # if possible, expand the prior to ensure that there is
           # at least one point to the left and the right  
           tleft = [t for t in tmlc if t<tpk-1]
           if tleft : tpk0 = min( tpk0, tleft[-1] )

           tright = [t for t in tmlc if t>tpk+1]
           if tright : tpk1 = max( tpk1, tright[0] )

           # sanity check: don't allow tpk range beyond 365 days
           tpk0 = max( tpk0, tpk-365)
           tpk1 = min( tpk1, tpk+365)

           # adjust ntpk to cover the final prior range
           ntpk = int( round( (tpk1 - tpk0)/dtpk ) )

           # display the prior and ask user to approve
           if interact : 
               clf()
               self.plotlc(stack=0.001, mags=True )
               ax = gca()
               ax.set_ylim( [28, self.peak['mag']-1.5 ] )
               ax.axvline( tpk0 , lw=2, ls=':',color='k')
               ax.axvline( tpk0 + ntpk*dtpk/2., lw=2, ls='--',color='k')
               ax.axvline( tpk0 + ntpk*dtpk , lw=2, ls=':',color='k')
               ion()
               draw()
               show()
               happy = 'n'
               while not (happy=='' or happy.lower().startswith('y')) : 
                   print( "  tpk0=%.1f   ntpk=%i    dtpk=%.1f"%(
                           tpk0,ntpk,dtpk) )
                   happy = raw_input("is this a good tpk range? [y]/n :")
                   if (happy=='' or happy.lower().startswith('y')):break 

                   newval = raw_input("tpk0 [%.1f] :"%tpk0)
                   if newval != '' : 
                       try : tpk0 = float(newval)
                       except : print("input error. try again"); continue
                   ax.axvline( tpk0 , lw=3, ls='--',color='m')
                   draw()
                   show()
                   
                   newval = raw_input("ntpk [%i] :"%20)
                   ntpkOld = ntpk
                   if newval == '' : ntpk = 20
                   else : 
                       try : ntpk = int(newval)
                       except : print("input error. try again"); continue
                   newval = raw_input("dtpk [%.1f] :"%(float(ntpkOld*dtpk/ntpk)))
                   if newval == '' : dtpk = float(ntpkOld*dtpk/ntpk) 
                   else :  dtpk = float(newval)
                   ioff()
                   clf()
                   self.plotlc(stack=0.001, mags=True)
                   ax = gca()
                   ax.set_ylim( [28, self.peak['mag']-1.5 ] )
                   ax.axvline( tpk0 , lw=2, ls='--',color='m')
                   ax.axvline( tpk0 + ntpk*dtpk/2.,lw=2,ls='-',color='m')
                   ax.axvline( tpk0 + ntpk*dtpk,lw=2,ls='--',color='m')
                   draw()
                   show()
                   ion()
                   
           return(tpk0,dtpk,ntpk)


class LightCurve(object):
    """
    Light curve for a single filter.
    Magnitudes are vega-based. 
    Fluxes are 'm25' fluxes, which are referenced 
    to a zero point of m=25 in every filter. 
    """
    def __init__(self, filt):
        """ pending """
        self.filt = filt

    def flux2mag():
        """ use the flux/dflux vectors to fill 
        the mag/dmag vectors """
        from numpy import array, log10, maximum
        if 'zpt' not in self.keys() : 
            self.zpt = array([25]*len(self.date))
        self.mag = -2.5 * log10( maximum(self.flux,1e-10) ) + zpt
        self.dmag = 2.5 * log10( 1+ self.dflux/maximum(self.flux,1e-10) )

    def mag2flux(self):
        """ use the mag/dmag vectors to fill 
        the flux/dflux vectors """
        if 'zpt' not in self.keys() : 
            self.zpt = array([25]*len(self.date))
        self.flux = 10**(-0.4*(self.mag-zpt) )
        self.dflux = abs( self.flux *( 10**(0.4*self.dmag) -1 ) )

    @property
    def time(self):
        """ alias for self.date """
        if 'date' in self.__dict__.keys():
            return( self.date )
        else: return(None)

    @property
    def signoise(self):
        """ the signal to noise ratio 
        across this lightcurve """
        from numpy import array
        if 'flux' not in self.__dict__.keys(): lc.mag2flux()
        elif len(self.flux)==0 or len(self.dflux)==0 : lc.mag2flux()
        noise = []
        for df in self.dflux : 
            noise.append( max(1e-30, abs(df)) )
        return( self.flux / array(noise) )

    def plot(self, mags=False, offset=0, axislabels=False,style='points',**kwargs):
        """ 
        plot the lightcurve.
        set mags=True to plot magnitudes instead of fluxes.
        Any additional keyword args (in **kwargs) are passed to the 
        pylab.errorbar() function.
        """
        from pylab import plot, errorbar, gca, xlabel, ylabel, draw, show

        if mags :
            if( not len(self.mag) and len(self.flux) ):
                self.flux2mag()
            if( not len(self.mag) ):
                print("No LC info for filter %s"%self.filtername)
                return(-1)
            y = self.mag
            dy = self.dmag 
        else : 
            if( not len(self.flux) and len(self.mag) ):
                self.mag2flux()
            if( not len(self.flux) ) :
                print("No LC info for filter %s"%self.filtername)
                return(-1)
            y = self.flux
            dy = self.dflux 

        ax = gca()
        if 'ls' not in kwargs.keys() and 'linestyle' not in kwargs.keys() :
            kwargs['ls'] = ' '
        errorbar(self.date, y+offset, dy, **kwargs)

        if mags :
            if not ax.yaxis_inverted(): ax.invert_yaxis()
            if axislabels:
                xlabel(r"t$_{obs}$")
                ylabel(r"%s mags"%filtername)
        else :
            if axislabels:
                xlabel(r"t$_{obs}$")
                ylabel(r"%s flux (m25)"%filtername)


class Model(object):
    """
    A supernova model from a given template
    at a particular location theta=(z,mue,tpk,Av,Rv)
    and with a given flux softening factor fxsoft.
    """
    def __init__( self, template, z, mue, tpk, Av, Rv, fxsoft=0.15, EBVmw=0, debug=False):
        """ 
        load a SOFT light curve model defined by a set of spline curves
        from a .spl file, converting them into data arrays for plotting.
        """
        from numpy import argmin,argmax, array, zeros, arange, sqrt, log10
        from spline import Spline
        import os
        import exceptions
        from copy import deepcopy
        if debug : import pdb; pdb.set_trace()

        # TODO : write the R = A/E(B-V) values directly into .spl files
        from constants import aebv

        # TODO : improve Rv handling.  This is really crude.
        RvHack = Rv - 3.1

        self.template = template
        self.z = z
        self.mue = mue
        self.tpk = tpk
        self.Av = Av
        self.Rv = Rv
        self.fxsoft = fxsoft
        self.EBVmw = EBVmw

        # Read in the spline file for the given template and z,
        # storing the knots in dictionaries keyed by filter name
        splfile = os.path.abspath(
            os.path.expanduser("%s/z%5.3f/%s.spl"%(TEMPLIBDIR,z,template)) )
        self.readsplknots( splfile )

        # modified spline interpolators (accounting for location parameters Av, mue, etc)
        # get stored as functions named after the filter.
        # so self.g( 10 ) gives the g-band magnitude at 10 days, and
        # self.flux_g(-5) gives g-band flux at -5 days
        self.spl = {}
        for filt in self.tknot.keys():
            # magnitude inerpolator
            self.spl[filt] = Spline(self.tknot[filt], self.mknot[filt])
            self.mag = lambda t,filt : self.spl[filt](t-tpk) + \
                mue + Av * (self.avslope[filt]+RvHack) + aebv[filt] * EBVmw
            self.flux = lambda t, filt : 10**(-0.4 * ( self.mag(t, filt) - 25. ))

    @property
    def filters( self ):
        def sortfunc(f):
            filtnum =  { 'U':0,'B':1,'V':2,'R':3,'I':4,'Z':5,
                         'J':6,'H':7,'K':8,
                         'u':100,'g':110,'r':120,'i':130,'z':140,'y':150,
                         "u'":101,"g'":111,"r'":121,"i'":131,"z'":141,"y'":151,
                         'F606W':24,'F775W':25,'F850LP':26, 
                         'F125W':34,'F160W':35 }
            if f in filtnum : return(filtnum[f])
            else : return(99)
        inorder = sorted( self.tknot.keys(), key=sortfunc)
        return(inorder)


    def readsplknots(self, splfile ) :
        """ read in knot locations from a spline file """
        import os
        import exceptions
        from numpy import argmin,argmax, array, zeros, arange, sqrt, log10

        if not os.path.isfile( splfile ) : 
            raise exceptions.RuntimeError(
                "batm.plottemp ERROR : %s is not a file."%splfile)

        fin = open(splfile,'r')
        data = fin.readlines()
        fin.close()

        # TODO : transition to a more general format for the .spl files
        #   maybe VO table?   
        #   maybe just using #-commented header lines?
        headlist1 = data[1].split()
        headlist3 = data[3].split()
        headlist4 = data[4].split()
        tknot,mknot,avslope,dmspl = {},{},{},{}
        filters = [ headlist1[i] for i in range(0,len(headlist1),2) ] 
        for filt in filters:
            # read spline file header
            filcol = headlist1.index(filt)
            nknot = int( headlist4[filcol+1] )
            if nknot==0 :
                print("No spline fit for filter %s"%filt)
                continue
            avslope[filt] = float( headlist4[filcol] )
            dmspl[filt] = float( headlist3[filcol] )  # rms spline fit err (mags)
        
            # read spline knots
            tknot[filt] = zeros(nknot, dtype=float)
            mknot[filt] = zeros(nknot, dtype=float)
            j = 0
            for line in data[5:5+nknot] :
                linelist = line.split()
                tknot[filt][j] = float(linelist[filcol])
                mknot[filt][j] = float(linelist[filcol+1])
                j+=1
        self.tknot=tknot
        self.mknot=mknot
        self.avslope=avslope
        self.dmspl=dmspl                
        

    def plotlc(self, filters=[],
               stack=0, mags=False,
               colors=['darkorchid','blue','darkgreen','darkorange','red'], **kwargs ):
        """ 
        Plot the multicolor lightcurve. 
        """
        from pylab import legend,xlabel,ylabel,subplot,text,gca,setp,\
            subplots_adjust, axes, xticks, yticks, gcf, plot, axis, \
            rcParams, axhline
        from numpy import argmin,argmax,zeros,array,arange
        from copy import copy
        if debug : import pdb; pdb.set_trace()

        # default is to plot all filters, sorted by wavelength
        if not filters : filters = self.filters

        # Axis labels 
        if mags : ylab = 'magnitude'
        else : ylab = r'flux [$m_{25}$]'
        if (self.tref)!=0: 
            xlab = r"MJD - %.1f"%(self.tref)
        else : 
            xlab = r"t$_{obs}$"

        fig = gcf()
        nrows = 1
        ncols =1 
 
        # Stacking lightcurves on one plot?
        # find the offset for the top of the stack
        if stack : top = len(filters) / 2 * stack
        else : 
            top = 0
            # Not stacking? Set up a whole-figure axes, with 
            # invisible axis,ticks, and ticklabels, which we 
            # use to get the xlabel and ylabel in the right place
            #ax = axes([0.15,0.15,0.84,0.84],frameon=False,
            ax = axes([0.08,0.06,0.9,0.9],frameon=False,
                      xticks=[],yticks=[],
                      ylabel=ylab, xlabel=xlab  )

            # Separating lightcurves by color?
            # find the number of subplot rows and columns
            if len(filters)>2 : nrows = 2
            if len(filters)>6 : nrows = 3
            if len(filters)>1 : ncols = 2
            if len(filters)>4 : ncols = 3

        # if a light curve fit is requested, make sure it exists 
        if fit : fit=fit.lower()
        if fit=='soft':
            if not 'lcmatch' in self.__dict__.keys():
                self.getlcmatch()
            if template in ['best',None] : 
                template = self.lcmatch['template_maxlike']
            if z==None : z= self.lcmatch[template]['z_maxlike']
            if Av==None : Av= self.lcmatch[template]['Av_maxlike']
            if Rv==None : Rv= self.lcmatch[template]['Rv_maxlike']
            if mue==None : mue= self.lcmatch[template]['mue_maxlike']
            if tpk==None : tpk= self.lcmatch[template]['tpk_maxlike']
            if fxsoft==None : fxsoft= self.lcmatch[template]['fxsoft']

            # Make the model, or use the existing one if possible
            if( 'model' not in self.__dict__ ): 
                self.model = Model( template, z, mue, tpk, Av, Rv, fxsoft=fxsoft, EBVmw=self.EBVmw )
            elif( template!=self.model.template or
                  z!=self.model.z or 
                  mue!=self.model.mue or
                  tpk!=self.model.tpk or
                  Av!=self.model.Av or
                  Rv!=self.model.Rv or
                  fxsoft!=self.model.fxsoft ): 
                self.model = Model( template, z, mue, tpk, Av, Rv, fxsoft=fxsoft, EBVmw=self.EBVmw )

        elif fit=='salt':
            if 'saltmodel' not in self.__dict__ : 
                print("No SALT lc fit. Use self.doSALT")
                return(None)

        # plotting : loop over filters
        for i in range(len(filters)) : 
            filt = filters[i]
            idat = i % len(datcolors)
            if stack : 
                ax = gca()
                offset = top - stack*i
            else :
                offset=0
                if i==0 :
                    ax = fig.add_subplot(nrows, ncols, 1)
                    ax1 = ax
                else : ax = fig.add_subplot(nrows, ncols, i+1, sharey=ax1, sharex=ax1)
                text( 0.1, 0.85, r"%s"%filt, color=textcolors[idat],
                     transform=ax.transAxes,weight='heavy',
                     fontsize=14)
                if i%ncols!=0 : setp( ax.get_yticklabels(), visible=False)
                if (len(filters)-i)>ncols : setp( ax.get_xticklabels(), visible=False)

            kwargsDat = copy(kwargs)
            kwargsDat['ls']=' '
            kwargsDat['ecolor']='k'
            kwargsDat['capsize']=0
            self.mlc[filt].plot(mags=mags,  
                                offset=offset,
                                style='points',
                                color=datcolors[idat], 
                                marker=datmarkers[idat], 
                                label=filt, **kwargsDat)
        
            # Overlaying a light curve fit 
            ifit = i % len(fitcolors) 
            if fit=='salt' and (filt in self.saltlc): 
                if showmodelerr: 
                    self.saltlc[filt].plot( mags=mags, offset=offset, 
                                            style='fill', facecolor=datcolors[idat])
                self.saltlc[filt].plot( mags=mags, offset=offset, 
                                        style='line', color=fitcolors[idat], 
                                        **kwargs)
            elif fit=='soft' and (filt in self.model.filters): 
                #if showmodelerr: 
                #    self.batmlc[filt].plot( mags=mags, offset=offset, 
                #                            style='fill', facecolor=fitcolors[idat])
                self.model.plotlc( mags=mags, offset=offset, 
                                   style='line', color=fitcolors[idat], **kwargs)
            elif fit=='mlcs':
                print("MLCS FIT NOT YET AVAILABLE")
            #elif fit=='sem':
            #    if filt in self.semlc : 
            #        self.semlc[filt].plot( mags=mags, offset=offset, 
            #                               style='line', color=datcolors[idat], 
            #                               **kwargs)
            #        tknot = self.semlc[filt].tknot
            #        mknot = self.semlc[filt].mknot
            #        if not mags: mknot = 10**(-0.4*(mknot-25))
            #        plot( tknot, mknot + offset, ls=' ',marker='D',
            #              color='w', mec='k', mew=3)
        if stack:
            if showlegend: legend( borderaxespad=0.0, borderpad=0.6,
                                   numpoints=1, handlelength=0)
            if self.tref!=0 : 
                xlabel(r"MJD - %.1f"%(self.tref))
            else : 
                xlabel(r"t$_{obs}$")
            ylabel(ylab)

            if stack>=1 : 
                for i in range(len(filters)) : 
                    axhline( top-stack*i, ls='--',color='k',
                             lw=rcParams['lines.linewidth']/2)

        else : 
            # push the subplots together
            subplots_adjust(hspace=0.001)
            subplots_adjust(wspace=0.001)
            subplots_adjust(bottom=0.1)
            subplots_adjust(left=0.11)
            subplots_adjust(right=0.98)
            subplots_adjust(top=0.92)
            # return the whole-figure background axis
            return( ax )

    
class MulticolorLightCurve(dict):
    """
    A SuperNova's Multi-color Light Curve is a dictionary
    of LightCurve objects, keyed by filter name.
    """




        
    
def recast( datum ):
    """ if the given datum is a string, re-cast it as 
    a float or int if possible and return it.
    """
    if type(datum) != str : return( datum )
    if datum.isdigit() : 
        return( int(datum) )
    elif datum.replace('.','').isdigit() :
        return( float(datum) )
    else : 
        return( datum )

