"""
S.Rodney
2012.05.03

Utilities to read, plot and analyze SNANA simulations.
The __init__ module defines two classes : 
SimTable for handling the full simulation, recovered
      from binary fits tables. 
SuperNova for handling an individual supernova (either 
      simulated or real )
"""


__all__=["simplot","simulate","classify","snsed","constants"]
import snsed
import simplot
import simulate
import classify
import extinction

import constants
from constants import SNTYPEDICT

import os
import time
import pyfits
import numpy as np
from matplotlib import pyplot as p
from matplotlib import rcParams
from matplotlib import patches
import glob
import exceptions
from collections import Sequence

def timestr( time ) : 
    """ convert an int time into a string suitable for use 
    as an object parameter, replacing + with p and - with m
    """
    if time>=0 : pm = 'p'
    else : pm = 'm'
    return( '%s%02i'%(pm,abs(time) ) )


class SimTable( object ):
    """ object class for results from a SNANA simulation,
    stored in fits binary tables under  $SNDATA_ROOT/SIM
    """
    def __init__( self, simname, verbose=False ):
        self.simname = simname

        # Determine if this is a monte carlo or grid simulation
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%simname )
        headfits = os.path.join( simdatadir, '%s_HEAD.FITS'%simname)
        gridfits = os.path.join( simdatadir, '%s.GRID'%simname)

        if os.path.isfile( headfits ) :
            if verbose>1: print("%s is a RANDOM simulation."%simname)
            self.ISGRID = False
            self.readsimhead( verbose=verbose ) 
        elif os.path.isfile( gridfits ) : 
            if verbose>1: print("%s is a GRID simulation."%simname)
            self.ISGRID = True
            self.readgrid( verbose=verbose )
        else : 
            raise exceptions.RuntimeError( "Can't find either  %s  or %s"%(headfits,gridfits))

        if os.path.isfile( headfits ) : self.readdump() 
        self.readreadme()
        self.readfilters()

        # Load in the survey data from constants.py  (default to HST)
        if 'SURVEY' not in self.__dict__ : 
            if simname : self.SURVEY = self.simname.split('_')[0]
            else : self.SURVEY = 'HST'
        from constants import SurveyDataDict 
        try : 
            self.SURVEYDATA = SurveyDataDict[self.SURVEY]
        except KeyError : 
            self.SURVEYDATA = SurveyDataDict['HST']

        return( None ) 

    
    @property
    def bands( self ) : 
        """ simulated flt passes """
        # TODO : sort by wavelength
        if 'FLT' in self.__dict__ : 
            return( ''.join( np.unique( self.FLT ) ) )
        elif 'BANDS' in self.__dict__ : 
            return( self.BANDS ) 
        elif 'FILTERCURVES' in self.__dict__ : 
            return( ''.join(self.FILTERCURVES.keys()) )
        return( [ key[-1] for key in self.__dict__.keys()
                  if key.startswith('SIM_PEAKMAG_')] )

    @property
    def nsim( self ) : 
        """ number of simulated SNe """
        if 'NSIM' in self.__dict__ :  return(self.NSIM)
        elif 'NOBS' in self.__dict__ : return( len(self.NOBS) )
        elif 'NGENTOT_LC' in self.__dict__ : return( len(self.NGENTOT_LC) )
        elif 'NGEN_LC' in self.__dict__ : return( len(self.NGEN_LC) )
        return(None)

    @property
    def SIM_SUBTYPE( self ) : 
        """ the SN sub-type for each simulated object """
        from constants import SUBTYPES
        if 'non1aTypes' in self.__dict__ :
            return( np.array([ non1aType for non1aType in self.non1aTypes ] ))
        elif 'SIM_NON1a' in self.__dict__ : 
            return( np.array([ SUBTYPES[non1aCode] for non1aCode in self.SIM_NON1a ] ))
        elif not self.GENMODEL.startswith('NON') : 
            return( np.array([ 'Ia' for i in range(len(self.LUMIPAR)) ]) ) 

    @property
    def COLORPAR( self ) :
        """ for GRID sims: the color parameter grid (e.g. SALT2 c or Av)
        for MC sims: the array of color parameters for all simulated SNe"""
        if 'c' in self.__dict__ : return( self.c )
        elif 'AV' in self.__dict__ : return( self.AV )
        elif 'Av' in self.__dict__ : return( self.Av )
        elif 'SIM_SALT2c' in self.__dict__ : return( self.SIM_SALT2c )
        elif 'SIM_AV' in self.__dict__ : return( self.SIM_AV )

    @property
    def COLORLAW( self ) :
        """ for GRID sims: the color-law parameter grid (e.g. SALT2 Beta or Rv)
        for MC sims: the array of color-law parameters for all simulated SNe"""
        if 'BETA' in self.__dict__ : return( self.BETA )
        elif 'RV' in self.__dict__ : return( self.RV )
        elif 'Rv' in self.__dict__ : return( self.Rv )
        elif 'SIM_SALT2beta' in self.__dict__ : return( self.SIM_SALT2beta )
        elif 'SIM_RV' in self.__dict__ : return( self.SIM_RV )

    @property
    def LUMIPAR( self ) :
        """ for GRID sims: the luminosity parameter grid (e.g. SALT2 x1 or MLCS delta)
        for MC sims: the array of luminosity parameters for all simulated SNe
        (For non1a sims this is the non1a model ID numbers)"""
        if 'x1' in self.__dict__ : return( self.x1 )
        elif 'dm15' in self.__dict__ : return( self.dm15 )
        elif 'delta' in self.__dict__ : return( self.delta )
        elif 'SIM_SALT2x1' in self.__dict__ : return( self.SIM_SALT2x1 )
        elif 'non1aModelIDs' in self.__dict__ : return( self.non1aModelIDs )
        elif 'SIM_NON1a' in self.__dict__ : return( self.SIM_NON1a )

    def readgrid( self, verbose=False ):
        """ Read in all the simulated light curves and associated
        SN metadata (z,x1,c,etc.) from a grid simulation (GENSOURCE: GRID)
        storing everything as a set of 2-D arrays. 
           .TOBS  : obsever-frame time in days
           .FLT : filters for each observation
           .FLUXCAL : observed flux
           .FLUXCALERR: flux err
           .MAG  : observed magnitudes
           .MAGERR : mag errors.
        TOBS array has length=Nobs (a constant for all SNe), and all other arrays have 
        length=NSIM (the number of simulated SNe), and width=NOBSxNBANDS (the total number 
        of observations across all bands). 
        The parameter order for grid sequencing is :  
          (1) redshift (logz), (2) color (c or Av), (3) color law (beta or Rv), 
          (4) luminosity parameter (x1,dm15,or non1a idx)
        """
        import time
        start = time.time()

        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )

        gridfits = os.path.join( simdatadir, '%s.GRID'%self.simname)
        if not os.path.isfile( gridfits ) : 
            raise exceptions.IOError( "ERROR: %s does not exist"%gridfits )

        # SNANA GRID file contents by HDU : 

        #   Here is an example Ia simulation grid
        # No.    Name         Type       Cards   Dimensions   Format
        # 0    PRIMARY      PrimaryHDU      20   (0,)         float32   
        # 1    SNPAR-INFO   BinTableHDU     32   6R x 7C      [20A, 8A, 1I, 1E, 1E, 1E, 1J]   
        # 2    LOGZ-GRID    BinTableHDU     12   3R x 1C      [1E]   
        # 3    COLOR-GRID   BinTableHDU     12   3R x 1C      [1E]   
        # 4    RV/BETA-GRID BinTableHDU     12   1R x 1C      [1E]   
        # 5    LUMI-GRID    BinTableHDU     12   3R x 1C      [1E]   
        # 6    FILTER-GRID  BinTableHDU     12   3R x 1C      [1E]   
        # 7    TREST-GRID   BinTableHDU     12   3R x 1C      [1E]   
        # 8    PTR_I2LCMAG  BinTableHDU     14   27R x 1C     [1J]   
        # 9    I2LCMAG      BinTableHDU     15   351R x 2C    [1I, 1I]  
        
        #   Here is an example CC simulation grid
        # No.    Name          Type      Cards   Dimensions   Format
        # 0    PRIMARY      PrimaryHDU      20   (0,)         float32   
        # 1    SNPAR-INFO   BinTableHDU     32   6R x 7C      [20A, 8A, 1I, 1E, 1E, 1E, 1J]   
        # 2    LOGZ-GRID    BinTableHDU     12   4R x 1C      [1E]   
        # 3    COLOR-GRID   BinTableHDU     12   4R x 1C      [1E]   
        # 4    RV/BETA-GRID BinTableHDU     12   1R x 1C      [1E]   
        # 5    LUMI-GRID    BinTableHDU     12   27R x 1C     [1E]   
        # 6    FILTER-GRID  BinTableHDU     12   3R x 1C      [1E]   
        # 7    TREST-GRID   BinTableHDU     12   4R x 1C      [1E]   
        # 8    NONIA-INFO   BinTableHDU     21   27R x 4C     [1I, 1I, 8A, 20A]   
        # 9    PTR_I2LCMAG  BinTableHDU     14   432R x 1C    [1J]   
        # 10   I2LCMAG      BinTableHDU     15   6912R x 2C   [1I, 1I]   

        # HDU-0 : Simulation settings in the primary HDU header
        head0 = pyfits.getheader( gridfits, ext=0 )
        self.survey = head0['SURVEY']
        self.BANDS = head0['FILTERS']
        self.NSIM = head0['NTOT_LC']
        self.OMEGA_M = head0['OMEGA_M']
        self.OMEGA_DE = head0['OMEGA_DE']
        self.W0 = head0['W0']
        self.H0 = head0['H0']
        self.GENMODEL = head0['GENMODEL']
        magpack = head0['MAGPACK'] # mag = I2mag / MAGPACK
        nullmag = head0['NULLMAG'] # value for undefined model mag

        # Define an index array, starting at 1
        self.SNID = np.arange( 1, self.NSIM+1, 1, dtype=int) 
        
        # HDU-1 : grid parameter settings (min, max, Nsteps)
        #  Define object properties to hold each of the grid parameter vectors as arrays
        #  (e.g.  self.c holds an ndarray with all the grid values for the SALT2 color parameter)
        # NOTE:  HDU 2 - 7 hold redundant data arrays giving these simulation parameter grid values
        head1 = pyfits.getheader( gridfits, ext=1 )
        data1 = pyfits.getdata( gridfits, ext=1 )
        Nrow = head1['NAXIS2']    # num. of rows in the table = num. of grid parameters
        Ncol = head1['TFIELDS'] # num. of columns in the table = num. of defined values per grid parameter
        for irow in range( Nrow ) :
            rowname = head1['TTYPE%i'%(irow+1)]
            rowdata = data1[irow]
            parname = rowdata[1]
            nbin = rowdata[2]
            valmin = rowdata[3]
            valmax = rowdata[4]
            self.__dict__[ parname ] = np.linspace( valmin, valmax, num=nbin, endpoint=True)

        # convert log10(redshift) to redshift
        self.z = 10**self.LOGZ 

        # NOBS : Number of observations per band = size of TREST grid
        # NBANDS : Number of bands per simulated SN 
        # Nobstot : total number of observations per simulated SN
        self.NOBS = len( self.TREST )
        self.NBANDS = len(self.BANDS) 
        Nobstot = self.NOBS * self.NBANDS

        # For NON1A sims, read in the reference table of non1a models: 
        #    model ID number, SN type, model name
        if self.GENMODEL in ['NON1A', 'NONIA']:
            datCC = pyfits.getdata( gridfits, ext=('NONIA-INFO',1))
            self.non1aModelIDs = np.array([ dat[1] for dat in datCC ] )
            self.non1aTypes    = np.array([ dat[2] for dat in datCC ])
            self.non1aModels   = np.array([ dat[3] for dat in datCC ] )

        # Read in the light curves 
        lcptrs = pyfits.getdata( gridfits, ext=('PTR_I2LCMAG',1))
        lcdata = pyfits.getdata( gridfits, ext=('I2LCMAG',1))
        
        # store the simulated mag and mag error for each grid position
        magmatrix = []
        magerrmatrix = []
        for isn in range(self.nsim) : 
            iptr = int(lcptrs[isn][0])
            # magScaled = np.array( [lcdata[i][0] for i in range(iptr+1,iptr+1+Nobstot)], dtype=float )
            # magerrScaled = np.array( [lcdata[i][1] for i in range(iptr+1,iptr+1+Nobstot)], dtype=float )
            magScaled = np.array( lcdata['I2MAG'][iptr+1:iptr+1+Nobstot], dtype=float )
            magerrScaled = np.array( lcdata['I2MAGERR'][iptr+1:iptr+1+Nobstot], dtype=float )
            mag = magScaled / magpack
            magerr = magerrScaled / magpack
            magmatrix.append( mag.tolist() )
            magerrmatrix.append( magerr.tolist() )
        self.MAG = np.array(magmatrix)
        self.MAGERR = np.array(magerrmatrix)

        # Define an array containing the filter name for each observation, for each simulated SN
        self.FLT = np.array( [ np.ravel( [ [band]*self.NOBS for band in self.BANDS] ) for i in range(self.NSIM) ] )

        # Derive the flux and flux error in SNANA FLUXCAL units
        self.FLUXCAL = 10**(-0.4*(self.MAG-27.5))  
        self.FLUXCALERR = 1.0857362 * self.MAGERR * self.FLUXCAL
        self.FLUXCAL[ np.where( self.MAG == 32 ) ] = 0.0

        # Derive the observer-frame time (relative to peak) for each redshift grid position 
        self.TOBS = np.array( [ self.TREST * (1+10**(logz)) for logz in self.LOGZ ] )

        # convert the 2D mag and flux arrays into 5-D arrays, with one dimension for each of the
        # 4 grid parameters, one dimension for the filters, and one for the time. Thus we get one 
        # single-band light curve per cell.
        # self.MAGGRID = self.MAG.reshape( len(self.LOGZ), len(self.COLORPAR),
        #                                  len(self.COLORLAW), len(self.LUMIPAR),
        #                                  len(self.BANDS), len(self.TREST) )
        self.LCMATRIX = self.MAG.reshape( len(self.LUMIPAR), len(self.COLORLAW),
                                          len(self.COLORPAR), len(self.LOGZ),
                                          len(self.BANDS), len(self.TREST) )

        return( None )



    def readsimhead( self, verbose=False):
        """ Read in simulation data from fits header table.
        Reads header info for all SNe (redshift, type, peak mags, etc)
        """
        start = time.time()

        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )

        # read in the fits bin table headers and data
        headfits = os.path.join( simdatadir, '%s_HEAD.FITS'%self.simname)
        hhead = pyfits.getheader( headfits, ext=1 ) 
        hdata = pyfits.getdata( headfits, ext=1 ) 
        
        # collect header data columns into arrays 
        # that are accessible as object properties.
        if verbose>1 : print('Reading SN metadata (z,type,etc.)...')

        Nsn = hhead['NAXIS2']    # num. of rows in the table = num. of simulated SNe
        Nhcol = hhead['TFIELDS'] # num. of table columns  = num. of data arrays

        # Store the header data (i.e. simulation parameters) as object properties
        for colname in hdata.names : 
            self.__dict__[ colname  ] = hdata[:][colname]

        # Make some shortcut aliases to most useful metadata
        for alias, fullname in ( [['z','SIM_REDSHIFT'], ['type','SNTYPE'],['mjdpk','SIM_PEAKMJD'],
                                  ['Hpk','SIM_PEAKMAG_H'],['Jpk','SIM_PEAKMAG_J'],['Wpk','SIM_PEAKMAG_W'],
                                  ['Zpk','SIM_PEAKMAG_Z'],['Ipk','SIM_PEAKMAG_I'],['Xpk','SIM_PEAKMAG_X'],
                                  ['Vpk','SIM_PEAKMAG_V'], ]) : 
            if fullname in self.__dict__.keys() : 
                if alias=='type' : self.__dict__[alias] = np.array( [ SNTYPEDICT[self.__dict__[fullname][isn]] for isn in range(Nsn)] )
                else : self.__dict__[alias] = self.__dict__[fullname]
       
        end = time.time()
        if verbose>1: print('   Done. Read metadata for %i SNe in %i seconds.'%( Nsn, end-start ) )

        return( None )


    def readdump( self ) :
        """ read in the .DUMP file """
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        dumpfile = os.path.join( simdatadir, '%s.DUMP'%self.simname ) 
        
        if not os.path.isfile( dumpfile ) : return( None )

        fin = open( dumpfile, 'r') 
        NVAR = int( fin.readline().split(':')[1]  )
        VARNAMES = fin.readline().split(':')[1].split()
        fin.close()
        
        try : 
            dumpdat  = np.loadtxt( dumpfile, skiprows=2, unpack=False, dtype=str )
            if len(dumpdat.shape)==1 : dumpdat = np.array( [ dumpdat ] )
        except IOError : 
            return( None )
        self.DUMP = { }  
        for varname in VARNAMES : 
            varcol = VARNAMES.index( varname ) + 1 
            self.DUMP[varname] = np.array( dumpdat[:,varcol], dtype=float )
        if 'SIM_EFFMASK' in VARNAMES : 
            ieffmask = VARNAMES.index( 'SIM_EFFMASK' ) + 1
            self.DUMP['IDET_EFFMASK'] = (np.array(dumpdat[:,ieffmask], dtype=int)>=3)
            self.DUMP['IREJ_EFFMASK'] = (np.array(dumpdat[:,ieffmask], dtype=int)<3)
            self.DUMP['idet'] = self.DUMP['IDET_EFFMASK']
        elif 'CUTMASK' in VARNAMES : 
            icutmask = VARNAMES.index( 'CUTMASK' ) + 1
            self.DUMP['IDET_CUTMASK'] = (np.array(dumpdat[:,icutmask], dtype=int)==511)
            self.DUMP['IREJ_CUTMASK'] = (np.array(dumpdat[:,icutmask], dtype=int)<511)
            self.DUMP['idet'] = self.DUMP['IDET_CUTMASK']
        else : 
            self.DUMP['idet'] = range(len(self.DUMP[varname]) )
            
        return(None) 

    def bindumpdata( self, Nbins=None ):
        """  Use numpy.digitize to sort the .DUMP data (read in via .readdump) 
        into bins based on their principal parameters of the simulated SNe:   
            REDSHIFT, MU, AV, S2x1, S2c 

        Fills  a dict .DUMPBINS with those 5 parameters as keywords.
        The .DUMPBINS dict  holds two lists for each parameter:
         list1 : the bin edges (a la pylab.histogram)
                 length = number of bins for this parameter
         list2 : the assigned bin number (first bin is 1) for each simulated SN
                 length = number of simulated SNe
         list3 : each list element is an array carrying the SN CID values 
                 for all simulated SNe landing in that bin. 
                 length = number of bins for this parameter

        Nbins specifies the number of bins, either as a scalar (applied to
         all parameters) or a dict (keyed by parameter name). If left empty 
         then it is set to the sqrt of the number of objects.
        """
        #  Parameters to bin over
        #   z  : SIM_REDSHIFT_FINAL, .DUMP['Z']
        #   mu : SIM_DLMU,  .DUMP['MU']
        #   Av : .DUMP['AV']
        #   x1 : SIM_SALT2x1, .DUMP['S2x1']  [for SNIa only, -9's in DUMP for CC]
        #    c : SIM_SALT2c, .DUMP['S2c']   [for SNIa only, -9's in DUMP for CC]

        if 'DUMP' not in self.__dict__ : self.readdump()
        keylist = ['Z', 'MU' ,'AV','S2x1','S2c']

        if not Nbins : 
            Nbins = dict( [(k,np.sqrt(self.nsim)) for k in keylist] )
        elif not np.iterable( Nbins ) : 
            Nbins = [ Nbins for k in keylist]

        outdict = {}
        for key  in  keylist: 
            valarray = self.DUMP[key]
            nbin = Nbins[key]
            bins = np.linspace( valarray.min() , valarray.max(), nbin )
            binindex = np.digitize( valarray, bins ) 
            cidvals = [ self.DUMP['CID'][np.where( binindex==ibin+1 )[0]].astype(int)
                        for ibin in range(len(bins)) ]
            outdict[key] = [bins, binindex, cidvals]
        self.DUMPBINS = outdict 
        return( None ) 


    def readfilters( self, allhstbands=False, verbose=False ) :
        """ read in the filter definition  .dat files """
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        filtdatadir = os.path.join( simdatadir,'%s.FILTERS'%self.simname )

        if allhstbands: 
            from simulate import filter2band
            filtdatadir = os.path.join( sndataroot, 'filters/HST' )

        self.FILTERCURVES = {}
        datfiles = glob.glob( os.path.join( filtdatadir, '*.dat' ) )
        if verbose : print( "Reading %i filter curves"%len(datfiles) )
        for datfile in datfiles : 
            if allhstbands : 
                basename =  os.path.basename( datfile ).split('.')[0]
                det = basename.split('_')[0]
                if det not in ['WFC3','ACS'] : continue
                try : band = filter2band( basename.split('_')[-1] )
                except : continue
            else : 
                band = os.path.basename( datfile ).split('.')[0]

            self.FILTERCURVES[ band ] = np.loadtxt( datfile ) 
        return(None) 



    def readreadme( self ) :
        """ read in useful numbers from the .README file """
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        dumpfile = os.path.join( simdatadir, '%s.README'%self.simname ) 
        
        if not os.path.isfile( dumpfile ) : return( None )

        fin = open( dumpfile, 'r') 
        readmelines = fin.readlines()
        fin.close()

        for line in readmelines : 
            if ("Number of SNe per season" in line) or ("Number of SN per season" in line): 
                # Number of simulated SN events within the survey volume and time
                if '=' in line : 
                    try : self.NSNSURVEY = int(line.split('=')[1])
                    except : self.NSNSURVEY = float(line.split('=')[1])
                elif ':' in line : 
                    try : self.NSNSURVEY = int(line.split(':')[1].split()[0])
                    except : self.NSNSURVEY = float(line.split(':')[1].split()[0])
            elif "Survey Volume" in line: 
                self.SURVEY_VOLUME = float(line.split('=')[1].split()[0])
            elif "Survey Time" in line: 
                self.SURVEY_TIME = float(line.split('=')[1].split()[0])
            elif "Generated" in line and "simulated light curves" in line : 
                self.NGENTOT = int( line.split()[1] )
                

        return(None) 


    def getSupernovae( self, verbose=False, clobber=False ):
        """ read in all the simulated light curves, storing them
        as a dict of snana.SuperNova objects, keyed by the supernova ID
        """
        import time
        from math import log10
        start = time.time()
        snidlength = int( log10( len(self.SNID) ) ) + 1
        for snid in self.SNID : 
            snidname = 'SN'+str(snid).zfill(snidlength)
            self.__dict__[snidname] = SuperNova( simname=self.simname, snid=int(snid) ) 
        end = time.time()
        if verbose: print("Read in %i SN objects in %i seconds"%(len(self.SNID), int(end-start) ) )
        

    def getClassCurves( self, **kwarg) : 
        """ alias for getLightCurves """
        self.getLightCurves( **kwarg )

    def getLightCurves( self, store2d=True, verbose=False, clobber=False ):
        """ read in all the simulated light curves from a monte carlo 
        simulation (GENSOURCE RANDOM), storing them as a set of arrays:
           .MJD  : dates of observation
           .FLT : filters for each observation
           .FLUXCAL : observed flux
           .FLUXCALERR: flux err
           .MAG  : observed magnitudes
           .MAGERR : mag errors.
           .ZEROPT : zero points

        store2d :  if every simulated light curve has identical observation 
          dates and filters (as is the case for classification sims generated 
          with classify.doMonteCarloSim) then you can set store2d=True to 
          reshape the light curve data into 2-d numpy.ndarrays, allowing 
          for very efficient comparison to an observed light curve with 
          the same observation sequence.
        """
        import time
        start = time.time()

        headcolumns     = ['PTROBS_MIN','PTROBS_MAX','NOBS','SNID','SNTYPE','MWEBV','PEAKMJD',
                           'SIM_MODEL_NAME','SIM_MODEL_INDEX', 'SIM_NON1a', 'SIM_REDSHIFT',
                           'SIM_DLMU', 'SIM_SALT2mB','SIM_SALT2x0','SIM_SALT2x1', 'SIM_SALT2c', 
                           'SIM_AV', 'SIM_RV', ]
        photcolumns = ['MJD','FLT','FLUXCAL','FLUXCALERR','MAG','MAGERR','ZEROPT']

        # read in the fits bin table headers and data
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        headfits = os.path.join( simdatadir, '%s_HEAD.FITS'%self.simname)
        hhead = pyfits.getheader( headfits, ext=1 ) # header info from the head table
        photfits = os.path.join( simdatadir, '%s_PHOT.FITS'%self.simname)
        phead = pyfits.getheader( photfits, ext=1 )  # header info from the photometry table

        Nsn = hhead['NAXIS2']    # num. of rows in the table = num. of simulated SNe
        # Nhcol = hhead['TFIELDS'] # num. of hdr table columns  = num. of header data arrays
        # Npcol = phead['TFIELDS'] # num. of phot table columns  = num. of phot data arrays

        headdat  = pyfits.getdata( headfits, ext=1 )   # all the header data
        photdat = pyfits.getdata( photfits, ext=1 )

        # Store the header data (i.e. simulation parameters) as object properties
        for colname in headcolumns : 
            if colname not in headdat.names : continue
            self.__dict__[ colname  ] = headdat[:][colname]            

        # Read in all the photometry data and store it in 2-D arrays as object attributes
        # with one array per column for each simulated SN
        Nrows = len(photdat['MJD'])

        goodrows =  [ irow for irow in range(Nrows) if irow not in self.PTROBS_MAX ]
        for colname in photcolumns : 
            if colname not in photdat.names : continue
            self.__dict__[ colname  ] = photdat[goodrows][colname]

        # Reshape identically-shaped light curves into 2-D arrays for efficient comparison
        # with the observed light curve (i.e. for computing classification probabilities)
        Nobs = self.NOBS[0]
        if store2d : 
            if verbose>1 : 
                print("Reshaping simulated SN photometry data into %i x %i 2-D arrays"%( Nsn, Nobs ))
            # Sanity check : make sure every light curve has the same number of observations
            nobsmean = np.mean(self.PTROBS_MAX+1 - self.PTROBS_MIN)
            nobsstd = np.std(self.PTROBS_MAX+1 - self.PTROBS_MIN)
            NrowsExpected = Nsn * (Nobs+1)
            if not ( NrowsExpected==Nrows and nobsstd==0 and np.std(self.NOBS)==0 and nobsmean == np.mean(self.NOBS) ): 
                raise exceptions.RuntimeError( 
                    "ERROR: unequal light curve lengths in %s. Cannot store light curves as 2D arrays."%self.simname )
            for colname in photcolumns : 
                if colname not in photdat.names : continue
                self.__dict__[ colname  ] = self.__dict__[ colname  ].reshape( Nsn, Nobs )            
        
        end = time.time()
        if verbose: print("   Read in photometry for %i SN objects in %i seconds"%(Nsn, int(end-start) ) )
        return( None )


    def samplephot( self, mjd, tmatch=3, bandlist=None, 
                    verbose=False, clobber=False ) :
        """ 
        Sample the photometry for all simulated SNe at the given
        mjd date (to within +-tmatch days)

        TODO: If there is no observation on the specified date, we use 
            linear interpolation from nearest obs. 

        Photometry samples get recorded in numpy.MaskedArrays accessible 
        as object properties named as self.X<MJD> , where 
        X is the SNANA filter ID and <MJD> is the mjd sample date. 
        e.g. :  sn.B55750 = B band mags for all simulated SNe at MJD=55750+-3
        """
        # TODO : add an option for linear interpolation instead of nearest obs 
        from simplot import timestr
        import time
        start = time.time()

        if bandlist==None : bandlist = self.bands
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )

        # collect header table data
        headfits = os.path.join( simdatadir, '%s_HEAD.FITS'%self.simname)
        hhead = pyfits.getheader( headfits, ext=1 ) 
        hdata = pyfits.getdata( headfits, ext=1 ) 

        Nsn = hhead['NAXIS2']    # num. of rows in the table = num. of simulated SNe
        Nhcol = hhead['TFIELDS'] # num. of table columns  = num. of data arrays
        snidlist = np.array([ int( hdata[isn]['SNID'] ) for isn in range(Nsn) ])

        # collect  phot table data
        photfits = os.path.join( simdatadir, '%s_PHOT.FITS'%self.simname)
        phead = pyfits.getheader( photfits, ext=1 ) 
        pdata = pyfits.getdata( photfits, ext=1 ) 
        Npcol = phead['TFIELDS'] # num. of table columns  = num. of data arrays

        # find which columns in the phot table have the data we need:
        #   bandpass, date, and mag for each observation
        for ipcol in range( Npcol ) : 
            if phead['TTYPE%i'%(ipcol+1)] == 'FLT' : ibandcol = ipcol
            elif phead['TTYPE%i'%(ipcol+1)] == 'MJD' : imjdcol = ipcol
            elif phead['TTYPE%i'%(ipcol+1)] == 'MAG' : imagcol = ipcol

        samplebandlist = []
        for band in bandlist : 
            # check if we have already built a parameter holding the array 
            # of mags sampled from the light curves at this tobs
            samplemagkey = '%s%i'%(band, int(mjd) )
            if samplemagkey in self.__dict__.keys() and not clobber: continue
            samplebandlist.append( band )
            self.__dict__[samplemagkey] = []

        if not len(samplebandlist) : 
            if verbose: print("Sampled photometry for mjd=%i exists. Not clobbering."%int(mjd))
            return(None) 

        bandobs = [pdata['FLT'][hdata['PTROBS_MIN'][i]-1:hdata['PTROBS_MAX'][i]] for i in np.array(self.SNID).astype( int )-1]
        mjdobs = [pdata['MJD'][hdata['PTROBS_MIN'][i]-1:hdata['PTROBS_MAX'][i]] for i in np.array(self.SNID).astype( int )-1]
        magobs = [pdata['MAG'][hdata['PTROBS_MIN'][i]-1:hdata['PTROBS_MAX'][i]] for i in np.array(self.SNID).astype( int )-1]

        for i in range(len(np.array(self.SNID).astype( int ))):
            tpk = hdata[i]['PEAKMJD']
            for band in samplebandlist : 
                samplemagkey = '%s%i'%(band, int(mjd) )
                samplemag = self.__dict__[samplemagkey]

                # isolate observations in this filter
                ithisband = np.where( (bandobs[i]==band) & (magobs[i]<99) & (magobs[i]>-99))[0]
                if not len(ithisband) : 
                    if verbose>1: print("WARNING : no good data for SN %i in flt %s"%(i, band))
                    samplemag.append( -99.0 )
                    continue

                # locate the observation date and mag nearest to the specified MJD
                inearest = np.argmin( np.abs(mjdobs[i][ithisband]-mjd) )
                mjdnearest = mjdobs[i][ithisband][inearest] 
                magnearest = magobs[i][ithisband][inearest] 

                # reject if nearest measurement is not within
                # 'tmatch' days of the requested sample date 'tobs'
                if abs(mjdnearest - mjd) > tmatch : 
                    samplemag.append( -99.0 )
                    continue

                # This is a good observation. Append the nearest mag to our sample list
                samplemag.append( magnearest )
                continue
            continue

        # convert lists to numpy masked arrays, where bad values (mag==-99) are masked
        for band in samplebandlist : 
            samplemagkey = '%s%i'%(band, int(mjd) )
            a = np.array( self.__dict__[samplemagkey] )
            self.__dict__[samplemagkey] = np.ma.masked_where( a<-98, a )

        end = time.time()
        if verbose: print("Sampled photometry for %i SNe in %i seconds"%(len(self.SNID), int(end-start) ) )

        return(None) 

        
class SuperNova( object ) : 
    """ object class for a single SN extracted from SNANA sim tables
        or from a SNANA-style .DAT file
    """
    def __init__( self, datfile=None, simname=None, snid=None, verbose=False ) : 
        """ Read in header info (z,type,etc) and full light curve data.
        For simulated SNe stored in fits tables, user must provide the simname and snid,
        and the data are collected from binary fits tables, assumed to exist 
        within the $SNDATA_ROOT/SIM/ directory tree. 
        For observed or simulated SNe stored in ascii .dat files, user must provide 
        the full path to the datfile.
        """
        if not (datfile or (snid and simname)) : 
            if verbose:  print("No datfile or simname provided. Returning an empty SuperNova object.")                

        # Set some defaults for parameters that might not otherwise be defined
        self.GRADE='A'
        self.DECLINER=False
        self.HOST_MORPHOLOGY='u' # u:unclassifiable
        self.HOST_SEDTYPE=None # not known

        if simname and snid :
            if verbose : print("Reading in data from binary fits tables for %s %s"%(simname, str(snid)))
            self.simname = simname
            self.snid = snid
            # read in header and light curve data from binary fits tables 
            gothd = self.getheadfits( ) 
            gotphot = self.getphotfits( )
            if not (gothd and gotphot) : 
                gotgrid = self.getgridfits()
                if not gotgrid : 
                    print("Unable to read in data for %s %s.  No sim product .fits files found."%(simname, str(snid)))
        elif datfile :  
            if verbose : print("Reading in data from light curve file %s"%(datfile))
            self.readdatfile( datfile ) 

        # Load in the survey data from constants.py  (default to HST)
        if 'SURVEY' not in self.__dict__ : 
            if simname : self.SURVEY = simname.split('_')[0]
            else : self.SURVEY = 'HST'
        from constants import SurveyDataDict 
        try : 
            self.SURVEYDATA = SurveyDataDict[self.SURVEY]
        except KeyError : 
            self.SURVEYDATA = SurveyDataDict['HST']



    @property
    def name(self):
        if 'NAME' in self.__dict__ :
            return( self.NAME )
        elif 'SNID' in self.__dict__ :
            return( self.SNID )
        elif 'NICKNAME' in self.__dict__ :
            return( self.NICKNAME )
        elif 'CID' in self.__dict__ :
            return( self.CID )
        elif 'IAUNAME' in self.__dict__ :
            return( self.IAUNAME )
        else : 
            return( '' )

    @property
    def nickname(self):
        if 'NICKNAME' in self.__dict__ :
            return( self.NICKNAME )
        elif 'NAME' in self.__dict__ :
            return( self.NAME )
        elif 'IAUNAME' in self.__dict__ :
            return( self.IAUNAME )
        elif 'SNID' in self.__dict__ :
            return( self.SNID )
        elif 'CID' in self.__dict__ :
            return( self.CID )
        else : 
            return( '' )
            
    @property
    def bandlist(self):
        if 'FLT' in self.__dict__ :
            return( np.unique( self.FLT ) )
        else : 
            return( np.array([]))

    @property
    def bands(self):
        return( ''.join(self.bandlist) )

    @property 
    def BANDORDER(self):
        return( self.SURVEYDATA.BANDORDER )

    @property 
    def bandorder(self):
        return( self.SURVEYDATA.BANDORDER )

    @property
    def signoise(self):
        """ compute the signal to noise curve"""
        if( 'FLUXCALERR' in self.__dict__ and 
            'FLUXCAL' in self.__dict__  ) :
            return( self.FLUXCAL / np.abs(self.FLUXCALERR) )
        else: 
            return( None)
    
    @property
    def pkmjd(self):
        if 'PEAKMJD' in self.__dict__.keys() :
            return( self.PEAKMJD )
        elif 'SIM_PEAKMJD' in self.__dict__.keys() :
            if type(self.SIM_PEAKMJD)==str:
                return( float(self.SIM_PEAKMJD.split()[0]) )
            return( self.SIM_PEAKMJD )
        else : 
            return( self.pkmjdobs )

    @property
    def pkmjderr(self):
        if 'PEAKMJDERR' in self.__dict__.keys() :
            return( self.PEAKMJDERR )
        elif 'SIM_PEAKMJDERR' in self.__dict__.keys() :
            return( self.SIM_PEAKMJDERR )
        elif 'SIM_PEAKMJD_ERR' in self.__dict__.keys() :
            return( self.SIM_PEAKMJD_ERR )
        else : 
            return( max( self.pkmjdobserr, 1.5*abs(self.pkmjdobs-self.pkmjd)  ) )

    @property
    def pkmjdobs(self):
        if 'SEARCH_PEAKMJD' in self.__dict__.keys() :
            return( self.SEARCH_PEAKMJD )
        elif 'SIM_PEAKMJD' in self.__dict__.keys() :
            return( self.SIM_PEAKMJD )
        else : 
            # crude guess at the peak mjd as the date of highest S/N
            return( self.MJD[ self.signoise.argmax() ] )

    @property
    def pkmjdobserr(self):
        if 'SEARCH_PEAKMJDERR' in self.__dict__.keys() :
            return( self.SEARCH_PEAKMJDERR )
        elif 'SEARCH_PEAKMJD_ERR' in self.__dict__.keys() :
            return( self.SEARCH_PEAKMJD_ERR )
        else : 
            # determine the peak mjd uncertainty
            ipk = self.signoise.argmax()
            pkband = self.FLT[ ipk ]
            ipkband = np.where(self.FLT==pkband)[0]
            mjdpkband = np.array( sorted( self.MJD[ ipkband ] ) )
            if len(ipkband)<2 : return( 30 )
            ipkidx = ipkband.tolist().index( ipk )
            if ipkidx == 0 : 
                return( 0.7*(mjdpkband[1]-mjdpkband[0]) )
            elif ipkidx == len(ipkband)-1 : 
                return( 0.7*(mjdpkband[-1]-mjdpkband[-2]) )
            else : 
                return( 0.7*0.5*(mjdpkband[ipkidx+1]-mjdpkband[ipkidx-1]) )

    @property
    def mjdpk(self):
        return( self.pkmjd )

    @property
    def mjdpkerr(self):
        return( self.pkmjderr )

    @property
    def mjdpkobs(self):
        return( self.pkmjdobs )

    @property
    def mjdpkobserr(self):
        return( self.pkmjdobserr )

    @property
    def isdecliner(self):
        if 'DECLINER' in self.__dict__ : 
            if self.DECLINER in ['True','TRUE',1] : return( True )
            else : return( False )
        if self.pkmjd < self.MJD.min() : return( True ) 
        else : return( False )

    @property
    def zphot(self):
        zphot = None
        for key in ['HOST_GALAXY_PHOTO-Z','ZPHOT']:
            if key in self.__dict__.keys() : 
                hostphotz = self.__dict__[key] 
                if type( hostphotz ) == str : 
                    zphot = float(hostphotz.split()[0])
                    break
                else : 
                    zphot = float(hostphotz)
                    break
        if zphot>0 : return( zphot ) 
        else : return( 0 ) 

    @property
    def zphoterr(self):
        zphoterr=None
        for key in ['HOST_GALAXY_PHOTO-Z_ERR','ZPHOTERR']:
            if key in self.__dict__.keys() : 
                zphoterr = float(self.__dict__[key])
                break
        if not zphoterr : 
            for key in ['HOST_GALAXY_PHOTO-Z',]:
                if key in self.__dict__.keys() : 
                    hostphotz = self.__dict__[key] 
                    if type( hostphotz ) == str : 
                        zphoterr = float(hostphotz.split()[2])
                        break 
        if zphoterr>0 : return( zphoterr ) 
        else : return( 0 ) 

    @property
    def zspec(self):
        zspec=None
        for key in ['HOST_GALAXY_SPEC-Z','SN_SPEC-Z','ZSPEC']:
            if key in self.__dict__.keys() : 
                hostspecz = self.__dict__[key] 
                if type( hostspecz ) == str : 
                    zspec = float(hostspecz.split()[0])
                    break
                else : 
                    zspec = float(hostspecz)
                    break
        if zspec>0 : return( zspec ) 
        else : return( 0 ) 

    @property
    def zspecerr(self):
        zspecerr=None
        for key in ['HOST_GALAXY_SPEC-Z_ERR','SN_SPEC-Z_ERR','ZSPECERR']:
            if key in self.__dict__.keys() : 
                zspecerr = float(self.__dict__[key])
                break
        if not zspecerr : 
            for key in ['HOST_GALAXY_SPEC-Z','SN_SPEC-Z']:
                if key in self.__dict__.keys() : 
                    specz = self.__dict__[key] 
                    if type( specz ) == str : 
                        zspecerr = float(specz.split()[2])
                        break 
        if zspecerr>0 : return( zspecerr )
        else : return( 0 ) 

    @property
    def z(self):
        zfin = None
        if 'REDSHIFT_FINAL' in self.__dict__ : 
            if type( self.REDSHIFT_FINAL ) == str : 
                zfin = float(self.REDSHIFT_FINAL.split()[0])
            else : 
                zfin = float(self.REDSHIFT_FINAL)
            if zfin > 0 : return( zfin ) 
        elif 'REDSHIFT' in self.__dict__ : return( self.REDSHIFT ) 
        elif self.zspec > 0 : return( self.zspec ) 
        elif self.zphot > 0 : return( self.zphot ) 
        elif 'SIM_REDSHIFT' in self.__dict__ : return( self.SIM_REDSHIFT ) 
        else : return( 0 )

    @property
    def zerr(self):
        # TODO : better discrimination of possible redshift labels
        if ( 'REDSHIFT_FINAL' in self.__dict__.keys() and
             type( self.REDSHIFT_FINAL ) == str ):
            return( float(self.REDSHIFT_FINAL.split()[2]))
        elif ( 'REDSHIFT_ERR' in self.__dict__.keys() ): 
            if type( self.REDSHIFT_ERR ) == str :
                return( float(self.REDSHIFT_ERR.split()[0]) )
            else : 
                return(self.REDSHIFT_ERR)
        if self.zspecerr > 0 : return( self.zspecerr ) 
        elif self.zphoterr > 0 : return( self.zphoterr ) 
        else : return( 0 )

    @property
    def z68(self):
        """ 68% redshift confidence limits """
        if 'HOST_GALAXY_Z68' in self.__dict__ : 
            zmin,zmax = self.HOST_GALAXY_Z68.split()[:2]
            return( float(zmin), float(zmax))
        else : 
            return( None )

    @property
    def z95(self):
        """ 95% redshift confidence limits """
        if 'HOST_GALAXY_Z95' in self.__dict__ : 
            zmin,zmax = self.HOST_GALAXY_Z95.split()[:2]
            return( float(zmin), float(zmax))
        else : 
            return( None )

    @property
    def nobs(self):
        return( len(self.FLUXCAL) )

    @property
    def chi2_ndof(self):
        """ The reduced chi2. 
        !! valid only for models that have been fit to observed data !!
        """
        if 'CHI2VEC' in self.__dict__ and 'NDOF' in self.__dict__ : 
            return( self.CHI2VEC.sum() / self.NDOF )
        elif 'CHI2' in self.__dict__ and 'NDOF' in self.__dict__ : 
            return( self.CHI2 / self.NDOF )
        else : 
            return( 0 ) 

    @property
    def chi2(self):
        """ The raw (unreduced) chi2. 
        !! valid only for models that have been fit to observed data !!
        """
        if 'CHI2VEC' in self.__dict__  : 
            return( self.CHI2VEC.sum() )
        elif 'CHI2' in self.__dict__  : 
            return( self.CHI2 )
        else : 
            return( 0 ) 
            
    def readdatfile(self, datfile ):
        """ read the light curve data from the SNANA-style .dat file.
        Metadata in the header are in "key: value" pairs
        Observation data lines are marked with OBS: 
        and column names are given in the VARLIST: row.
        Comments are marked with #.
        """
        # TODO : could make the data reading more general: instead of assuming the 6 known 
        #   columns, just iterate over the varlist.
            
        from numpy import array,log10,unique,where

        if not os.path.isfile(datfile): raise exceptions.RuntimeError( "%s does not exist."%datfile) 
        self.datfile = os.path.abspath(datfile)
        fin = open(datfile,'r')
        data = fin.readlines()
        fin.close()
        flt,mjd=[],[]
        fluxcal,fluxcalerr=[],[]
        mag,magerr=[],[]

        # read header data and observation data
        for i in range(len(data)):
            line = data[i]
            if(len(line.strip())==0) : continue
            if line.startswith("#") : continue 
            if line.startswith('END:') : break
            if line.startswith('VARLIST:'):
                colnames = line.split()[1:]
                for col in colnames : 
                    self.__dict__[col] = []
            elif line.startswith('NOBS:'):
                nobslines = int(line.split()[1])
            elif line.startswith('NVAR:'):
                ncol = int(line.split()[1])
            elif line.startswith('OBS:') : 
                obsdat = line.split()[1:]
                for col in colnames : 
                    icol = colnames.index(col)
                    self.__dict__[col].append( str2num(obsdat[icol]) )
            else : 
                colon = line.find(':')
                key = line[:colon].strip()
                val = line[colon+1:].strip()
                self.__dict__[ key ] = str2num(val)

        for col in colnames : 
            self.__dict__[col] = array( self.__dict__[col] )
        return( None )

    def writedatfile(self, datfile, mag2fluxcal=False, **kwarg ):
        """ write the light curve data into a SNANA-style .dat file.
        Metadata in the header are in "key: value" pairs
        Observation data lines are marked with OBS: 
        and column names are given in the VARLIST: row.
        Comments are marked with #.

        mag2fluxcal : convert magnitudes and errors into fluxcal units
          and update the fluxcal and fluxcalerr arrays before writing
        """
        from numpy import array,log10,unique,where

        if mag2fluxcal : 
            from .. import hstsnphot
            self.FLUXCAL, self.FLUXCALERR = hstsnphot.mag2fluxcal( self.MAG, self.MAGERR ) 

        fout = open(datfile,'w')
        for key in ['SURVEY','NICKNAME','SNID','IAUC','PHOTOMETRY_VERSION',
                    'SNTYPE','FILTERS','MAGTYPE','MAGREF','DECLINER',
                    'RA','DECL','MWEBV','REDSHIFT_FINAL',
                    'HOST_GALAXY_PHOTO-Z','HOST_GALAXY_SPEC-Z','REDSHIFT_STATUS',
                    'SEARCH_PEAKMJD','SEARCH_PEAKMJDERR',
                    'PEAKMJD','PEAKMJDERR',
                    'SIM_SALT2c','SIM_SALT2x1','SIM_SALT2mB','SIM_SALT2alpha',
                    'SIM_SALT2beta','SIM_REDSHIFT','SIM_PEAKMJD','HOST_SEDTYPE',
                    'HOST_MORPHOLOGY','HOST_B-K','HOST_MK', 'HOST_REFF', 
                    'HOST_GALAXY_Z68','HOST_GALAXY_Z95','GRADE'] : 
            if key in kwarg : 
                print>> fout, '%s: %s'%(key,str(kwarg[key]))
            elif key in self.__dict__ : 
                print>> fout, '%s: %s'%(key,str(self.__dict__[key]))
        print>>fout,'\nNOBS: %i'%len(self.MAG)
        print>>fout,'NVAR: 7'
        print>>fout,'VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     MAGERR\n'

        for i in range(self.nobs):
            print>>fout, 'OBS: %9.3f  %s  %s %8.3f %8.3f %8.3f %8.3f'%(
                self.MJD[i], self.FLT[i], self.FIELD[i], self.FLUXCAL[i], 
                self.FLUXCALERR[i], self.MAG[i], self.MAGERR[i] )
        print >>fout,'\nEND:'
        fout.close()
        return( datfile )

    def getheadfits( self ) :
        """ read header data for the given SN from a binary fits table
        generated using the SNANA Monte Carlo simulator (GENSOURCE=RANDOM)
        """
        # read in the fits bin table headers and data
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        headfits = os.path.join( simdatadir, '%s_HEAD.FITS'%self.simname)
        if not os.path.isfile( headfits ) : return(False)
        hhead = pyfits.getheader( headfits, ext=1 ) 
        hdata = pyfits.getdata( headfits, ext=1 ) 

        # collect header data into object properties.
        Nsn = hhead['NAXIS2']    # num. of rows in the table = num. of simulated SNe
        Nhcol = hhead['TFIELDS'] # num. of table columns  = num. of data arrays
        snidlist = np.array([ int( hdata[isn]['SNID'] ) for isn in range(Nsn) ])
        if self.snid not in snidlist : 
            raise exceptions.RuntimeError( "SNID %s is not in %s"%(self.snid,headfits) )
        isn = np.where( snidlist== self.snid )[0][0]
        for ihcol in range( Nhcol ) : 
            self.__dict__[ hhead['TTYPE%i'%(ihcol+1)] ] = hdata[isn][ihcol] 
        
        # Make some shortcut aliases to most useful metadata
        for alias, fullname in ( [['z','SIM_REDSHIFT'], ['type','SNTYPE'],['mjdpk','SIM_PEAKMJD'],
                                  ['Hpk','SIM_PEAKMAG_H'],['Jpk','SIM_PEAKMAG_J'],['Wpk','SIM_PEAKMAG_W'],
                                  ['Zpk','SIM_PEAKMAG_Z'],['Ipk','SIM_PEAKMAG_I'],['Xpk','SIM_PEAKMAG_X'],['Vpk','SIM_PEAKMAG_V'],
                                  ]) : 
            if fullname in self.__dict__.keys() : 
                if alias=='type' : self.__dict__[alias] = SNTYPEDICT[self.__dict__[fullname]]
                else : self.__dict__[alias] = self.__dict__[fullname]
        return(True)

    def getphotfits( self ) :
        """ read phot data for the given SN from the photometry fits table
        generated using the SNANA Monte Carlo simulator (GENSOURCE=RANDOM)
        """
        # read in the fits bin table headers and data
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        photfits = os.path.join( simdatadir, '%s_PHOT.FITS'%self.simname)
        if not os.path.isfile( photfits ) : return(False)

        phead = pyfits.getheader( photfits, ext=1 ) 
        pdata = pyfits.getdata( photfits, ext=1 ) 

        # find pointers to beginning and end of the obs sequence
        sndata = pdata[ self.PTROBS_MIN-1:self.PTROBS_MAX ]

        # collect phot data into object properties.
        Npcol = phead['TFIELDS'] # num. of table columns  = num. of data arrays
        for ipcol in range( Npcol ) : 
            self.__dict__[ phead['TTYPE%i'%(ipcol+1)] ] = np.array( [ sndata[irow][ipcol] for irow in range(self.NOBS) ] )
        return( True )

    def getgridfits( self ) : 
        """ read phot data for this SN from the binary fits table
        generated using the SNANA Grid simulator (GENSOURCE=GRID)
        """
        # read in the fits bin table headers and data
        sndataroot = os.environ['SNDATA_ROOT']
        simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simname )
        gridfits = os.path.join( simdatadir, '%s.GRID'%self.simname)
        if not os.path.isfile( gridfits ) : return(False)
        
        isn = int(self.snid) - 1  # index of this SN in the grid

        # HDU-0 : Simulation settings in the primary HDU header
        head0 = pyfits.getheader( gridfits, ext=0 )
        self.survey = head0['SURVEY']
        self.BANDS = head0['FILTERS']
        self.OMEGA_M = head0['OMEGA_M']
        self.OMEGA_DE = head0['OMEGA_DE']
        self.W0 = head0['W0']
        self.H0 = head0['H0']
        self.GENMODEL = head0['GENMODEL']
        magpack = head0['MAGPACK'] # mag = I2mag / MAGPACK
        nullmag = head0['NULLMAG'] # value for undefined model mag
        
        # HDU-1 : grid parameter settings (min, max, Nsteps)
        # NOTE:  HDU 2 - 7 hold redundant data arrays giving these simulation parameter grid values
        head1 = pyfits.getheader( gridfits, ext=1 )
        data1 = pyfits.getdata( gridfits, ext=1 )
        Nrow = head1['NAXIS2']  # num. of rows in the table = num. of grid parameters
        #Ncol = head1['TFIELDS'] # num. of columns in the table = num. of defined values per grid parameter

        # Every grid has four dimensions defining four GRID parameters : 
        # (1) redshift, (2) color (AV or c), (3) color law (RV or Beta) and (4) luminosity parameter
        # For a NON1A GRID the luminosity parameter dimension is used to store 
        # a sparse index that runs from 1 to the number of non-Ia templates that 
        # are specified with the NON1A: keyword in the sim-input file. 
        gridDim = [ data1[irow][2] for irow in range(4) ] 
        offlp = gridDim[2] * gridDim[1] * gridDim[0]
        offcl = gridDim[1] * gridDim[0]
        offcp = gridDim[0]
        ilp = int( (isn) / offlp ) # LUMIPAR
        icl = int( (isn-(ilp*offlp)) / offcl ) # COLORPAR
        icp = int( (isn-(ilp*offlp)-(icl*offcl)) / offcp ) # COLORLAW
        iz = isn-(ilp*offlp)-(icl*offcl)-(icp*offcp) # REDSHIFT
        paridx = [ iz, icp, icl, ilp ]  #indices for each of the four GRID parameters

        #  Define object properties to hold each of the grid parameters
        #  (e.g.  self.c holds a scalar with the value of the SALT2 color parameter for this SN)
        for irow in range( Nrow ) :
            rowname = head1['TTYPE%i'%(irow+1)]
            rowdata = data1[irow]
            parname = rowdata[1]
            nbin = rowdata[2]
            valmin = rowdata[3]
            valmax = rowdata[4]
            paramarray = np.linspace( valmin, valmax, num=nbin, endpoint=True)
            if irow<4 : 
                # Store the 4 GRID parameters as scalars
                self.__dict__[ parname ] = paramarray[ paridx[irow] ]
            else : 
                # Store the FILTER and TREST grids as arrays
                self.__dict__[ parname ] = np.linspace( valmin, valmax, num=nbin, endpoint=True)

        # convert log10(redshift) to redshift
        self.REDSHIFT_FINAL = 10**self.LOGZ 

        # NOBS : Number of observations per band = size of TREST grid
        # NBANDS : Number of bands per simulated SN 
        # Nobstot : total number of observations per simulated SN
        self.NOBS = len( self.TREST )
        self.NBANDS = len(self.BANDS) 
        Nobstot = self.NOBS * self.NBANDS

        # For NON1A sims, read in the reference table of non1a models: 
        #    model ID number, SN type, model name
        if self.GENMODEL in ['NON1A', 'NONIA']:
            datCC = pyfits.getdata( gridfits, ext=('NONIA-INFO',1))
            self.non1aModelID = np.array([ dat[1] for dat in datCC ])[ilp]
            self.non1aType    = np.array([ dat[2] for dat in datCC ])[ilp]
            self.non1aModel   = np.array([ dat[3] for dat in datCC ])[ilp]

        # Read in the light curve
        iptr = int(pyfits.getdata( gridfits, ext=('PTR_I2LCMAG',1))[isn][0])
        lcdata = pyfits.getdata( gridfits, ext=('I2LCMAG',1))
        magScaled = np.array( [lcdata[i][0] for i in range(iptr+1,iptr+1+Nobstot)], dtype=float )
        magerrScaled = np.array( [lcdata[i][1] for i in range(iptr+1,iptr+1+Nobstot)], dtype=float )
        mag = magScaled / magpack
        magerr = magerrScaled / magpack
        self.MAG = np.array(mag)
        self.MAGERR = np.array(magerr)

        # Define an array containing the filter name for each observation, for each simulated SN
        self.FLT = np.ravel( [ [band]*self.NOBS for band in self.BANDS] ) 

        # Derive the flux and flux error in SNANA FLUXCAL units
        self.FLUXCAL = 10**(-0.4*(self.MAG-27.5))  
        self.FLUXCALERR = 1.0857362 * self.MAGERR * self.FLUXCAL

        # Derive the observer-frame time (relative to peak) for every observation
        self.TOBS = np.ravel( [ self.TREST * (1+self.z) ] * self.NBANDS )
        self.MJD = np.ravel( [ self.TREST * (1+self.z) ] * self.NBANDS )
                
        return( True )

        
    def getmodelerror( self, bandlist, redshift, moderrbase, errfloorbase ):
        """ Our template models are poorly constrained in the UV.
        This function expands the model uncertainty for restframe UV
        to account for this. 

        Returns : two vectors giving the model error and error floor
          for each band in bandlist
        """
        # expanding the model uncertainty at UV wavelengths by 
        # a factor  epsilon, where 
        #     epsilon = A * exp( -wave / tau )   for wave<4000 angstroms
        #             =  1   for wave>=4000 angstroms
        #  and A = 256,  tau=721
        #  (derived from the UV-optical SN Ia color dispersions in
        #    Milne et al 2013 )
        #  

        # more conservative UV error inflation
        #A = 256.     
        #tau = 721.

        # more aggressive UV error inflation
        A = 755.   
        tau = 604. 

        restwave = np.array( [ self.SURVEYDATA.FLTWAVE[ band ] / ( 1. + redshift ) for band in bandlist ])
        epsilon = np.where( restwave<4000., A * np.exp( -restwave / tau ) , 1 )
        modelerrVec  = np.where( moderrbase * epsilon<0.8, moderrbase * epsilon, 0.8)
        errfloorVec  = np.where( errfloorbase * epsilon<0.8, errfloorbase * epsilon, 0.8) 
        return( modelerrVec, errfloorVec )


    def getSALT2fit(self, getphotz=False, photversion=1, verbose=False, clobber=False ):
        """ execute the SALT2 light curve fit 
        getphotz=1 : do a 'constrained photo-z fit' with fixed cosmology
        getphotz=2 : do a 'cosmology photo-z fit' with free cosmo params
        """
        import subprocess
        import os
        import simulate

        # Set up the $SNDATA_ROOT/lcmerge directory
        sndatadir = os.environ['SNDATA_ROOT']
        lcdir = os.path.join(sndatadir,'lcmerge' )
        survey = '%s%i'%(self.SURVEY, photversion)
        # write out the .DAT file (if it doesn't exist)
        datfile = os.path.join(lcdir,"HST_%s_%s.DAT"%( survey, self.nickname.lower()))
        if not os.path.isfile( datfile ) or clobber : 
            self.writedatfile( datfile )
        elif verbose : 
            print( "%s exists. Not clobbering."%datfile )

        # append this SN to the .LIST file 
        listfile = os.path.join(lcdir,'HST_%s.LIST'%(survey))
        ignorefile = os.path.join(lcdir,'HST_%s.IGNORE'%(survey))
        if os.path.isfile( listfile ) :
            fin = open(listfile,'r')
            listlines = fin.readlines()
            fin.close()
            if os.path.basename(datfile) not in listlines : 
                if os.path.isfile( ignorefile ) : os.remove( ignorefile )
                os.rename( listfile, ignorefile )
                fout = open(listfile,'w')
                print >>fout, os.path.basename(datfile)
                fout.close()
            elif verbose : 
                print( "%s exists. Not clobbering."%listfile )
        else : 
            # construct an empty .IGNORE file 
            fout = open(ignorefile,'w')
            print >>fout,""
            fout.close()
            # construct a single-SN .LIST file 
            fout = open(listfile,'w')
            print >>fout, os.path.basename(datfile)
            fout.close()

        # construct an empty .README file  if needed
        readmefile = os.path.join(lcdir,'HST_%s.README'%(survey))
        if not os.path.isfile( readmefile ) :
            fout = open(readmefile,'w')
            print >>fout,""
            fout.close()
        elif verbose : 
            print( "%s exists. Not clobbering."%readmefile )

        # construct a single-SN .nml file 
        nmlfile = 'HST_%s.nml'%(self.nickname.lower())
        nmltext = """ 
  &SNLCINP
     VERSION_PHOTOMETRY = 'HST_%s'
     HFILE_OUT          = 'snfit_%s_%s.his'
     HFILE_KCOR         = 'HST/kcor_HST_SALT2.his'
     NFIT_ITERATION = 6
     INTERP_OPT     = 1
     USE_MWCOR = T
     LTUP_SNANA = T
     LTUP_SKY   = T
 
     H0_REF   =  %.3f
     OLAM_REF =  %.3f
     OMAT_REF =  %.3f, 0.03
     W0_REF   =  %.3f, 0.05
 
     SNTEL_LIST  = 'HST'
     CUTWIN_CID  =  0, 5
     SNCID_LIST  =  0
     SNCCID_LIST =  '%s'
     SNCCID_IGNORE = ''
     LDMP_SNFAIL =  T
  &END
    """%( survey, survey, self.nickname.lower(), 
          constants.H0, constants.OMEGA_LAMBDA, constants.OMEGA_MATTER, constants.W0_LAMBDA, 
          self.nickname.lower() )  
 
        nmltext+= """
 &FITINP
     LFIXPAR_ALL     = F
     FITMODEL_NAME  = 'SALT2.Guy10_UV2IR' 
     FITRES_DMPFILE = 'snfit_%s.fitres'
     SALT2_DICTFILE = 'snfit_%s.dictfile'

	  SALT2alpha = 0.11
	  SALT2beta  = 3.2
     PRIOR_MJDSIG        = 15.0
     PRIOR_LUMIPAR_RANGE = -5.0, 5.0
     PRIOR_LUMIPAR_SIGMA = 0.1

     TREST_REJECT  = -20.0, 60.0
     NGRID_PDF     = 0 
     LTUP_FITRES   = T
     LTUP_RESIDUAL = T 
     !FILTLIST_FIT = 'BGVRIXZWYJMHNLOPQ'  ! All possible ACS/WFC3 filters
     FILTLIST_FIT = '%s'    ! filters represented in selected SNe
     FILTLIST_DMPFUN = ''
"""%(self.nickname.lower(), self.nickname.lower(), self.FILTERS)

        # Set up the SN-photoz fitting
        if getphotz==0 : 
            nmltext+= """
     DOFIT_PHOTOZ      = F
     INISTP_DLMAG      = 0.0   ! 0=> constrain DLMAG; non-zero => float DLMAG
     PRIOR_MUERRSCALE  = 100.0 ! scale error on distance modulus prior
"""
        elif getphotz==1 : 
            nmltext+= """
     DOFIT_PHOTOZ      = T
     INISTP_DLMAG      = 0.0  ! 0=> constrain DLMAG; non-zero => float DLMAG
     PRIOR_MUERRSCALE  = 100.0 ! scale error on distance modulus prior
"""
        elif getphotz==2 : 
            nmltext+= """
     DOFIT_PHOTOZ      = T
     INISTP_DLMAG      = 0.1  ! 0=> constrain DLMAG; non-zero => float DLMAG
     PRIOR_MUERRSCALE  = 1.0 ! scale error on distance modulus prior
"""
        nmltext+= """
     OPT_PHOTOZ        = 0    ! 1=>hostgal photZ prior; 2=> specZ prior
     PRIOR_ZERRSCALE   = 1.0  ! scale error on host-photoZ prior
  &END
"""
        if not os.path.isfile( nmlfile ) or clobber: 
            fout = open( nmlfile, 'w')
            print >> fout, nmltext 
            fout.close()
        elif verbose : 
            print( "%s exists. Not clobbering."%nmlfile )

        # Run the SALT2 light curve fitter and/or read in the resulting parameters
        fitresfile = 'snfit_%s.fitres'%(self.nickname.lower())
        if not os.path.isfile( fitresfile ) or clobber: 
            hisfile = 'snfit_%s.his'%(self.nickname.lower())        
            logfile = 'snfit_%s.log'%(self.nickname.lower())        
            snlcfit = os.path.join(os.environ['SNANA_DIR'],'bin/snlc_fit.exe')+ ' %s'%nmlfile
            if verbose : print(snlcfit + ' > %s'%logfile)
            flog = open(logfile,'w')
            subprocess.call( snlcfit, shell=True, stdout=flog)
            flog.close()
        elif verbose : 
            print( "%s exists. Not clobbering."%fitresfile )
        self.salt2fit = SALT2fit( fitresfile )

        if not self.salt2fit.fitsuccess : 
            print( "SALT2 fit failed." )
            return( -1 )

        # Run a simulation to generate the best-fit SALT2 light curve model
        if getphotz : zfit = self.salt2fit.ZPHOT
        else  : zfit = self.salt2fit.Z
        pkmjd = self.salt2fit.PKMJD
        c = self.salt2fit.c
        x1 = self.salt2fit.x1
        mB = self.salt2fit.mB
        #x0 = self.salt2fit.x0
        #Av = 2.2*c   # Av ~ (Beta - 1 ) * c

        simname = 'sim_%s_SALT2fit_z%.2f'%(self.nickname.lower(), zfit )
        self.salt2fitModel = simulate.doSingleSim( 
            simname=simname, z=zfit, pkmjd=pkmjd, model='Ia', 
            trestrange=[-20,60], Av=0, mB=mB, x1=x1, c=c, 
            survey=survey, field=self.name, bands=self.bands,
            cadence=3, mjdlist=[], bandlist=[], perfect=True,
            verbose=verbose, clobber=clobber )



    def getClassSim( self, simroot='HST_classify', Nsim=2000, objectname='ClassSim',
                     simpriors=False, dustmodel='mid', clobber=False, verbose=False ):
        """ run and/or read in the results of a SNANA simulation for 
        photometric classification. 
        dustmodel : distribution of host extinction ['high','mid','low']
        simpriors : whether to embed priors within the simulation
            if False, use flat distributions (presumably priors get applied later, as
             in the case of a bayesian classification approach)
            if True, embed ALL priors within the simulation, so simulated SNe
             reflect realistic distributions in luminosity, shape, color, and redshift.
           
            NOTE: in all cases the CCSN luminosity functions are embedded in
              the simulation. i.e. the CC MAGSMEAR parameters are non-zero.
        """
        # set the redshift range of the simulation
        zmin = self.z-self.zerr
        zmax = self.z+self.zerr

        # set the range of peak mjds to simulate 
        pkmjdrange = [self.pkmjd-self.pkmjderr,self.pkmjd+self.pkmjderr]

        # check for existing simulation products
        sndatadir = os.environ['SNDATA_ROOT']
        simname = '%s_%s_dust%s'%(simroot,self.name,dustmodel)
        simdatadir = os.path.join( sndatadir,'SIM/%s'%simname )
        simisdone = np.all([ [ 
                    os.path.isfile( os.path.join( simdatadir+'_%s'%sntype, '%s_%s_%s.FITS'%(simname,sntype,sfx) ) ) 
                    for sntype in ['Ia','Ibc','II'] ] for sfx in ['PHOT','HEAD'] ] )
        if not simisdone or clobber : 
            # run the simulation 
            if verbose>2: print(" simsdone=%s  clobber=%s ..."%(simisdone,clobber))
            if verbose>1: print(" Running SNANA simulation for %s ..."%simname)
            simname = classify.doMonteCarloSim( 
                simroot=simname, survey=self.SURVEYDATA.SURVEYNAME, field=self.SURVEYDATA.FIELDNAME, 
                Nsim=Nsim, zrange=[zmin,zmax], pkmjdrange=pkmjdrange, 
                bandlist=self.FLT, mjdlist=self.MJD, 
                etimelist=np.ones(self.nobs)*10000, 
                ratemodel='flat', dustmodel=dustmodel, 
                perfect=True, simpriors=simpriors, clobber=clobber )

        elif verbose>1 : 
            print( "%s simulation exists. Not clobbering."%simname )

        # read in the simulation results if needed
        needread = True
        if objectname in self.__dict__  and not clobber : 
            simnameIa = self.__dict__[objectname].Ia.simname 
            if simnameIa == simname+'_Ia' : 
                needread = False
                if verbose>1 : print( "%s sim already exists as .%s object."%(simname,objectname) )
        if needread : 
            self.__dict__[objectname] = classify.rdClassSim(simname,verbose=verbose)
            if verbose>1: print("%s sim imported as .%s object"%(simname,objectname) )


    def getColorClassification( self, xaxis='W-J', yaxis='H', mjd='peak', 
                                classfractions='mid', dustmodel='mid',
                                Nsim=2000, modelerror=[0.0,0.0,0.0], 
                                clobber=False, verbose=True ) :
        """ 
        Using a monte carlo classification simulation, project the
        simulated SNe onto color-mag or color-color space.  Then count
        up the simulated points, with each point contributing a weight
        defined by a gaussian weighting function in that space.  The
        weighting function is centered at the observed SN position
        with widths defined by the SN observation errors.

        Results are stored in self.colorClassification as a dict of dicts, 
        keyed by MJD and 'color.mag'.  e.g.  
           self.colorClassification[55650]['W-H.H']

        xaxis : the quantity to plot along the x axis. 
        yaxis : ditto for the y axis. 
           These are strings, and either may be a color ('W-H') or a magnitude ('H')

        mjd : the observation date at which to compute the classification probability
            Use 'peak' to compute at the peak observed date, use 'all' to do every 
            observed epoch (separately)

        classfractions : either explicitly provide a set of 3 values giving the 
           fraction of SNe belonging to each class [Ia,Ibc,II],   OR
           provide a string to set the class fractions at each z based on one of 
           the 3 baseline rates assumptions. 
           If the string contains 'high' then we use the 'maximal Ia fraction'
           assumption, if it contains 'low' then we use the 'minimal Ia fraction',
           and otherwise we use the baseline assumption.  
           e.g.   classfractions='minIa'  

        dustmodel : 'high','mid','low','flat' or None for the SNANA default

        """
        # define the color-mag / mag-mag / color-color space 
        if xaxis.find('-')>0: band1, band2 = xaxis.split('-') 
        else : band1, band2 = xaxis,xaxis
        if yaxis.find('-')>0: band3, band4 = yaxis.split('-') 
        else : band3, band4 = yaxis,yaxis

        self.getClassSim( simroot='HST_colormag', simpriors=True, dustmodel=dustmodel, Nsim=Nsim, clobber=clobber, verbose=verbose ) 

        if mjd=='all' : 
            # break up observations into epochs 
            mjd = clusterfloats( self.MJD, dmax=6 )
        elif mjd=='peak' : 
            mjd = self.pkmjdobs
        if np.iterable( mjd ): 
            # recursively call this function for each mjd with clobbering off so we
            # make/read the classification simulation only the first time through
            psetlist = []
            for thismjd in mjd :                 
                # pia,pibc,pii = self.getColorClassification( xaxis=xaxis, yaxis=yaxis, mjd=thismjd, 
                pia,pibc,pii = self.getColorClassification( xaxis=xaxis, yaxis=yaxis, mjd=thismjd, 
                                                            dustmodel=dustmodel, modelerror=modelerror,
                                                            clobber=False, verbose=verbose )
                psetlist.append( [pia,pibc,pii] )
            return(np.array(psetlist))
        # ( mjd is scalar from here on out )

        # find all observations in the desired filters that are ~coincident 
        # with this observation date
        try: 
            i1 = np.where( ( np.abs(self.MJD - mjd)<5 ) & (self.FLT==band1) )[0][0]
            i2 = np.where( ( np.abs(self.MJD - mjd)<5 ) & (self.FLT==band2) )[0][0]
            i3 = np.where( ( np.abs(self.MJD - mjd)<5 ) & (self.FLT==band3) )[0][0]
            i4 = np.where( ( np.abs(self.MJD - mjd)<5 ) & (self.FLT==band4) )[0][0]
        except Exception as e:
            if verbose>1: 
                print("Missing one of the filters %s for  mjd = %.1f"%( 
                        ''.join(np.unique([band1,band2,band3,band4])), mjd)) 
            return(None)

        # Set the width of the 2-D gaussian weighting function
        if band1==band2 : snx = abs(self.MAG[i1])
        else : snx = abs(self.MAG[i1])-abs(self.MAG[i2])
        if band3==band4 : sny = abs(self.MAG[i3])
        else : sny = abs(self.MAG[i3])-abs(self.MAG[i4])
        dsnx1,dsnx2 = self.MAGERR[i1], self.MAGERR[i2]
        dsny3,dsny4 = self.MAGERR[i3], self.MAGERR[i4]
        if band1==band2 : dsnx = dsnx1
        else : dsnx = np.sign(dsnx1)*np.sign(dsnx2) * np.sqrt( dsnx1**2 + dsnx2**2 )
        if band3==band4 : dsny = dsny3
        else : dsny = np.sign(dsny3)*np.sign(dsny4) * np.sqrt( dsny3**2 + dsny4**2 )

        Nwhtlist = []
        likelist = []
        for sim in self.ClassSim :
            # sample the simulated light curves at the given MJD
            sim.samplephot( mjd, tmatch=5 )
            m1 = sim.__dict__['%s%i'%(band1, int(mjd))]
            m2 = sim.__dict__['%s%i'%(band2, int(mjd))]
            m3 = sim.__dict__['%s%i'%(band3, int(mjd))]
            m4 = sim.__dict__['%s%i'%(band4, int(mjd))]
            # limit to observations with legit data
            igood = np.where( (m1<90) & (m1>-90) & 
                              (m2<90) & (m2>-90) &
                              (m3<90) & (m3>-90) &
                              (m4<90) & (m4>-90) )[0]
            mag1,mag2 = m1[igood], m2[igood]
            mag3,mag4 = m3[igood], m4[igood]
            
            if not len(mag1) : 
                print( "ERROR: no good mags in sim %s for one of %s"%(
                        sim.simname, ''.join(np.unique([band1,band2,band3,band4]))))
                import pdb; pdb.set_trace()
                return( None ) 

            if band1==band2 : xarray = mag1
            else : xarray = mag1-mag2
            if band3==band4 : yarray = mag3
            else : yarray = mag3-mag4           

            # set the class fractions from the rates prior if needed 
            if type(classfractions) == str : 
                classfractions = classfractions.lower()
                if classfractions.find('max') >=0 or classfractions.find('high') >=0 :  
                    IaRateModel = 'high'
                elif classfractions.find('mid') >= 0 or classfractions.find('med') >= 0: 
                    IaRateModel = 'mid' 
                elif classfractions.find('min') >= 0 or classfractions.find('low') >= 0: 
                    IaRateModel = 'low'
                else : 
                    IaRateModel = 'flat'
                classfractions = getClassFractions(sim.z[igood], IaRateModel)
            # break up the class fractions by class
            if sim.simname.find('Ia')>=0: cfrac = classfractions[0]
            elif sim.simname.find('Ibc')>=0 :cfrac = classfractions[1]
            elif sim.simname.find('II')>=0 : cfrac = classfractions[2]

            # determine the modelerror factor for this class
            # and define the composite error term for each dimension
            if sim.simname.endswith('Ia') : modelerr = modelerror[0]
            elif sim.simname.endswith('Ibc') : modelerr = modelerror[1]
            elif len(modelerror)>2: modelerr = modelerror[2]
            else : modelerr = modelerror[1]
            if band1==band2 : 
                dx = np.sqrt( dsnx**2 + modelerr**2 ) 
            else : 
                dx = np.sqrt( dsnx**2 + modelerr**2 + modelerr**2 ) 
            if band3==band4 :
                dy = np.sqrt( dsny**2 + modelerr**2 ) 
            else : 
                dy = np.sqrt( dsny**2 + modelerr**2 + modelerr**2 ) 

            # compute a weighted count of all the simulated points, where the weights are defined
            # by the distance from the observed SN to the simulated SN in this color-mag space, 
            #  Scale by cfrac: the assumed fraction of all SNe belonging to this class at this z
            A = cfrac / (2*np.pi*np.abs(dx*dy) )
            gaussWeights = A * np.exp( -( (xarray - snx)**2/(2*dx**2) + (yarray - sny)**2/(2*dy**2) ) )

            # If we have an upper limit, then mask out some simulated SNe to get a one-sided gaussian
            if (dsnx < 0) and (band1 == band2) : # lower limit on a magnitude (i.e. upper limit on a flux)
                gaussWeights = np.ma.masked_where( xarray < snx, gaussWeights ) # mask any simulated SN brighter than the limit
            elif (dsnx < 0) and (band1 != band2) : # upper limit in a color component, e.g. d(W-H) < 0 
                if (dsnx1 < 0) and (dsnx2 < 0) : # dW < 0 and dH < 0  (~no constraint on W or H)
                    pass # this will already be sigma=-9.0, so no change needed
                if dsnx1 < 0 : # lower lim on the blue magnitude (e.g. dW=-9) 
                    gaussWeights = np.ma.masked_where( xarray < snx, gaussWeights ) # mask any simulated SNe bluer than the limit
                elif dsnx2 < 0 : # lower lim on the red magnitude (e.g. dH=-9) 
                    gaussWeights = np.ma.masked_where( xarray > snx, gaussWeights ) # mask any simulated SNe redder than the limit

            if (dsny < 0) and (band3 == band4) : # lower limit on a magnitude (i.e. upper limit on a flux)
                gaussWeights = np.ma.masked_where( yarray < sny, gaussWeights ) # mask any simulated SN brighter than the limit
            elif (dsny < 0) and (band3 != band4) : # upper limit in a color component, e.g. d(W-H) < 0 
                if (dsny3 < 0) and (dsny4 < 0) : # dW < 0 and dH < 0  (~no constraint on W or H)
                    pass # this will already be sigma=-9.0, so no change needed
                if dsny3 < 0 : # lower lim on the blue magnitude (e.g. dW=-9) 
                    gaussWeights = np.ma.masked_where( yarray < sny, gaussWeights ) # mask any simulated SNe bluer than the limit
                elif dsny4 < 0 : # lower lim on the red magnitude (e.g. dH=-9) 
                    gaussWeights = np.ma.masked_where( yarray > sny, gaussWeights ) # mask any simulated SNe redder than the limit

            gaussWeightedSum = gaussWeights.sum()            
            if not gaussWeightedSum : gaussWeightedSum = 0.0  # in case all values are masked

            if verbose>2: print( "  Nweighted : %.4e"%gaussWeightedSum )
            Nwhtlist.append( gaussWeightedSum )


            # Compute the integrated "pseudo-likelihood" from a flux comparison
            fluxlist= 10**(-0.4*( np.array([mag1,mag2,mag3,mag4]) - 27.5 ) )
            snfluxlist = np.array([abs(self.FLUXCAL[i1]),abs(self.FLUXCAL[i2]),abs(self.FLUXCAL[i3]),abs(self.FLUXCAL[i4])])
            snfluxerrlist = np.array([abs(self.FLUXCALERR[i1]),abs(self.FLUXCALERR[i2]),abs(self.FLUXCALERR[i3]),abs(self.FLUXCALERR[i4])])
            bandlist = [band1,band2,band3,band4]            
            iublist = [ bandlist.index( ub ) for ub in np.unique( bandlist ) ] 
            like = 0
            for iub in iublist :
                sig2 = ( snfluxerrlist[iub]**2 + (modelerr*fluxlist[iub])**2 )
                alpha = cfrac /np.sqrt(2*np.pi*sig2)
                like += (alpha * np.exp( -0.5 * ( snfluxlist[iub] - fluxlist[iub] )**2 / sig2 )).sum()
            likelist.append( like ) 

        Nia,Nibc,Nii = Nwhtlist
        pia, pibc, pii = 0,0,0
        if Nia>0 or Nibc>0 or Nii>0 : 
            if Nia>0 : pia = float(Nia)/(Nia+Nibc+Nii)
            if Nibc>0 : pibc = float(Nibc)/(Nia+Nibc+Nii)
            if Nii>0 : pii = float(Nii)/(Nia+Nibc+Nii)
        if verbose :  print( "    P(Ia) =  %.4f"%pia )
      
        #self._colorClassificationMag = np.array([pia, pibc, pii])
        self._colorClassification = np.array([pia, pibc, pii])

        # # 2013.06.03  : flux-based classifications are inaccurate
        # #  disabling for now.
        # pia, pibc, pii = 0,0,0
        # likeia, likeibc, likeii = likelist
        # if likeia>0 or likeibc>0 or likeii>0 : 
        #     if likeia>0 : pia = float(likeia)/(likeia+likeibc+likeii)
        #     if likeibc>0 : pibc = float(likeibc)/(likeia+likeibc+likeii)
        #     if likeii>0 : pii = float(likeii)/(likeia+likeibc+likeii)
        # if verbose :  print( "    P(Ia)_flux =  %.4f"%pia )
        # self._colorClassificationFlux = np.array([pia, pibc, pii])
        
        return( self._colorClassification )
        
        
    def printColorClassification( self ):
        from classify import printColorClassification
        printColorClassification( self ) 
       
    def doClassify(self, bands='all', Nsim=2000, trestrange=[-15,30], 
                   modelerror=[0.05,0.07,0.07], errfloor=0.001, 
                   useLuminosityPrior=False, plot=False,
                   x1prior=lambda x1: bifgauss( x1, 0, 1.5, 0.9), 
                   cprior=lambda c: bifgauss( c, 0, 0.08, 0.14), 
                   avprior=lambda Av: avpriorexp( Av, 0.7), 
                   zprior= lambda z : np.ones(len(z)), npkmjd = 30,
                   clobber=False, verbose=True, debug=False, 
                   kcorfile='HST/kcor_HST.fits',
                   pdzfile='', nlogz=0):
        """ redirects to doGridClassify.  See the doGridClassify doc string for help.
        See doClassifyMC for classifications using MC sims """
        self.doGridClassify( bands=bands, Nsim=Nsim, trestrange=trestrange, 
                           modelerror=modelerror, errfloor=errfloor, useLuminosityPrior=useLuminosityPrior, plot=plot,
                           x1prior=x1prior, cprior=cprior, avprior=avprior, zprior=zprior, npkmjd =npkmjd,
                           clobber=clobber, verbose=verbose, debug=debug, 
                           kcorfile=kcorfile, pdzfile=pdzfile, nlogz=nlogz)


    def doClassifyMC(self, bands='all', Nsim=2000, trestrange=[-20,50],
                             modelerror=[0.08,0.1,0.1], errfloor='auto', 
                             useLuminosityPrior=True, 
                             x1prior=lambda x1: bifgauss( x1, 0, 1.5, 0.9), 
                             cprior= extinction.midIa_c,
                             avprior= extinction.midCC,
                             zprior= lambda z : np.ones(len(z)),
                             kcorfile='HST/kcor_HST.fits', pdzfile='', 
                             clobber=False, verbose=True, debug=False):
        """ Compute classification probabilities from comparison of this observed SN light curve 
        against synthetic light curves from SNANA Monte Carlo simulations.

        OPTIONS 
        bands      : a string listing the bands to use. e.g. 'HJW'. Use 'all' for all bands.
        trestrange : use only observations within this rest-frame time window (rel. to peak)  
        modelerror : fractional flux error to apply to each SN model for chi2 calculation
                     The first value applies to the Ia model and the second to CC templates.
        Nsim       : the total number of SNe to simulate in each of the 3 SN classes
        plot       : make plots showing histograms of chi2 and posterior probabilities 
        useLuminosityPrior  : if True, compare each simulated SN to the observations as-is 
                     (i.e. including the luminosity assumptions that are baked in to the 
                     SNANA simulations) 
                     if False, allow a free parameter for scaling the flux of each simulated 
                     SN so that it most closely matches the observed fluxes (i.e. remove all 
                     "baked-in" priors on luminosity from cosmology or luminosity functions)
        errfloor   : minimum flux error for the model (e.g. for zero-flux extrapolations)
                     With the default 'auto' setting, the errfloor is automatically set on a 
                     filter-by-filter basis, using the getErrFloor function.

        NOTE: clobber decrements by 1 in CC sims to prevent re-running all sims via getClassSim each time
          so use clobber=3 to re-make the sims once, but higher than that will result in redundant calls to
          the external SNANA snlc_sim executable.

        STORED RESULTS : 
        self.ClassMC : a classify.ClassSim object (a SimTable sub-class) holding the classification results
        self.ClassMC.P[Ia,Ibc,II] : final (scalar) classification probabilities for each class
        self.ClassMC.[Ia,Ibc,II]  : the simulation and classification results from comparison to each SN sub-class
        self.ClassMC.[Ia,Ibc,II].CHI2 : len==Nsim vectors of chi2 values from comparison to each simulated SN, by class
        self.ClassMC.[Ia,Ibc,II].LIKE : len==Nsim vectors of likelihood values (i.e. exp(-chi2/2) )
        self.ClassMC.[Ia,Ibc,II].PRIOR[z,c,x1,Av,Rv] : the prior probability functions 
        self.ClassMC.[Ia,Ibc,II].PROB : len==Nsim vectors of bayesian posterior probability values
        """
        if debug : import pdb; pdb.set_trace()

        # check for existing probabilities
        if( not clobber and  'ClassMC' in self.__dict__ ) : 
            if verbose : print( "Monte Carlo Classification already exists. Not clobbering.")
            return(None)

        # set the modelerror (for backwards compatibility)
        if modelerror in [None,0] : modelerror = [0,0]
        elif not np.iterable( modelerror ) : modelerror = [ modelerror, modelerror, modelerror ]
        elif len(modelerror)==2 : modelerror = [modelerror[0], modelerror[1], modelerror[2] ]

        # compute chi2 and likelihood vectors (if needed) 
        self.getChi2LikelihoodMC( 'Ia', Nsim=Nsim, bands=bands,trestrange=trestrange,
                                  modelerror=modelerror[0], errfloor=errfloor, 
                                  useLuminosityPrior=useLuminosityPrior, verbose=verbose, clobber=clobber )
        self.getChi2LikelihoodMC( 'Ibc', Nsim=Nsim, bands=bands,trestrange=trestrange,
                                  modelerror=modelerror[1], errfloor=errfloor, 
                                  useLuminosityPrior=useLuminosityPrior, verbose=verbose, clobber=(clobber>1) )
        self.getChi2LikelihoodMC( 'II', Nsim=Nsim, bands=bands,trestrange=trestrange,
                                  modelerror=modelerror[2], errfloor=errfloor, 
                                  useLuminosityPrior=useLuminosityPrior, verbose=verbose, clobber=(clobber>1))
        
        # -----    COMPUTE THE POSTERIOR PROBABILITY ARRAYS ------ 
        # NOTE on the absence of a factor for the parameter sampling interval: 
        # To define 'proper' priors (i.e. require that they integrate to unity)
        # we could define the sampling interval for each parameter e.g: 
        #   x1 = self.ClassMC.Ia.SIM_SALT2x1                 # simulated x1 values
        #   dx1 = (x1.max() - x1.min())/len(x1)    # mean x1 step size
        #   px1 = x1prior( self.ClassMC.Ia.SIM_SALT2x1 )     # values of the (unnormalized) prior dist'n
        #   px1proper = px1 / (px1.sum()*dx1)    # normalized (proper) prior
        # Then when we integrate the posterior probabilities to get the final classification
        # probability, we would have:
        #   self.postProbIa = self.likeIa * px1proper * dx1
        # which is equivalent to 
        #   self.postProbIa = self.likeIa * px1 / px1.sum()

        # ----    TYPE IA POSTERIOR PROBABILITY -------
        # define the priors
        px1 = x1prior( self.ClassMC.Ia.SIM_SALT2x1 ) 
        pc  = cprior( self.ClassMC.Ia.SIM_SALT2c )
        if self.zerr>0.01: pz  = zprior( self.ClassMC.Ia.SIM_REDSHIFT )
        else : pz = 1 
        self.ClassMC.Ia.PRIORx1 = x1prior
        self.ClassMC.Ia.PRIORc = cprior
        self.ClassMC.Ia.PRIORz = zprior
        # convert the likelihood dist'n into posterior probability dist'n
        self.ClassMC.Ia.PROB = px1 * pc * pz * self.ClassMC.Ia.LIKE

        # ----    TYPE IB/C POSTERIOR PROBABILITY -------
        # Define the priors
        if self.zerr>0.01: pz  = zprior( self.ClassMC.Ibc.SIM_REDSHIFT )
        else : pz = 1 
        pAv = avprior( self.ClassMC.Ibc.DUMP['AV'] )
        self.ClassMC.Ibc.PRIORz = zprior
        self.ClassMC.Ibc.PRIORAv = avprior
        # convert the likelihood dist'n into posterior probability dist'n
        self.ClassMC.Ibc.PROB = pz * pAv * self.ClassMC.Ibc.LIKE

        # ----    TYPE II POSTERIOR PROBABILITY -------
        # Define the priors
        if self.zerr>0.01: pz  = zprior( self.ClassMC.II.SIM_REDSHIFT )
        else : pz = 1 
        pAv  = avprior( self.ClassMC.II.DUMP['AV'] )
        self.ClassMC.II.PRIORz = zprior
        self.ClassMC.II.PRIORAv = avprior
        # convert the likelihood dist'n into posterior probability dist'n
        self.ClassMC.II.PROB = pz * pAv * self.ClassMC.II.LIKE

        # Finally, marginalize over nuisance parameters and normalize
        # to get the (scalar) classification probabilities: 
        #   the probability that this object belongs to each SN class
        pIa, pIbc, pII = self.ClassMC.Ia.PROB.sum(), self.ClassMC.Ibc.PROB.sum(), self.ClassMC.II.PROB.sum()
        self.ClassMC.PIa  = pIa / ( pIa + pIbc + pII )
        self.ClassMC.PIbc = pIbc / ( pIa + pIbc + pII )
        self.ClassMC.PII  = pII / ( pIa + pIbc + pII )

        if verbose>1 : 
            print("P(Ia) = %.3f\nP(Ib/c) = %.3f\nP(II) = %.3f\n"%(self.ClassMC.PIa,self.ClassMC.PIbc,self.ClassMC.PII) )

    def doGridClassify(self, bands='all', Nsim=0, trestrange=[-20,50], 
                       modelerror=[0.05,0.07,0.07], errfloor=0.001, inflateUVerr=True,
                       useLuminosityPrior=True, magnification=1,
                       x1prior=lambda x1: bifgauss( x1, 0, 1.5, 0.9), cprior= extinction.midIa_c, 
                       avprior= extinction.midCC, zprior='host', #lambda z : np.ones(len(z)), 
                       classfractions='mid', clobber=False, verbose=True, debug=False,
                       kcorfile='HST/kcor_HST.fits', pdzfile='', 
                       nlogz=0, ncolorpar=0, ncolorlaw=0, nlumipar=0, npkmjd = 0,
                       omitTemplateIbc='', omitTemplateII='', getSystematicError=False, 
                       getSNphotz=True ):
        """ Bayesian photometric SN classification using SNANA grid simulations.
        
        The user must set the grid size. You can set Nsim for automatic definition of 
        the grid shape, or you can explicitly provide all of the grid dimensions using
        the five parameters: [ nlogz, ncolorpar, ncolorlaw, nlumipar, npkmjd ]. 

        OPTIONS 
        bands      : a string listing the bands to use. e.g. 'HJW'. Use 'all' for all bands.
        trestrange : use only observations within this rest-frame time window (rel. to peak)  
        modelerror : fractional flux error to apply to each SN model for chi2 calculation
                     The first value applies to the Ia model and the second to CC templates.
        useLuminosityPrior  : 
                  if False: allow a free parameter for scaling the flux of each simulated 
                       SN so that it most closely matches the observed fluxes
                  if == True or == 1 : allow the free parameter for flux scaling, but 
                       also apply a prior based on the value of that optimal flux scaling
                       factor, relative to the assumed luminosity function for each class
                  if >1,  extend the CCSN simulation grids along a new dimension
                       that samples the assumed luminosity functions (and apply priors)
        pdzfile    : a redshift probability distribution from a photo-z fit used to set the 
                     redshift prior.  First column should be redshift, the second P(z).

        classfractions : prior probabilities for the 3 SN classes. This may be:
                     1. a list giving   [p(Ia), p(Ibc), p(II)] 
                     2. 'high', 'mid', or 'low' for a redshift-dependent prior based 
                        on observed SN rates and SN class fractions (gives a  high, 
                        medium, or low relative rate of SNIa at all z)
                     3. 'galsnid'  to use the user-defined HOST_TYPE to set the prior,
                         following the Foley+Mandel galsnid approach, without allowing 
                         any redshift dependence
                     4. 'galsnid-high','galsnid-mid','galsnid-low' to combine options 2 
                        and 3 by using the redshift-dependent class fractions as priors 
                        for galsnid, to get a crudely redshift-dependent galsnid posterior

        omitTemplateIbc/omitTemplateII: Remove a particular template from the classifications.
                                        This is intended for SNPhotCC trials.

        clobber :  (0) don't do anything unless there are no existing probability results 
                   (1) remove existing posterior probabilities, recompute them (possibly with new priors)
                   (2) remove existing posteriors, read in fresh data from existing simulations
                   (3+) (re-)run the simulations, read them in, (re-)compute probabilities

        STORED RESULTS : 
        self.chi2[Ia,Ibc,II]  : vectors of chi2 values from comparison to each simulated SN, by class
        self.postProb[Ia,Ibc,II]  : vectors of posterior probability values from comparison to each sim SN
        self.P[Ia,Ibc,II]    : final (scalar) classification probabilities for each class

        self.maxlike[Ia,Ibc,II] : best-fitting (ie. min chi2 = max likelihood) model for each class
        self.maxprob[Ia,Ibc,II] : best-match model, including the priors (i.e. maximizing the Bayes numerator)
        """
        from constants import CCMODELS
        from copy import deepcopy
        import galsnid
        if debug : import pdb; pdb.set_trace()

        if pdzfile: 
            # if a P(z) file is provided, use it to set the redshift prior
            zprior = zpriorFromFile( pdzfile )
        elif zprior=='host' : 
            # set the redshift prior using the SN's best-estimate redshift and uncertainty
            zprior = zpriorFromHost( self.z, dz=self.zerr, z68=self.z68, z95=self.z95 )

        # check for existing probabilities
        if( not clobber and  'PIa' in self.__dict__ and 
            'PIbc' in self.__dict__ and 'PIbc' in self.__dict__ ) : 
            if verbose : print( "p(Ia,Ibc,II) already exists. Not clobbering.")
            return(None)

        if modelerror in [None,0] : modelerror = [0,0,0]
        elif not np.iterable( modelerror ) : modelerror = [ modelerror, modelerror, modelerror ]
        elif len(modelerror)==2 : modelerror = [ modelerror[0], modelerror[1], modelerror[1] ]
        if len(modelerror)!=3 : 
            raise exceptions.RuntimeError("modelerror must be a list of 3 values giving errIa,errIbc,errII")

        if not Nsim and not np.all( [nlogz, ncolorpar, ncolorlaw, nlumipar] ) : 
            raise exceptions.RuntimeError(
                "You must provide either Nsim or all of the grid parameters: [nlogz, ncolorpar, ncolorlaw, nlumipar, npkmjd]")
        if not npkmjd : npkmjd = int( (2*self.pkmjderr + 1.)/(1+self.z) )

        # compute chi2 and likelihood vectors (if needed) 
        # NOTE: clobber decrement by 1 in CC sims to prevent re-running all sims via getClassSim each time
        #  so use clobber=3 to re-make the sims once, but higher than that will be redundant.
        if verbose>3: print("Computing chi2 values across the Ia grid...")
        self.getChi2LikelihoodGrid( 'Ia', Nsim=Nsim, bands=bands,trestrange=trestrange,
                                    modelerror=modelerror[0], errfloor=errfloor, inflateUVerr=inflateUVerr,
                                    useLuminosityPrior=useLuminosityPrior, magnification=magnification,
                                    verbose=verbose, clobber=max(0,clobber-1), kcorfile=kcorfile, 
                                    npkmjd = npkmjd, nlogz=nlogz, ncolorpar=ncolorpar, 
                                    ncolorlaw=ncolorlaw, nlumipar=nlumipar, 
                                    omitTemplateIbc = omitTemplateIbc,
                                    omitTemplateII = omitTemplateII  )
        if verbose>3: print("Computing chi2 values across the Ib/c grid...")
        self.getChi2LikelihoodGrid( 'Ibc', Nsim=Nsim, bands=bands,trestrange=trestrange,
                                    modelerror=modelerror[1], errfloor=errfloor, inflateUVerr=inflateUVerr,
                                    useLuminosityPrior=useLuminosityPrior, magnification=magnification,
                                    verbose=verbose, clobber=max(0,clobber-1), kcorfile=kcorfile, 
                                    npkmjd = npkmjd, nlogz=nlogz, ncolorpar=ncolorpar, 
                                    ncolorlaw=ncolorlaw, nlumipar=nlumipar,
                                    omitTemplateIbc = omitTemplateIbc,
                                    omitTemplateII = omitTemplateII)
        if verbose>3: print("Computing chi2 values across the II grid...")
        self.getChi2LikelihoodGrid( 'II', Nsim=Nsim, bands=bands,trestrange=trestrange,
                                    modelerror=modelerror[2], errfloor=errfloor, inflateUVerr=inflateUVerr,
                                    useLuminosityPrior=useLuminosityPrior, magnification=magnification,
                                    verbose=verbose, clobber=max(0,clobber-1), kcorfile=kcorfile, 
                                    npkmjd = npkmjd, nlogz=nlogz, ncolorpar=ncolorpar, 
                                    ncolorlaw=ncolorlaw, nlumipar=nlumipar,
                                    omitTemplateIbc = omitTemplateIbc,
                                    omitTemplateII = omitTemplateII )

        simIa = self.ClassSim.Ia
        simIbc = self.ClassSim.Ibc
        simII = self.ClassSim.II

        self.ClassSim.Ia.modelerror = modelerror[0]
        self.ClassSim.Ibc.modelerror = modelerror[1]
        self.ClassSim.II.modelerror = modelerror[2]
        self.ClassSim.Ia.errfloor = errfloor
        self.ClassSim.Ibc.errfloor = errfloor
        self.ClassSim.II.errfloor = errfloor
        self.classModelError = modelerror
        self.classErrFloor = errfloor

        # If needed, set the assumption about how the fraction of SNe
        # that are Type Ia evolves with redshift.  (To be used later
        # for modifying the redshift priors for each class)
        IaClassFraczModel = None
        if type(classfractions) == str : 
            classfractions = classfractions.lower()
            if classfractions.find('high') >=0 :  
                IaClassFraczModel = 'high'
            elif classfractions.find('mid') >= 0: 
                IaClassFraczModel = 'mid' 
            elif classfractions.find('low') >= 0: 
                IaClassFraczModel = 'low'
            elif classfractions.find('galsnid') >= 0: 
                # No redshift dependence for the class priors. The priors
                # will reflect only the host galaxy properties via galsnid
                pIa, pIbc, pII = galsnid.classify( self.HOST_MORPHOLOGY, self.HOST_SEDTYPE, 1, 1, 1 )
            else : 
                raise exceptions.RuntimeError("classfractions=%s is not understood."%classfractions)
        elif type(classfractions) == list :
            # No redshift dependence for the class priors. The priors
            # have been  explicitly provided by the user.
            pIa, pIbc, pII = classfractions
        else : 
            raise exceptions.RuntimeError("classfractions must be a list or a string.")

        # Define the MJDpk parameter vector, common to all SN classes 
        pkmjdmin =  self.pkmjd - self.pkmjderr
        pkmjdstep = (2.*self.pkmjderr+1) / npkmjd
        pkmjdvalues = np.array([ pkmjdmin + ipkmjd*pkmjdstep for ipkmjd in xrange(npkmjd) ])

        # FILLING THE POSTERIOR PROBABILITY DISTIRIBUTIONS 

        # ----    TYPE IA POSTERIOR PROBABILITY -------
        # fill the prior probability matrices
        z,c,x1,mjdpk = [],[],[],[]
        for Iamjdpk in pkmjdvalues:
            for Iax1 in simIa.x1:
                for Iac in simIa.c:
                    for Iaz in simIa.z:
                        z += [Iaz]
                        c += [Iac]
                        x1 += [Iax1]
                        mjdpk += [Iamjdpk]
        z,c,x1,mjdpk = np.array(z),np.array(c),np.array(x1),np.array(mjdpk)
        px1 = x1prior( x1 )
        pc  = cprior( c )
        pz  = zprior( z )
        pmjdpk = np.ones( len(mjdpk) )
        if pz.sum()==0 : 
            raise exceptions.RuntimeError("ERROR : redshift prior is zero everywhere! Check your photoz input file %s"%pdzfile)

        if IaClassFraczModel in ['high','mid','low'] : 
            # set the class prior to reflect redshift dependence of
            # the SN class fractions 
            pIaz, pIbcz, pIIz = getClassFractions( z, IaClassFraczModel )
            if 'galsnid' in classfractions : 
                # update the redshift-dependent Ia class prior to reflect host galaxy properties
                pIaz, junk1, junk2 = galsnid.classify( self.HOST_MORPHOLOGY, self.HOST_SEDTYPE, pIaz, pIbcz, pIIz )
            # pull out the first value to serve as a constant factor, 
            # and use the normalized class prior to reshape the redshift prior
            pIa = pIaz[0] 
            pz = pz * pIaz/pIa

        # define the values of each 1d Prior *VECTOR*, as realized across the
        # one-dimensional sampling range of each parameter
        logz1d, z1d = simIa.LOGZ, simIa.z
        x11d, c1d = simIa.x1, simIa.c
        mjdpk1d  = pkmjdvalues
        simIa.zPriorVec = zprior( z1d ) 
        simIa.cPriorVec = cprior( c1d )
        simIa.x1PriorVec = x1prior( x11d )
        simIa.mjdpkPriorVec = np.ones( len(mjdpk1d) )

        # Define the step sizes for each parameter dimension
        # Note that the redshift grid is in log10(z) space.
        dlogz,dx1,dc,dmjdpk = 1,1,1,1
        if(len(z1d)>1) : dlogz = np.mean( np.diff(logz1d) ) or 1
        if(len(x11d)>1) : dx1 = np.mean( np.diff(x11d) ) or 1
        if(len(c1d)>1) : dc = np.mean( np.diff(c1d) ) or 1
        if(len(mjdpk1d)>1) : dmjdpk = np.mean( np.diff(mjdpk1d) ) or 1

        # Define the normalization factor KIa that makes the 
        # composite prior proper, i.e. the factor that 
        # causes p(theta) to integrate to unity over the 
        # entire N-dimensional parameter space. 
        dtheta = dc * dx1 * dlogz * dmjdpk
        KIa = 1. / ( pc * px1 * pz * pmjdpk * dtheta ).sum()
        
        if False and np.iterable( pIa ) :
            # The class prior may vary with redshift, i.e. p(Ia) 
            # is p(Ia|z), so it gets a separate normalization, using
            # only the non-redshift dimensions of parameter space
            Notherdim = len(x11d)*len(c1d)*len(mjdpk1d)
            dotherdim = dc * dx1 * dmjdpk
            pIa = pIa / ( Notherdim * dotherdim )

        # Store the normalization factor KIa and the (improper)
        # priors, each with a shape that matches the size of the full
        # multi-dimensional grid
        simIa.priorNormFactor = KIa
        simIa.x1PriorMatrix = px1
        simIa.cPriorMatrix = pc
        simIa.zPriorMatrix = pz
        simIa.mjdpkPriorMatrix = pmjdpk
        simIa.classPriorMatrix = pIa

        # store the step-sizes for each parameter dimension
        simIa.dc     = dc
        simIa.dx1    = dx1
        simIa.dlogz  = dlogz  
        simIa.dmjdpk = dmjdpk 
        simIa.dtheta = dtheta

        # Note that there is no pMagoff component here:
        # For Type Ia, we always have no prior penalty associated
        # with a magnitude offset. Either b/c we don't apply a
        # mag offset (useLumPrior>=1) or b/c we are doing a 
        # luminosity-independent classification (useLumPrior=False)

        # Multiply priors times the likelihood dist'n to get unnormalized 
        # posterior probability distribution functions (the Bayes theorem numerator)
        # [these will be normalized below after doing the same for Ib/c and II classes]
        self.postProbIa =  KIa * pIa * px1 * pc * pz * pmjdpk * self.likeIa * dtheta

        # ----    TYPE Ib/c and TYPE II POSTERIOR PROBABILITIES -------
        # Define the priors
        # Note : We define pIbc as the prior probability that any given observed SN 
        #   is of type Ibc (at each redshift).  This prior probability is distributed
        #   evenly over all of the Type Ib/Ic templates in the SNANA template library
        #   (via the prior on 'tmp' the 'template parameter' dimension).
        #   A better solution would be to distribute it according to the sub-class 
        #   population sizes, as measured by Li:2011, Smartt:2009, etc. 
        #   That would require computing the posterior probability separately for 
        #   each template, and applying the appropriate sub-class prior to each.
        #   We do not add that complexity here. 

        for simCC,CCtype in zip( [simIbc,simII], ['Ibc','II']) : 
            # fill the prior probability grids
            z,Av,tmp,mjdpk = [],[],[],[]
            if useLuminosityPrior>1 : 
               nmagoff, pMagoff, dMagoff = nlumipar,[],[]
            elif useLuminosityPrior==1 : 
                nmagoff, pMagoff,dMagoff = 1, simCC.FLUXSCALEPRIOR, 1
            else : 
                nmagoff, pMagoff, dMagoff = 1, 1, 1
            for iMagoff in range(nmagoff):
                for CCpkmjd in pkmjdvalues:
                    for CCTmp in simCC.non1aTypes:
                        if useLuminosityPrior>1 : 
                            pMag,dMag = magOffsetPrior( CCTmp, iMagoff, nmagoff )
                        for CCAv in simCC.AV:
                            for CCz in simCC.z:
                                z += [CCz]
                                Av += [CCAv]
                                tmp += [CCTmp]
                                mjdpk += [CCpkmjd]
                                if useLuminosityPrior>1 : 
                                    pMagoff += [pMag]
                                    dMagoff += [dMag]
            z,Av,tmp,mjdpk = np.array(z),np.array(Av),np.array(tmp),np.array(mjdpk)
            pz = zprior( z )
            pAv = avprior( Av )
            ptmp = np.ones( len( tmp ) ) 
            pmjdpk = np.ones( len(mjdpk) )
            if useLuminosityPrior>1: pMagoff = np.array(pMagoff)

            if IaClassFraczModel in ['high','mid','low'] : 
                # set the class prior to reflect redshift dependence of
                # the SN class fractions 
                pIaz, pIbcz, pIIz = getClassFractions( z, IaClassFraczModel )
                if 'galsnid' in classfractions : 
                    # update the redshift-dependent Ia class prior to reflect host galaxy properties
                    junk1, pIbcz, pIIz = galsnid.classify( self.HOST_MORPHOLOGY, self.HOST_SEDTYPE, pIaz, pIbcz, pIIz )
                if CCtype=='Ibc': pCCz = pIbcz
                else : pCCz = pIIz
                # pull out the first value to serve as a constant factor, 
                # and use the normalized class prior to reshape the redshift prior
                pCC = pCCz[0] 
                pz = pz * pCCz/pCC
            elif CCtype=='Ibc': pCC = pIbc
            elif CCtype=='II' : pCC = pII


            # store the values of each Prior *VECTOR*, as realized across the
            # one-dimensional sampling range of each parameter
            logz1d, z1d  = simCC.LOGZ, simCC.z
            tmp1d, Av1d = np.arange( len(simCC.non1aTypes)), simCC.AV
            mjdpk1d = pkmjdvalues
            simCC.zPriorVec = zprior( z1d ) 
            simCC.AvPriorVec = avprior( Av1d )
            simCC.tmpPriorVec = np.ones( len(tmp1d) )
            simCC.mjdpkPriorVec = np.ones( len(mjdpk1d) )
            if useLuminosityPrior>1:  
                # There is no single 1-d vector that can encapsulate the mag offset prior, 
                # because it is separately defined for each sub-class, so we define a dict
                # keyed by the sub-class names.
                simCC.magoffPriorDict = dict( [ [ CCTmp,[ magOffsetPrior(CCTmp,iMagoff,nmagoff) 
                                                        for iMagoff in range(nmagoff) ] ] 
                                              for CCTmp in simCC.non1aTypes ] )
            elif useLuminosityPrior==1:  
                simCC.magoffPriorDict = dict( [ [ CCTmp,[magOffsetPrior(CCTmp,iMagoff,nlumipar) 
                                                         for iMagoff in range(nlumipar) ] ]
                                               for CCTmp in simCC.non1aTypes ] )

            # Define the step sizes for each parameter dimension
            # Note that the redshift grid is in log10(z) space.
            dlogz,dtmp,dAv,dmjdpk = 1,1,1,1
            if(len(z1d)>1) : dlogz = np.mean( np.diff(logz1d) )
            if(len(tmp1d)>1) : dtmp = np.mean( np.diff(tmp1d) )
            if(len(Av1d)>1) : dAv = np.mean( np.diff(Av1d) )
            if(len(mjdpk1d)>1) : dmjdpk = np.mean( np.diff(mjdpk1d) )

            # Note that the magnitude offset step size can have a
            # different value at each point in the N-d parameter space
            # (unlike the other parameters that have fixed step size),
            # because we have different luminosity functions for each
            # CC sub-class. 
            if useLuminosityPrior>1: dMagoff = np.array(dMagoff)

            # Define the normalization factor KCC that makes the 
            # composite prior proper, i.e. the factor that 
            # causes p(theta) to integrate to unity over the 
            # entire N-dimensional parameter space. 
            dtheta = dAv * dtmp * dlogz * dmjdpk * dMagoff
            KCC = 1./( pAv * ptmp * pz * pmjdpk * pMagoff * dtheta ).sum()

            if useLuminosityPrior==1 :
                # In this case the magnitude offset (aka flux scaling factor) 
                # is a "pseudo-parameter dimension". We have picked out the 
                # optimal flux scaling factor and have applied a
                # normalized 1-D gaussian prior. Here we effectively 
                # renormalize that prior so that it now integrates to unity
                # over the entire N-dimensional volume of parameter space.
                pMagoff =  pMagoff / ( dtheta * len(pMagoff) )

            # The class prior may vary with redshift, i.e. p(II) 
            # is p(II|z), so it gets a separate normalization, using
            # only the non-redshift dimensions of parameter space
            if False and np.iterable( pCC ) :
                Notherdim = len(Av1d) * len(tmp1d) * len(mjdpk1d) * nmagoff
                dotherdim = dAv * dtmp * dmjdpk * dMagoff
                pCC = pCC / ( Notherdim * dotherdim )

            # Now store the values of each normalized prior MATRIX, 
            # as applied across the entire multi-dimensional grid
            simCC.priorNormFactor = KCC
            simCC.tmpPriorMatrix = ptmp
            simCC.AvPriorMatrix = pAv
            simCC.zPriorMatrix = pz
            simCC.mjdpkPriorMatrix = pmjdpk
            simCC.magoffPriorMatrix = pMagoff
            simCC.classPriorMatrix = pCC

            simCC.dAv     = dAv    
            simCC.dtmp    = dtmp   
            simCC.dlogz   = dlogz  
            simCC.dmjdpk  = dmjdpk 
            simCC.dMagoff = dMagoff
            simCC.dtheta  = dtheta

            # multiply prior matrices times the likelihood dist'n to get unnormalized 
            # posterior probability distribution functions (the Bayes theorem numerator)
            # [these will be normalized below after doing the same for Ib/c and II classes]
            self.__dict__['postProb'+CCtype] = KCC * pCC * pMagoff * pAv * pz * pmjdpk * self.__dict__['like'+CCtype] * dtheta

        # END FOR LOOP OVER Ib/c AND II 
        # ------------------------------------------------------------
   
        # Finally, divide by the total integrated probability across
        # all SN classes to get the final posterior probability
        # distributions, and then marginalize over nuisance parameters
        # to get the (scalar) classification probabilities: the
        # probability that this object belongs to each SN class
        PIa, PIbc, PII = self.postProbIa.sum(), self.postProbIbc.sum(), self.postProbII.sum()

        self.bayesNormFactor = ( PIa + PIbc + PII )
        self.postProbIa  /= self.bayesNormFactor
        self.postProbIbc /= self.bayesNormFactor
        self.postProbII  /= self.bayesNormFactor
        self.PIa  = PIa  /  self.bayesNormFactor
        self.PIbc = PIbc /  self.bayesNormFactor
        self.PII  = PII  /  self.bayesNormFactor

        simIa.postProb = self.postProbIa
        simIbc.postProb = self.postProbIbc
        simII.postProb = self.postProbII

        # Redshift estimation and uncertainties.
        # Here we marginalize the posterior probability array over
        # all non-redshift parameters to get a likelihood
        # distribution in redshift space. Then locate the peak and
        # measure the 1-sigma width of the distribution.
        for sim in [simIa,simIbc,simII] : 
            if len(sim.LOGZ)<=3 : 
                sim.zsnphot = self.z
                sim.zsnphoterrPlus = self.zerr
                sim.zsnphoterrMinus = self.zerr
                sim.zsnphoterr = self.zerr
                continue
            if sim.NMAGOFFSET>1 : 
                postProb = sim.postProb.reshape( sim.NMAGOFFSET, len(sim.PKMJD),  len(sim.LUMIPAR), 
                                                 len(sim.COLORLAW), len(sim.COLORPAR), len(sim.LOGZ))
            else : 
                postProb = sim.postProb.reshape( len(sim.PKMJD),  len(sim.LUMIPAR), 
                                                 len(sim.COLORLAW), len(sim.COLORPAR), len(sim.LOGZ))
            Naxes=len(postProb.shape)
            iaxeslist = range(Naxes-1)
            postProbMarginalized = deepcopy(postProb)
            for igax in reversed(iaxeslist) : 
                postProbMarginalized = postProbMarginalized.sum( axis=igax )
            postProbMarginalized /= sum( postProbMarginalized )
            ipostProbSorted = np.argsort( postProbMarginalized )
            ppsum, zpeaklist = 0,[]
            zerrmin = np.abs(sim.z[ipostProbSorted][-1]-sim.z[ipostProbSorted][-2])
            for ipps in reversed(ipostProbSorted) : 
                ppsum += postProbMarginalized[ipps]
                zpeaklist.append( sim.z[ipps] )
                if ppsum >= 0.68 : 
                    sim.zsnphot = zpeaklist[0]
                    sim.zsnphoterrPlus = np.max(zerrmin,np.max(zpeaklist)-zpeaklist[0])
                    sim.zsnphoterrMinus = np.max(zerrmin,zpeaklist[0] - np.min(zpeaklist))
                    sim.zsnphoterr = np.sqrt( sim.zsnphoterrMinus**2 + sim.zsnphoterrPlus**2 )
                    break
        if self.PIa>0.5 : 
            self.zsnphot = simIa.zsnphot
            self.zsnphoterrPlus = simIa.zsnphoterrPlus
            self.zsnphoterrMinus = simIa.zsnphoterrMinus
            self.zsnphoterr = simIa.zsnphoterr
        elif self.PIbc>=self.PII : 
            self.zsnphot = simIbc.zsnphot
            self.zsnphoterrPlus = simIbc.zsnphoterrPlus
            self.zsnphoterrMinus = simIbc.zsnphoterrMinus
            self.zsnphoterr = simIa.zsnphoterr
        else : 
            self.zsnphot = simII.zsnphot
            self.zsnphoterrPlus = simII.zsnphoterrPlus
            self.zsnphoterrMinus = simII.zsnphoterrMinus
            self.zsnphoterr = simIa.zsnphoterr

        # construct a SuperNova object to hold the model with the 
        # highest posterior probability for each type
        for classname,classSimTab,moderr in zip(['Ia','Ibc','II'],[simIa,simIbc,simII],modelerror) : 
            imaxprob = classSimTab.postProb.argmax()

            iz, icp, icl, ilp, ipm, ifs = classSimTab.index2gridCoord( imaxprob )
            imaxprobSimSN = classSimTab.gridCoord2snanaIndex( [iz,icp,icl,ilp,ipm,ifs] )

            maxProbModel = SuperNova( simname=classSimTab.simname, snid=imaxprobSimSN ) 
            maxProbModel.SURVEY = self.SURVEY
            maxProbModel.SURVEYDATA = self.SURVEYDATA
            maxProbModel.NAME = 'maxProb%sModel'%classname
            maxProbModel.SNID = imaxprobSimSN
            maxProbModel.INDEX = imaxprob
            maxProbModel.SNANAINDEX = imaxprobSimSN
            #maxProbModel.SIMNAME = classSimTab.simname
            #maxProbModel.FLT = classSimTab.FLT[0]
            if classname=='Ia' : 
                maxProbModel.FLUXSCALE = 1
                maxProbModel.MAGOFF =  0
            else : 
                maxProbModel.FLUXSCALE = classSimTab.FLUXSCALE[imaxprob]
                if maxProbModel.FLUXSCALE > 0 : 
                    maxProbModel.MAGOFF = ( -2.5*np.log10( maxProbModel.FLUXSCALE ) )
                else : 
                    maxProbModel.MAGOFF = np.nan
            maxProbModel.FLUXCAL = maxProbModel.FLUXCAL * maxProbModel.FLUXSCALE
            maxProbModel.FLUXCALERR = maxProbModel.FLUXCAL * moderr + errfloor
            maxProbModel.MAG = maxProbModel.MAG + maxProbModel.MAGOFF
            maxProbModel.MAGERR = np.abs(1.0857*maxProbModel.FLUXCALERR/maxProbModel.FLUXCAL) 
            maxProbModel.CHI2 = self.__dict__['chi2%s'%classname][imaxprob]
            maxProbModel.NDOF = classSimTab.NDOF
            maxProbModel.REDSHIFT = classSimTab.z[iz]
            maxProbModel.LOGZ = classSimTab.LOGZ[iz]
            maxProbModel.COLORPAR = classSimTab.COLORPAR[icp]
            maxProbModel.COLORLAW = classSimTab.COLORLAW[icl]
            maxProbModel.LUMIPAR = classSimTab.LUMIPAR[ilp]
            maxProbModel.PEAKMJD = classSimTab.PKMJD[ipm]
            maxProbModel.PEAKMJDERR = np.mean( np.diff(classSimTab.PKMJD) )/2.
            maxProbModel.MJD = maxProbModel.MJD + maxProbModel.PEAKMJD

            if classname == 'Ia' : 
                maxProbModel.TEMPLATE = classSimTab.GENMODEL
                maxProbModel.TYPE = 'Ia'
            else : 
                maxProbModel.TYPE =  CCMODELS['%03i'%maxProbModel.LUMIPAR][0]
                maxProbModel.TEMPLATE = CCMODELS['%03i'%maxProbModel.LUMIPAR][1]

            self.__dict__['maxProb%sModel'%classname] = maxProbModel


        self.PIaString = '%.2f'%( self.PIa )
        if getSystematicError : 
            # repeat the classification with alternate dust and rates prior assumptions
            # to get the systematic classification uncertainties
            snCopy = deepcopy( self )
            
            if 'mid' in classfractions : 
                cfHigh=classfractions.replace('mid','high')
                cfLow=classfractions.replace('mid','low')
            elif 'galsnid' in classfractions : 
                # this is 'galsnid' alone, i.e. a redshift independent prior
                cfHigh=classfractions
                cfLow=classfractions
            elif classfractions[0] == classfractions[1] : 
                # e.g. classfractions=[1,1,1]
                cfHigh=classfractions
                cfLow=classfractions
            else : 
                # e.g. classfractions=[0.25,0.15,0.60]
                cfHigh= [ classfractions[0]*2, classfractions[1], classfractions[2] ]
                cfHigh = [ round( float(cf)/np.sum(cfHigh),2)  for cf in cfHigh ]
                cfLow= [ classfractions[0]/2., classfractions[1], classfractions[2] ]
                cfLow = [ round( float(cf)/np.sum(cfLow),2)  for cf in cfLow ]

            pialist = [ self.PIa ]
            # Upper Limit on P(Ia) : high dust and high Ia fraction priors
            snCopy.doGridClassify( bands=bands, Nsim=Nsim, trestrange=trestrange, modelerror=modelerror, 
                                   errfloor=errfloor, useLuminosityPrior=useLuminosityPrior, x1prior=x1prior, 
                                   cprior= extinction.highIa_c, avprior= extinction.highCC, classfractions=cfHigh, 
                                   zprior=zprior, clobber=1, verbose=min(0,verbose-1), debug=False,
                                   kcorfile=kcorfile, pdzfile=pdzfile,  magnification=magnification,
                                   nlogz=nlogz, ncolorpar=ncolorpar, ncolorlaw=ncolorlaw, nlumipar=nlumipar, npkmjd=npkmjd, 
                                   omitTemplateIbc=omitTemplateIbc, omitTemplateII=omitTemplateII, getSystematicError=False )
            pialist.append( snCopy.PIa )

            # Lower Limit on P(Ia) : low dust and low Ia fraction priors
            snCopy.doGridClassify( bands=bands, Nsim=Nsim, trestrange=trestrange, modelerror=modelerror, 
                                errfloor=errfloor, useLuminosityPrior=useLuminosityPrior, x1prior=x1prior, 
                                cprior= extinction.lowIa_c, avprior= extinction.lowCC, classfractions=cfLow, 
                                zprior=zprior, clobber=1, verbose=min(0,verbose-1), debug=debug,
                                kcorfile=kcorfile, pdzfile=pdzfile,  magnification=magnification,
                                nlogz=nlogz, ncolorpar=ncolorpar, ncolorlaw=ncolorlaw, nlumipar=nlumipar, npkmjd=npkmjd, 
                                omitTemplateIbc=omitTemplateIbc, omitTemplateII=omitTemplateII, getSystematicError=False )
            pialist.append( snCopy.PIa )

            del( snCopy )
            self.PIaMin = np.min( pialist ) 
            self.PIaMax = np.max( pialist ) 
            self.PIaString = '%.2f +%.2f -%.2f'%( self.PIa, abs(self.PIaMax-self.PIa), abs(self.PIa-self.PIaMin) )
            if verbose : print("%s P(Ia) = %s"%(self.nickname, self.PIaString ) )
        if verbose : print( "%s : PIa,Ibc,II = %.2f  %.2f  %.2f"%(self.nickname, self.PIa, self.PIbc, self.PII) )

    def getChi2LikelihoodGrid( self, classname, Nsim=2000, bands='all', trestrange=[-15,30], 
                               modelerror=0.05, errfloor=0.001, inflateUVerr=True, 
                               useLuminosityPrior=True, magnification=1,
                               verbose=True, clobber=False, debug=False, 
                               nlogz=0, ncolorpar=0, ncolorlaw=0, nlumipar=0, npkmjd=0,
                               omitTemplateIbc='', omitTemplateII='',
                               kcorfile='HST/kcor_HST.fits' ):
        """D. Jones - 3/27/13 

        classname  :  the SN class to compare against ('Ia', 'Ibc', or 'II')
                     This selects which snana.SimTable object to use (presumably generated 
                     in a classification simulation, so that it contains 2-D phot data arrays 
                     holding all the simulated light curves for that class)
        bands      : a string listing the bands to fit, e.g. 'HJW'. Use 'all' for all bands (default)
        trestrange : fit only observations within this rest-frame time window (rel. to peak)  
        modelerror : fractional flux error to apply to each SN model for chi2 calculation
        errfloor   : minimum flux error for the model (e.g. for zero-flux extrapolations)
        useLuminosityPrior  : 
                  if False: allow a free parameter for scaling the flux of each simulated 
                       SN so that it most closely matches the observed fluxes
                  if == True or == 1 : allow the free parameter for flux scaling, but 
                       also apply a prior based on the value of that optimal flux scaling
                       factor, relative to the assumed luminosity function for each class
                  if >1,  extend the CCSN simulation grids along a new dimension
                       that samples the assumed luminosity functions (and apply priors)

        STORED RESULTS : 
          self.chi2<class> : 1-d array of chi2 values (one for each synthetic SN)
          self.like<class> : 1-d array of likelihood values (one for each synthetic SN)

        RETURNS :
          chi2vector  : the 1-d array of chi2 values, one for each comparison model in classSimTab 
          postProb    : the posterior probability, with one element for each comparison SN model in classSimTab.
        """
        # NOTE : For now the useLuminosityPrior=2  setting only results in an 
        #  extra dimension in the simulated grid for CCSNe. We use the nlumipar 
        #  input parameter (normally irrelevent for CCSNe) to set the number of 
        #  sampling points along the luminosity function.  This implicitly assumes
        #  that the variance of the Ia model is fully captured by the model 
        #  uncertainty paramater, but we reserve the right to change this 
        #  assumption in the future. 

        if debug : import pdb; pdb.set_trace()
        from constants import CCMODELS

        if bands=='all' : bands = self.bands

        # read in the classification simulation metadata 
        if clobber>2 or 'ClassSim' not in self.__dict__ :
            if verbose>1 : print("running getGridSim() to generate+read the grid simulation")
            self.getGridSim( Nsim=Nsim, verbose=verbose, 
                             clobber=max(0,clobber-1), kcorfile=kcorfile,
                             trestrange=trestrange, nlogz=nlogz, ncolorpar=ncolorpar, 
                             ncolorlaw=ncolorlaw, nlumipar=nlumipar,
                             omitTemplateIbc=omitTemplateIbc,
                             omitTemplateII=omitTemplateII )

        # read in the simulated light curves for the class of interest
        if classname=='Ia': 
            if clobber>1 or 'postProb' not in self.ClassSim.Ia.__dict__ : 
                if verbose: print("Reading in %i simulated Type Ia light curves."%self.ClassSim.Ia.nsim)
                self.ClassSim.Ia.readgrid( )
            classSimTab = self.ClassSim.Ia
            nmagoff = 1
        elif classname=='Ibc': 
            if clobber>1 or 'postProb' not in self.ClassSim.Ibc.__dict__ : 
                if verbose: print("Reading in %i simulated Type Ib/c light curves."%self.ClassSim.Ibc.nsim)
                self.ClassSim.Ibc.readgrid( )
            classSimTab = self.ClassSim.Ibc
            if useLuminosityPrior>1 : nmagoff = nlumipar
            else : nmagoff=1
        elif classname=='II' : 
            if clobber>1 or 'postProb' not in self.ClassSim.II.__dict__ : 
                if verbose: print("Reading in %i simulated Type II light curves."%self.ClassSim.II.nsim)
                self.ClassSim.II.readgrid( )
            classSimTab = self.ClassSim.II
            if useLuminosityPrior>1 : nmagoff = nlumipar
            else : nmagoff=1
        else : 
            raise exceptions.RuntimeError("I don't know how to classify for type %s"%classname)
       
        # sort the observed and simulated SN light curves by MJD
        isortobs = self.MJD.argsort()
        isortsim = classSimTab.TREST.argsort()

        # make a list of obs indices (from the mjd-sorted list) for 
        # observations  that we should include in the calculation
        # We define 'trestobs' as the rest-frame observation dates
        # for the observed SN light curve by assuming that the 
        # observed SN is precisely at the best-avaialable redshift (self.z)
        # and use that assumption to define 'igoodobs', which excludes
        # any points  outside the valid range of the simulation models:
        mjdobs = self.MJD[ isortobs ]
        trestobs = (mjdobs - self.mjdpk)/(1+self.z) 
        fltobs = self.FLT[isortobs]
        fluxobs = self.FLUXCAL[isortobs]
        fluxerrobs = self.FLUXCALERR[isortobs]
        igoodobs = np.where( (trestobs>trestrange[0]) & 
                             (trestobs<trestrange[1]) & 
                             ( np.array([ flt in bands for flt in fltobs ] )) )[0]

        # We have 2-D FLUXCAL and FLUXCALERR arrays from the Grid SimTable,
        # where the first dimension contains 4 sim parameters, and the
        # second dimension comprises the filter and rest-frame time rel. to peak.

        # Here we extend the first dimension by npkmjd steps to include the PKMJD 
        # dimension of parameter space, and we interpolate between the points to 
        # match the MJD values of the observed light curve
        
        # For CC simulations we also extend into another dimension with length 
        # nmagoff (for SNIa we've set nmagoff=1) where each position sets 
        # a different magnitude offset (i.e. flux scaling factor), so as to
        # sample the luminosity functions

        # Set up zero arrays to get filled with simulated SN fluxes. 
        # - In fluxsim_tall we include all simulated observation dates, making 
        # it suitable for defining our interpolation functions and for constructing
        # a well-sampled model for plotting.
        # - mjdsim_tall will carry the observer-frame MJD for each simulated SN,
        # after appropriately shifting the date of peak.
        # - In fluxsim_tobs each simulated SN will be interpolated and sampled at 
        # the real SN's observation dates. This is what we will use to compute 
        # the chi2 and likelihood arrays.
        ## fluxsim_tall = np.zeros([len(classSimTab.FLT) * npkmjd * nmagoff, len(classSimTab.FLT[0])])
        ## mjdsim_tall = np.zeros([len(classSimTab.FLT) * npkmjd * nmagoff, len(classSimTab.TREST)])
        ## fluxsim_tobs = np.zeros([len(classSimTab.FLT) * npkmjd * nmagoff, len(igoodobs)]) 

        # Define the PKMJD values we will sample
        # and set up an array containing the PKMJD 
        # values at each pkmjd grid point
        pkmjdmin =  self.pkmjd - self.pkmjderr
        pkmjdstep = (2.*self.pkmjderr+1) / npkmjd
        pkmjdvalues = np.array([ pkmjdmin + ipkmjd*pkmjdstep for ipkmjd in xrange(npkmjd) ])
        classSimTab.PKMJD = pkmjdvalues

        nlogz = len(classSimTab.LOGZ)
        ncolorpar= len(classSimTab.COLORPAR)
        ncolorlaw = len(classSimTab.COLORLAW)
        nlumipar = len(classSimTab.LUMIPAR)

        classSimTab.NDOF = max(1,len(igoodobs) - (nlogz>1) - (ncolorpar>1) - (ncolorlaw>1) - (nlumipar>1) - (npkmjd>1) - (nmagoff>1))

        # define a few convenience functions for dealing with grid indices
        def index2gridCoord( isn ) : 
            """ convert the integer index 'isn' into a list 
            of parameter indices, one for each dimension of 
            the simulation grid (including the extra dimensions
            used for pkmjd and luminosity function flux scaling)

            isn = 1 + sum_{i=1}^{4} ( off_i * (idx_i - 1)

             i=1 : redshift    i=2 : colorpar
             i=3 : colorlaw    i=4 : lumipar / template
             ( i=5 : pkmjd     i=6 : fluxscale )
             
            idx_i runs from 1 to NGRID_i for parameter i

                   logz     colorpar  colorlaw  lumipar    pkmjd    fluxscale
            isn = off1*i1 + off2*i2 + off3*i3 + off4*i4 + off5*i5  + off6*i6 
                = 1 * iz 
                  + nlogz * icp 
                  + nlogz * ncolorpar * icl
                  + nlogz * ncolorpar * ncolorlaw * ilp
                  + nlogz * ncolorpar * ncolorlaw * nlumipar * ipm
                  + nlogz * ncolorpar * ncolorlaw * nlumipar * npkmjd * ifs
            
            USAGE :  
               iz, icp, icl, ilp, ipm, ifs = index2gridCoord( iminchi2 )              
            """
            off1 = nlogz 
            off2 = nlogz * ncolorpar 
            off3 = nlogz * ncolorpar * ncolorlaw 
            off4 = nlogz * ncolorpar * ncolorlaw * nlumipar 
            off5 = nlogz * ncolorpar * ncolorlaw * nlumipar * npkmjd 

            ifluxscale= int( (isn) / off5 ) 
            ipkmjd    = int( (isn-(ifluxscale*off5)) / off4 ) 
            ilumipar  = int( (isn-(ifluxscale*off5)-(ipkmjd*off4)) / off3 )
            icolorlaw = int( (isn-(ifluxscale*off5)-(ipkmjd*off4)-(ilumipar*off3)) / off2 )
            icolorpar = int( (isn-(ifluxscale*off5)-(ipkmjd*off4)-(ilumipar*off3)-(icolorlaw*off2)) / off1 )
            ilogz     = isn - (ifluxscale*off5) - (ipkmjd*off4) - (ilumipar*off3) - (icolorlaw*off2) - (icolorpar*off1)
            return( [ilogz,icolorpar,icolorlaw,ilumipar,ipkmjd,ifluxscale] )
        classSimTab.index2gridCoord = index2gridCoord

        def gridCoord2index( iparlist ) : 
            """ Compute the extended-grid index number from a list of parameter
            indices, one for each dimension of the simulation grid (including 
            the extra dimensions for pkmjd and luminosity function).
            isn = 1 * iz 
                  + nlogz * icp 
                  + nlogz * ncolorpar * icl
                  + nlogz * ncolorpar * ncolorlaw * ilp
                  + nlogz * ncolorpar * ncolorlaw * nlumipar * ipm
                  + nlogz * ncolorpar * ncolorlaw * nlumipar * npkmjd * ifs
            NOTE : this index counter starts at 0, so it can be used without 
             modification to access parameter values from the 1-d (raveled) 
             python arrays generated by doGridClassify().
            """
            iz,icp,icl,ilp,ipm,ifs = iparlist 
            isn = iz + nlogz*(icp + ncolorpar*(icl + ncolorlaw*(ilp + nlumipar*(ipm + npkmjd*(ifs) ) ) ) )
            return( isn  )
        classSimTab.gridCoord2index = gridCoord2index

        def gridCoord2snanaIndex( iparlist ) : 
            """ Compute the SNANA simulation index number from a list of parameter
            indices, one for each dimension of the pre-extension simulation grid 
            (i.e. only the parameter dimensions that SNANA generates, so not 
              including the extra STARDUST dimensions for pkmjd and magoff).  
            NOTE : this index counter starts at 1, so it matches up with the SNID
              values encoded in the .GRID file generated by SNANA, and can be used
              as the 'snid' parameter passed to the SuperNova class constructor along
              with the simname. """
            iz,icp,icl,ilp = iparlist[:4] 
            isn = iz + nlogz*(icp + ncolorpar*(icl + ncolorlaw*ilp ) ) + 1
            return( isn  )
        classSimTab.gridCoord2snanaIndex = gridCoord2snanaIndex


        # define dictionaries that pick out the light curve of a single bandpass for interpolation
        #   first the indices in the observed SN where each filter was observed
        ithisfiltobs = dict( [ [f,np.where( fltobs[igoodobs] == f)[0]] for f in bands ])
        #   then ditto for the simulated SNe (all simulated SNe have identical filter sequences)
        ithisfiltsim = dict( [ [f,np.where( classSimTab.FLT[0] == f)[0]] for f in bands ])  
        
        if useLuminosityPrior>1 and classname!='Ia':  
            # create a look-up table with arrays of flux scaling factors that 
            # define the luminosity function for each SN sub-type 
            subtypelist = np.unique(classSimTab.SIM_SUBTYPE)
            fluxscaledict = dict( [ [subtype, [magOffsetFluxScale( subtype, imagoff, nmagoff )
                                               for imagoff in xrange(nmagoff) ] ] 
                                    for subtype in subtypelist ] )
            fluxscalepriordict = dict( [ [subtype, [magOffsetPrior( subtype, imagoff, nmagoff )
                                               for imagoff in xrange(nmagoff) ] ] 
                                         for subtype in subtypelist ] )

        # Here we step through all 6 dimensions of the simulation space and fill
        # in the chi2 and likelihood arrays
        fluxsim_tall = np.zeros([len(classSimTab.FLT) * npkmjd * nmagoff, len(classSimTab.FLT[0])])

        ngridpos = classSimTab.nsim * npkmjd * nmagoff 
        classSimTab.FLUXSCALE = np.zeros( ngridpos )
        classSimTab.FLUXSCALEPRIOR = np.zeros( ngridpos )
        self.__dict__['chi2%s'%classname] = np.zeros( ngridpos )
        self.__dict__['like%s'%classname] = np.zeros( ngridpos )
        for iz in xrange(nlogz) :
            tobssim = classSimTab.TOBS[iz] # observer-frame time for simulated SNe at this redshift
            zthissim = classSimTab.z[iz]
            for icp in xrange(ncolorpar) :
                for icl in xrange(ncolorlaw) :
                    for ilp in xrange(nlumipar) :
                        if classname=='Ia': subtype='Ia'
                        else : subtype = classSimTab.SIM_SUBTYPE[ilp]
                        isimSN =  iz + nlogz*(icp + ncolorpar*(icl + ncolorlaw*(ilp) ) )
                        fluxsim = classSimTab.FLUXCAL[isimSN] 
                        fluxerrsim = classSimTab.FLUXCALERR[isimSN]

                        for ipm in xrange(npkmjd) :
                            mjdsim = tobssim + pkmjdvalues[ipm]  # the same mjdsim array is applicable for all filters

                            # for each band, interpolate from simulated MJDs to the actual observation times
                            fluxsim_tobs_unscaled = dict( [ [ f, np.interp( mjdobs[igoodobs][ithisfiltobs[f]], mjdsim, 
                                                                            fluxsim[ithisfiltsim[f]], left=0, right=0 ) ]
                                                            for f in bands ] )

                            # construct a vector of model errors, one for each obs point, 
                            # inflating the rest-frame UVIS uncertainty if requested
                            if inflateUVerr : 
                                # TODO : add a trest component to increase uncertainties far from peak
                                modelerrorObs, errfloorObs = self.getmodelerror(  fltobs[igoodobs], zthissim, modelerror, errfloor )
                            else : 
                                modelerrorObs = np.ones( len(igoodobs) ) * modelerror 
                                errfloorObs = np.ones( len(igoodobs) ) * errfloor 

                            for imagoff in xrange(nmagoff) :                                
                                isimSNextended = gridCoord2index( [iz,icp,icl,ilp,ipm,imagoff] ) 

                                # fill in the fluxsim_tall and mjdsim_tall arrays with
                                # the simulated flux points in all bands and all simulated times
                                # mjdsim_tall[ isimSNextended ] = mjdsim
                                # fluxsim_tall[ isimSNextended ]  = fluxsim

                                # fill in the fluxsim_tobs array, giving the simulated flux at each 
                                # observed date in the appropriate band for that observation.
                                fluxsim_tobs = np.zeros( len(igoodobs) ) 
                                for f in bands:
                                    fluxsim_tobs[ithisfiltobs[f]]  = np.interp( mjdobs[igoodobs][ithisfiltobs[f]], mjdsim, fluxsim[ithisfiltsim[f]], left=0, right=0 )
                                    
                                # compute the mag offset (flux scaling factor) and apply it
                                if useLuminosityPrior>1 and classname!='Ia':  
                                    # Define the magnitude offset (i.e. flux scaling
                                    # factor) to be applied to each simulated SN so
                                    # as to sample the luminosity function of each 
                                    # sub-class. (For SNIa the lum. function is 
                                    # already baked in to the model via x1 and c)
                                    fluxscale = fluxscaledict[subtype][imagoff] 
                                    priorfluxscale = fluxscalepriordict[subtype][imagoff][0]
                                elif useLuminosityPrior : 
                                    if classname == 'Ia' : 
                                        fluxscale, priorfluxscale = 1, 1
                                    else : 
                                        # Define the optimal scaling factor
                                        # that minimizes the chi2 for each individual simulated SN
                                        fluxscale = optimalFluxScale( fluxsim_tobs, fluxobs[igoodobs], fluxerrobs[igoodobs], modelerrorObs )
                                        priorfluxscale  = optimalFluxScalePrior( fluxscale, subtype )
                                else : 
                                    fluxscale = optimalFluxScale( fluxsim_tobs, fluxobs[igoodobs], fluxerrobs[igoodobs], modelerrorObs )
                                    #fluxscale = 1
                                    priorfluxscale = 1

                                classSimTab.FLUXSCALE[isimSNextended] = fluxscale 
                                classSimTab.FLUXSCALEPRIOR[isimSNextended] = priorfluxscale
                                fluxerrsim_tobs =  (fluxsim_tobs*fluxscale * modelerrorObs + errfloorObs )

                                # compute the total chi2, comparing this single simulated SN to our observed SN
                                chi2val = ( (fluxsim_tobs*fluxscale - fluxobs[igoodobs])**2 / (fluxerrsim_tobs**2 + fluxerrobs[igoodobs]**2) ).sum()
                                self.__dict__['chi2%s'%classname][isimSNextended] = chi2val

                                # compute the likelihood, assuming gaussian errors, i.e.  e^(-chi2/2)
                                sig2sum = (fluxerrsim_tobs**2 + fluxerrobs[igoodobs]**2).sum()
                                likeval = np.exp(-chi2val/2) / np.sqrt( 2 * np.pi * sig2sum  )
                                self.__dict__['like%s'%classname][isimSNextended] = likeval

        classSimTab.USELUMPRIOR = useLuminosityPrior 
        classSimTab.NMAGOFFSET =  nmagoff
        if useLuminosityPrior>1 and classname=='Ia' : 
            classSimTab.FLUXSCALE = [1]
            classSimTab.FLUXSCALEPRIOR = [1]
            classSimTab.MAGOFFSET = [0]
            classSimTab.MAGOFFSETPRIOR = [1]
        else : 
            classSimTab.FLUXSCALE = np.array( classSimTab.FLUXSCALE )
            classSimTab.FLUXSCALEPRIOR = np.array( classSimTab.FLUXSCALEPRIOR )
            posfluxscale = np.where( classSimTab.FLUXSCALE>0, classSimTab.FLUXSCALE, 1 )
            classSimTab.MAGOFFSET = -2.5*np.log10( posfluxscale )
            classSimTab.MAGOFFSETPRIOR = classSimTab.FLUXSCALEPRIOR

        self.Ndof = classSimTab.NDOF
        self.__dict__['Ndof%s'%classname] = classSimTab.NDOF
        self.__dict__['chi2%s'%classname] = np.array( self.__dict__['chi2%s'%classname] )
        self.__dict__['like%s'%classname] = np.array( self.__dict__['like%s'%classname] )
        classSimTab.CHI2 = self.__dict__['chi2%s'%classname]
        classSimTab.LIKE = self.__dict__['like%s'%classname]
        classSimTab.FLUXSCALED = fluxsim_tobs
        classSimTab.FLUXERRSCALED = fluxerrsim_tobs

        # construct a SuperNova object to hold the max likelihood (i.e. min chi2) model for this type
        iminchi2 =  self.__dict__['chi2%s'%classname].argmin()
        iz,icp,icl,ilp,ipm,imagoff = index2gridCoord( iminchi2 )
        iminchi2snana = gridCoord2snanaIndex( [iz,icp,icl,ilp,ipm,imagoff] )
        maxLikeModel = SuperNova() 
        maxLikeModel.NAME = 'maxLike%sModel'%classname
        maxLikeModel.SURVEY = self.SURVEY
        maxLikeModel.SURVEYDATA = self.SURVEYDATA
        maxLikeModel.SIMNAME = classSimTab.simname
        maxLikeModel.SNID = iminchi2snana
        maxLikeModel.INDEX = iminchi2
        maxLikeModel.SNANAINDEX = iminchi2snana
        maxLikeModel.FLT = classSimTab.FLT[0]
        tobssim = classSimTab.TOBS[iz] # observer-frame time for simulated SNe at this redshift
        mjdsim = tobssim + pkmjdvalues[ipm]  # the same mjdsim array is applicable for all filters
        maxLikeModel.MJD = np.ravel( [ mjdsim for band in maxLikeModel.bands] )

        if useLuminosityPrior>1 and classname=='Ia' : 
            maxLikeModel.FLUXSCALE = 1
            maxLikeModel.MAGOFF =  0
        else : 
            maxLikeModel.FLUXSCALE = classSimTab.FLUXSCALE[iminchi2]
            if maxLikeModel.FLUXSCALE > 0 : 
                maxLikeModel.MAGOFF = ( -2.5*np.log10( maxLikeModel.FLUXSCALE ) )
            else : 
                maxLikeModel.MAGOFF = np.nan

        maxLikeModel.REDSHIFT = classSimTab.z[iz]
        modelerrorSim, errfloorSim = self.getmodelerror(  maxLikeModel.FLT, maxLikeModel.REDSHIFT, modelerror, errfloor )
        fluxmaxlike = classSimTab.FLUXCAL[iminchi2snana-1] * maxLikeModel.FLUXSCALE
        fluxerrmaxlike = fluxmaxlike * modelerrorSim + errfloorSim
        maxLikeModel.FLUXCAL = fluxmaxlike
        maxLikeModel.FLUXCALERR = fluxerrmaxlike
        posfluxmaxlike = np.where( fluxmaxlike>0, fluxmaxlike, 0.015848931924611134 )
        maxLikeModel.MAG = -2.5*np.log10( posfluxmaxlike ) + 27.5 
        maxLikeModel.MAGERR = np.where( fluxmaxlike!=0, np.abs(1.0857*fluxerrmaxlike/posfluxmaxlike), 
                                        -2.5*np.log10( errfloor )+27.5 ) 
        maxLikeModel.LOGZ = classSimTab.LOGZ[iz]
        maxLikeModel.COLORPAR = classSimTab.COLORPAR[icp]
        maxLikeModel.COLORLAW = classSimTab.COLORLAW[icl]
        maxLikeModel.LUMIPAR = classSimTab.LUMIPAR[ilp]
        maxLikeModel.PEAKMJD = pkmjdvalues[ipm]
        maxLikeModel.PEAKMJDERR = np.mean( np.diff(pkmjdvalues) )/2.
        fluxmaxlike_tobs = np.zeros( len(igoodobs) ) 
        for f in bands:
            fluxmaxlike_tobs[ithisfiltobs[f]]  = np.interp( mjdobs[igoodobs][ithisfiltobs[f]], maxLikeModel.MJD[ithisfiltsim[f]], fluxmaxlike[ithisfiltsim[f]], left=0, right=0 )
        fluxerrmaxlike_tobs = fluxmaxlike_tobs * modelerrorObs + errfloorObs
        maxLikeModel.CHI2VEC = (fluxmaxlike_tobs - fluxobs[igoodobs])**2 / (fluxerrmaxlike_tobs**2 + fluxerrobs[igoodobs]**2)
        maxLikeModel.CHI2MJD = mjdobs[igoodobs]
        maxLikeModel.CHI2FLT = fltobs[igoodobs]
        maxLikeModel.NDOF = classSimTab.NDOF

        if classname == 'Ia' : 
            maxLikeModel.TEMPLATE = classSimTab.GENMODEL
            maxLikeModel.TYPE = 'Ia'
        else : 
            maxLikeModel.TYPE =  CCMODELS['%03i'%maxLikeModel.LUMIPAR][0]
            maxLikeModel.TEMPLATE = CCMODELS['%03i'%maxLikeModel.LUMIPAR][1]

        self.__dict__['maxLike%sModel'%classname] = maxLikeModel

        return( None )



    def getGridSim( self, simroot='HST_classify', Nsim=2000, 
                    simpriors=False, clobber=False, verbose=False, 
                    kcorfile='HST/kcor_HST.fits', trestrange=[-15,35],
                    nlogz=0, ncolorpar=0, ncolorlaw=0, nlumipar=0,
                    omitTemplateIbc='', omitTemplateII='' ):
        """ D. Jones - 3/27/13
        run and/or read in the results of a SNANA GRID simulation for 
        photometric classification. 
        simpriors : if True, embed priors within the simulation, so simulated SNe
            reflect realistic distributions in shape, color, redshift, etc.
            if False, use flat distributions (presumably priors get applied later, as
             in the case of a bayesian classification approach)
        trestrange : the range of time to simulate, in rest-frame days rel. to peak 
        """
        # TODO : adjust the pkmjdrange by class

        # set the redshift range of the simulation
        zmin = self.z-self.zerr
        zmax = self.z+self.zerr

        # TODO : get the range of plausible peak mjds (as defined by the .dat file)
        # and use it to define the default trestrange 
        #if trestrange==None : 
        #    pkmjdrange = [self.pkmjd-self.pkmjderr,self.pkmjd+self.pkmjderr]
        #    trestrange = [ self.pkmjderrpkmjdrange[0]

        # check for existing simulation products
        sndatadir = os.environ['SNDATA_ROOT']
        simname = '%s_%s'%(simroot,self.name)
        simdatadir = os.path.join( sndatadir,'SIM/%s'%simname )
        simisdone = np.all([ os.path.isfile(os.path.join(simdatadir+'_%s'%sntype,'%s_%s.GRID'%(simname,sntype))) 
                             for sntype in ['Ia','Ibc','II'] ] )
        if not simisdone or clobber : 
            # run the simulation 
            if not simisdone and verbose>1: print("%s simulation does not exist. Running SNANA..."%simname)
            elif clobber and verbose>1: print("Clobbering existing %s simulation. Running SNANA..."%simname)
            simname = classify.doGridSim( simroot=simname, Nsim=Nsim, zrange=[zmin,zmax], trestrange=trestrange,
                                          bands=self.bands, kcorfile=kcorfile, 
                                          nlogz=nlogz, ncolorpar=ncolorpar, ncolorlaw=ncolorlaw, nlumipar=nlumipar, 
                                          clobber=clobber, omitTemplateIbc=omitTemplateIbc, omitTemplateII=omitTemplateII )

        elif verbose>1 : 
            print( "%s simulation exists. Not clobbering."%simname )

        # read in the simulation results if needed
        needread = True
        if 'ClassSim' in self.__dict__  and not clobber : 
            simnameIa = self.ClassSim.Ia.simname 
            if simnameIa == simname+'_Ia' : 
                needread = False
                if verbose>1 : print( "%s sim already exists as .ClassSim object."%simname )
        if needread : 
            self.ClassSim = classify.rdClassSim(simname,verbose=verbose)
            if verbose>1: print("%s sim imported as .ClassSim object"%simname )


    def getMaxLikeModelsMC( self, trestrange=[-15,30], applypriors=False, clobber=False, verbose=False, **kwargs ) :
        """ Find the maximum likelihood light curve for each class, using  
        likelihood arrays computed from comparison with a SNANA simulation table 
        in doClassify(). 
        Run a single-object simulation for each class to produce a 1-day-cadence
        synthetic light curve (suitable for plotting). Store each max-likelihood
        SN model as a separate SuperNova object. 

        OPTIONS: 
        applypriors : when selecting the best fit light curve, include the priors 
            on SN light curve and location (i.e. use the posterior
            probability distribution instead of the likelihood distribution)
        trestrange :  trestrange is computed automatically for the best-fit z and mjdpk

        """
        from constants import CCMODELS

        # check if products from the probability computations are already present,
        # run getClassProb if needed: 
        if( 'ClassMC' not in self.__dict__ ): 
            if verbose: print("ERROR: no ClassMC object available. Run  doClassifyMC to get classification probability distributions.")
            return(None) 

        # set up an empty SN sequence object to hold the max likelihood models
        ClassMC = self.ClassMC
        ClassMC.maxLikeModels = SuperNovaSet( SuperNova(), SuperNova(), SuperNova() )

        # generate a new simulation with 1-day sampling for each max-like fit a la SALT2 
        if applypriors : imaxIa = ClassMC.Ia.PROB.argmax()
        else : imaxIa = ClassMC.Ia.LIKE.argmax()
        pkmjd = ClassMC.Ia.SIM_PEAKMJD[imaxIa]
        z  = ClassMC.Ia.SIM_REDSHIFT[imaxIa]
        mB  = ClassMC.Ia.SIM_SALT2mB[imaxIa]
        x1  = ClassMC.Ia.SIM_SALT2x1[imaxIa]
        c  = ClassMC.Ia.SIM_SALT2c[imaxIa]
        beta = ClassMC.Ia.SIM_SALT2beta[imaxIa]
        simname = 'sim_%s_maxlikeIa_z%.2f'%(self.nickname.lower(), z )
        #trestrange = [ (self.MJD.min()-pkmjd)/(1+z)-2, (self.MJD.max()-pkmjd)/(1+z)+5 ]
        ClassMC.maxLikeModels.Ia = simulate.doSingleSim( 
            simname=simname, z=z, pkmjd=pkmjd, model='Ia', 
            trestrange=trestrange, Av=0, mB=mB, x1=x1, c=c, 
            survey=self.SURVEY, field=self.SURVEYDATA.FIELDNAME, 
            bands=self.bands,
            cadence=0, mjdlist=[], bandlist=[],  
            perfect=True, verbose=verbose, clobber=clobber )
        if ClassMC.Ia.modelerror >0 : 
            #  overwrite the fluxcalerr and magerr vectors using the modelerror and errfloor parameters
            errfloor = ClassMC.Ia.errfloor
            if errfloor=='auto': 
                errfloor = getErrFloor( ClassMC.maxLikeModels.Ia.SURVEYDATADICT, ClassMC.maxLikeModels.Ia.z, ClassMC.maxLikeModels.Ia.FLT, 'Ia' )
            ClassMC.maxLikeModels.Ia.FLUXCALERR = np.max( [ClassMC.maxLikeModels.Ia.FLUXCAL * ClassMC.Ia.modelerror, 
                                                           errfloor * np.ones(ClassMC.maxLikeModels.Ia.nobs)] , 
                                                          axis=0 )
            ClassMC.maxLikeModels.Ia.MAGERR = 1.0857 * ClassMC.maxLikeModels.Ia.FLUXCALERR / ( ClassMC.maxLikeModels.Ia.FLUXCAL + errfloor )
        ClassMC.maxLikeModels.Ia.TYPE='Ia'
        ClassMC.maxLikeModels.Ia.TEMPLATE='SALT2'
        ClassMC.maxLikeModels.Ia.LUMIPAR=x1
        ClassMC.maxLikeModels.Ia.COLORPAR=c
        ClassMC.maxLikeModels.Ia.COLORLAW=beta
        ClassMC.maxLikeModels.Ia.REDSHIFT=z
        ClassMC.maxLikeModels.Ia.PEAKMJD=pkmjd
        ClassMC.maxLikeModels.Ia.MAGOFF=0
        ClassMC.maxLikeModels.Ia.CHI2VEC, ClassMC.maxLikeModels.Ia.CHI2FLT, ClassMC.maxLikeModels.Ia.NDOF, ClassMC.maxLikeModels.Ia.like  = self.chi2likeSingle( ClassMC.maxLikeModels.Ia, trestrange=trestrange )

        if applypriors : imaxIbc = ClassMC.Ibc.PROB.argmax()
        else : imaxIbc = ClassMC.Ibc.LIKE.argmax()
        pkmjd = ClassMC.Ibc.SIM_PEAKMJD[imaxIbc]
        z  = ClassMC.Ibc.SIM_REDSHIFT[imaxIbc]
        Av  = ClassMC.Ibc.SIM_AV[imaxIbc]
        Rv  = ClassMC.Ibc.SIM_RV[imaxIbc]
        simname = 'sim_%s_maxlikeIbc_z%.2f'%(self.nickname.lower(), z )
        model = '%s.%s'%(ClassMC.Ibc.SNTYPE[imaxIbc],ClassMC.Ibc.SIM_NON1a[imaxIbc])
        # trestrange = [ (self.MJD.min()-pkmjd)/(1+z)-2, (self.MJD.max()-pkmjd)/(1+z)+5 ]
        ClassMC.maxLikeModels.Ibc = simulate.doSingleSim( 
            simname=simname, z=z, pkmjd=pkmjd, model=model, 
            trestrange=trestrange, Av=Av, mB=0, x1=0, c=0, 
            survey=self.SURVEY, field=self.SURVEYDATA.FIELDNAME, bands=self.bands,
            cadence=0, mjdlist=[], bandlist=[], 
            perfect=True, verbose=verbose, clobber=clobber )
        non1aModelID = '%03i'%int(model.split('.')[1])
        ClassMC.maxLikeModels.Ibc.TYPE=CCMODELS[ non1aModelID ][0]
        ClassMC.maxLikeModels.Ibc.TEMPLATE=CCMODELS[ non1aModelID ][1]
        ClassMC.maxLikeModels.Ibc.LUMIPAR=0
        ClassMC.maxLikeModels.Ibc.COLORPAR=Av
        ClassMC.maxLikeModels.Ibc.Av = Av
        ClassMC.maxLikeModels.Ibc.COLORLAW=Rv
        ClassMC.maxLikeModels.Ibc.REDSHIFT=z
        ClassMC.maxLikeModels.Ibc.PEAKMJD=pkmjd

        # The simulated SN in the ClassMC table that yields the
        # minimum chi2 has had some MAGSMEAR offset applied by SNANA
        # in the classification monte carlo sim, but the value of that
        # offset is not stored anywhere.  Our new maxLikeModel has no
        # mag smearing applied, so we need to solve for and then put
        # in the same mag offset in order to match the original
        # simulated SN. 
        # NOTE : although the mag smearing is nominally a coherent
        #  magnitude offset applied equally across all bands, in practice
        #  there are small deviations from band to band in the simulated 
        #  peak magnitudes for our maxLikeModels, due to rounding errors 
        #  in the simulation parameters (Av, z, peakmjd)
        magoffListIbc = []
        for band in self.bands : 
            pkmagClassMC = ClassMC.Ibc.__dict__[ 'SIM_PEAKMAG_%s'%band ][imaxIbc]
            pkmagMaxLikeModel = ClassMC.maxLikeModels.Ibc.__dict__[  'SIM_PEAKMAG_%s'%band ]
            magoff = pkmagClassMC - pkmagMaxLikeModel
            magoffListIbc.append( magoff ) 
            iband = np.where( ClassMC.maxLikeModels.Ibc.FLT ==band )[0]
            ClassMC.maxLikeModels.Ibc.MAG[ iband ] = ClassMC.maxLikeModels.Ibc.MAG[ iband ]  + magoff
            ClassMC.maxLikeModels.Ibc.FLUXCAL[ iband ] = ClassMC.maxLikeModels.Ibc.FLUXCAL[ iband ] * 10**(-0.4*magoff)
        ClassMC.maxLikeModels.Ibc.MAGOFF = np.mean(magoffListIbc)
        ClassMC.maxLikeModels.Ibc.MAGOFFSTD = np.std(magoffListIbc)
        if ClassMC.Ibc.modelerror >0 : 
            #  overwrite the fluxcalerr and magerr vectors using the modelerror and errfloor parameters
            errfloor = ClassMC.Ibc.errfloor
            if errfloor=='auto': 
                errfloor = getErrFloor( ClassMC.maxLikeModels.Ibc.SURVEYDATADICT, ClassMC.maxLikeModels.Ibc.z, ClassMC.maxLikeModels.Ibc.FLT, 'Ibc' )
            ClassMC.maxLikeModels.Ibc.FLUXCALERR = np.max( [ClassMC.maxLikeModels.Ibc.FLUXCAL * ClassMC.Ibc.modelerror, 
                                                           errfloor * np.ones(ClassMC.maxLikeModels.Ibc.nobs)] , 
                                                          axis=0 )
            ClassMC.maxLikeModels.Ibc.MAGERR = 1.0857 * ClassMC.maxLikeModels.Ibc.FLUXCALERR / ( ClassMC.maxLikeModels.Ibc.FLUXCAL + errfloor )
        ClassMC.maxLikeModels.Ibc.CHI2VEC, ClassMC.maxLikeModels.Ibc.CHI2FLT, ClassMC.maxLikeModels.Ibc.NDOF, ClassMC.maxLikeModels.Ibc.like  = self.chi2likeSingle( ClassMC.maxLikeModels.Ibc, trestrange=trestrange )


        if applypriors : imaxII = ClassMC.II.PROB.argmax()
        else : imaxII = ClassMC.II.LIKE.argmax()
        pkmjd = ClassMC.II.SIM_PEAKMJD[imaxII]
        z  = ClassMC.II.SIM_REDSHIFT[imaxII]
        Av  = ClassMC.II.DUMP['AV'][imaxII]
        model = '%s.%s'%(ClassMC.II.SNTYPE[imaxII],ClassMC.II.SIM_NON1a[imaxII])
        simname = 'sim_%s_maxlikeII_z%.2f'%(self.nickname.lower(), z )
        # trestrange = [ (self.MJD.min()-pkmjd)/(1+z)-2, (self.MJD.max()-pkmjd)/(1+z)+5 ]
        ClassMC.maxLikeModels.II = simulate.doSingleSim( 
            simname=simname, z=z, pkmjd=pkmjd, model=model, 
            trestrange=trestrange, Av=Av, mB=0, x1=0, c=0, 
            survey=self.SURVEY, field=self.SURVEYDATA.FIELDNAME, bands=self.bands,
            cadence=0, mjdlist=[], bandlist=[], 
            perfect=True, verbose=verbose, clobber=clobber )
        non1aModelID = '%03i'%int(model.split('.')[1])
        ClassMC.maxLikeModels.II.TYPE=CCMODELS[ non1aModelID ][0]
        ClassMC.maxLikeModels.II.TEMPLATE=CCMODELS[ non1aModelID ][1]
        ClassMC.maxLikeModels.II.LUMIPAR=0
        ClassMC.maxLikeModels.II.COLORPAR=Av
        ClassMC.maxLikeModels.II.Av = Av
        ClassMC.maxLikeModels.II.COLORLAW=Rv
        ClassMC.maxLikeModels.II.REDSHIFT=z
        ClassMC.maxLikeModels.II.PEAKMJD=pkmjd
        # compute the mag offset (see note above)
        magoffListII = []
        for band in self.bands : 
            pkmagClassMC = ClassMC.II.__dict__[ 'SIM_PEAKMAG_%s'%band ][imaxII]
            pkmagMaxLikeModel = ClassMC.maxLikeModels.II.__dict__[  'SIM_PEAKMAG_%s'%band ]
            magoff = pkmagClassMC - pkmagMaxLikeModel
            magoffListII.append( magoff ) 
            iband = np.where( ClassMC.maxLikeModels.II.FLT ==band )[0]
            ClassMC.maxLikeModels.II.MAG[ iband ] = ClassMC.maxLikeModels.II.MAG[ iband ] + magoff
            ClassMC.maxLikeModels.II.FLUXCAL[ iband ] = ClassMC.maxLikeModels.II.FLUXCAL[ iband ] * 10**(-0.4*magoff)
        ClassMC.maxLikeModels.II.MAGOFF = np.mean(magoffListII)
        ClassMC.maxLikeModels.II.MAGOFFSTD = np.std(magoffListII)
        if ClassMC.II.modelerror >0 : 
            #  overwrite the fluxcalerr and magerr vectors using the modelerror and errfloor parameters
            errfloor = ClassMC.II.errfloor
            if errfloor=='auto': 
                errfloor = getErrFloor( ClassMC.maxLikeModels.II.SURVEYDATADICT, ClassMC.maxLikeModels.II.z, ClassMC.maxLikeModels.II.FLT, 'II' )
            ClassMC.maxLikeModels.II.FLUXCALERR = np.max( [ClassMC.maxLikeModels.II.FLUXCAL * ClassMC.II.modelerror, 
                                                           errfloor * np.ones(ClassMC.maxLikeModels.II.nobs)] , 
                                                          axis=0 )
            ClassMC.maxLikeModels.II.MAGERR = 1.0857 * ClassMC.maxLikeModels.II.FLUXCALERR / ( ClassMC.maxLikeModels.II.FLUXCAL + errfloor )
        ClassMC.maxLikeModels.II.CHI2VEC, ClassMC.maxLikeModels.II.CHI2FLT, ClassMC.maxLikeModels.II.NDOF, ClassMC.maxLikeModels.II.like  = self.chi2likeSingle( ClassMC.maxLikeModels.II, trestrange=trestrange )

        return(None)


    def plotLightCurve(self, ytype='flux', xtype='mjd', bands='all', mjdpk=None,
                       showlegend=False, showpkmjdrange=False, 
                       showsalt2fit=False, showclassfit=False, 
                       showclasstable=True, 
                       filled=False, autozoom=True, 
                       savefig='', verbose=False,  **kwarg ) : 
        """ Plot the observed multi-color light curve data. 
        WFC3-IR filters are plotted as circles with solid lines
        ACS-WFC bands are squares with dashed lines
        WFC3-UVIS filters are shown as triangles with dotted lines
                
          OPTIONS 
        ytype : 'flux', 'mag', 'chi2Ia', 'chi2Ibc', 'chi2II'  
            - Default of 'flux' uses SNANA's FLUXCAL units, ZPT:27.5
            - The 'chi2' options presume an existing maxLikeModel, and plot
              the chi2 contribution from each light curve point 
        xtype : 'mjd', 'tobs', 'trest'  (tobs and trest are days rel. to peak)
        bands : list of SNANA filter IDs or 'all'  (e.g.  bands='HJW')

        showlegend : put a legend in the upper corner
        showpkmjdrange : add a vertical line and bar marking the range of PKMJD
        showsalt2fit : overplot the best-fit SALT2 model (if available)
        showclassfit : 'Ia.maxlike', 'II.maxlike', 'Ibc.maxlike', 'Ia.maxprob', 'II.maxprob', 'Ibc.maxprob' 
            overplot the best-fit model (either max likelihood or max posterior probability) 
            for the given class from a (previously executed) classification simulation. 
        showclasstable : print a table of parameter values and chi2 statistics for the 
            best-fit model on the right side of the figure
        savefig : filename for saving the figure directly to disk (extension sets the filetype)

           (The following options are typically used for plotting finely sampled models, 
            like a SALT2 model fit or a max likelihood model from a classification sim)
        filled : plot semi-transparent filled curves instead of points and connecting lines
        autozoom : True/False to toggle on/off the automatic rescaling 

        Any additional keyword args are passed to the matplotlib.pyplot.plot() function 
          (e.g: ms=10, ls=' '  to plot large markers with no lines)
        """ 
        from matplotlib.patches import FancyArrowPatch
        from constants import IBCMODELS, IIMODELS
        fig = p.gcf()
        ax = fig.gca()

        if showpkmjdrange :
            if ytype=='flux':
                ymax=self.FLUXCAL.max() + 3*self.FLUXCALERR.max()
                ymin=self.FLUXCAL.min() - 3*self.FLUXCALERR.max()
            else : 
                ymax=self.MAG.max() + 3*self.MAGERR.max()
                ymin=self.MAG.min() - 3*self.MAGERR.max()
            if xtype=='mjd' : 
                ax.axvline( self.pkmjd, color='0.5',lw=0.7,ls='--' )
                pkmjdbar = patches.Rectangle( [ self.pkmjd-self.pkmjderr, ymin], 2*self.pkmjderr, ymax+abs(ymin), 
                                              color='0.5', alpha=0.3, zorder=-100 )
                ax.add_patch( pkmjdbar )
            elif xtype=='trest' : 
                ax.axvline( 0.0, color='0.5',lw=0.7,ls='--' )
                pkmjdbar = patches.Rectangle( [ -self.pkmjderr/(1+self.z), ymin], 2*self.pkmjderr/(1+self.z), ymax+abs(ymin), 
                                              color='0.5', alpha=0.3, zorder=-100 )
                ax.add_patch( pkmjdbar )
            elif xtype=='tobs' : 
                ax.axvline( 0.0, color='0.5',lw=0.7,ls='--' )
                pkmjdbar = patches.Rectangle( [ -self.pkmjderr, ymin], 2*self.pkmjderr, ymax+abs(ymin), 
                                              color='0.5', alpha=0.3, zorder=-100 )
                ax.add_patch( pkmjdbar )

        
        if( showsalt2fit ) : 
            if ( 'salt2fitModel' in self.__dict__ and 'salt2fit' in self.__dict__ ) : 
                self.salt2fitModel.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                                   showlegend=False, autozoom=autozoom, 
                                                   showsalt2fit=False, filled=True, marker=' ', ls='-')
               
                ax = p.gca()
                if showclasstable : 
                    ax.text(1,1, "Ia (SALT2)\nz=%.3f\nx1=%.2f\nc=%.2f\nmB=%.2f\nchi2=%.2f/%i=%.2f\np=%.3f"%(
                            self.salt2fit.z,self.salt2fit.x1,self.salt2fit.c,self.salt2fit.mB,self.salt2fit.CHI2,
                            self.salt2fit.NDOF,self.salt2fit.CHI2/self.salt2fit.NDOF,
                            self.salt2fit.FITPROB), transform=ax.transAxes,ha='right',va='top',
                            )#backgroundcolor='w' )
            else : 
                print("No SALT2 fit model available.  Use self.getSALT2fit()" )

        if str(showclassfit).startswith('Ia') :
            if 'ClassMC' in self.__dict__ :  
                bestFitModelIa = self.ClassMC.maxLikeModels.Ia
                PIa = self.ClassMC.PIa
            elif 'ClassSim' in self.__dict__ : 
                if showclassfit.endswith('like'): bestFitModelIa = self.maxLikeIaModel
                else : bestFitModelIa = self.maxProbIaModel
                PIa = self.PIa
            else :  bestFitModelIa = None               
            if bestFitModelIa == None : 
                print("The max likelihood Ia model is not available.")
                print("Run doGridClassify or doClassifyMC + getMaxLikeModelsMC" )
            else : 
                bestFitModelIa.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                               showlegend=False, autozoom=autozoom, 
                                               showsalt2fit=False, showclassfit=False,
                                               filled=True, marker=' ')
                if self.ClassSim.Ia.USELUMPRIOR==0 : magoffsetString = '$\Delta$m=%.1f\n'%bestFitModelIa.MAGOFF 
                else : magoffsetString = ''
                ax = p.gca()
                if showclasstable :                             
                    toptxt = "P(Ia$|$D)=%.2f\nP(Ibc$|$D)=%.2f\nP(II$|$D)=%.2f\nz$_{SN}$=%.3f$\pm$%.3f"%(
                        self.PIa, self.PIbc, self.PII, self.zsnphot, self.zsnphoterr)
                    bottxt = "Best-fit Model:\n%s(%s)\nMJD$_{pk}$=%i\nz=%.3f\n$x_1$=%.2f\n$\mathcal{C}$=%.1f\n$\\beta$=%.1f\n%s$\chi^2_{\\nu}$=%.1f"%(
                        bestFitModelIa.TYPE, 'SALT2', 
                        round(bestFitModelIa.PEAKMJD),bestFitModelIa.REDSHIFT,
                        bestFitModelIa.LUMIPAR,bestFitModelIa.COLORPAR,bestFitModelIa.COLORLAW,
                        magoffsetString,  bestFitModelIa.chi2_ndof)
                    ax.text(0.95, 0.9, toptxt,
                            transform=ax.transAxes,ha='right',va='top' )
                    ax.text(0.95, 0.7, bottxt,
                            transform=ax.transAxes,ha='right',va='top' )
                mjdpk = bestFitModelIa.mjdpk

        elif str(showclassfit).startswith('Ibc'):
            if 'ClassMC' in self.__dict__ :  
                bestFitModelIbc = self.ClassMC.maxLikeModels.Ibc
                PIbc = self.ClassMC.PIbc
            elif 'ClassSim' in self.__dict__ : 
                if str(showclassfit).endswith('like'): bestFitModelIbc = self.maxLikeIbcModel
                else : bestFitModelIbc = self.maxProbIbcModel
                PIbc = self.PIbc
            else :  bestFitModelIbc = None               
            if bestFitModelIbc == None : 
                print("The max likelihood Ibc model is not available.")
                print("Run doGridClassify or doClassifyMC + getMaxLikeModelsMC" )
            else : 
                bestFitModelIbc.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                                showlegend=False, autozoom=autozoom, 
                                                showsalt2fit=False, showclassfit=False,
                                                filled=True, marker=' ')
                ax = p.gca()
                if showclasstable : 
                    toptxt = "P(Ia$|$D)=%.2f\nP(Ibc$|$D)=%.2f\nP(II$|$D)=%.2f\nz$_{SN}$=%.3f$\pm$%.3f"%(
                        self.PIa, self.PIbc, self.PII, self.zsnphot, self.zsnphoterr)
                    bottxt = "Best-fit Model:\n%s(%s)\nMJD$_{pk}$=%i\nz=%.3f\n$A_V$=%.2f\n$R_V$=%.1f\n$\Delta$m=%.1f\n$\chi^{2}_{\\nu}$=%.1f"%(
                            bestFitModelIbc.TYPE, bestFitModelIbc.TEMPLATE.replace('_','').replace('+',''), 
                            round(bestFitModelIbc.PEAKMJD),bestFitModelIbc.REDSHIFT,
                            bestFitModelIbc.COLORPAR,bestFitModelIbc.COLORLAW,
                            bestFitModelIbc.MAGOFF, bestFitModelIbc.chi2_ndof)
                    ax.text(0.95, 0.9, toptxt,
                            transform=ax.transAxes,ha='right',va='top' )
                    ax.text(0.95, 0.7, bottxt,
                            transform=ax.transAxes,ha='right',va='top' )
                mjdpk = bestFitModelIbc.mjdpk
        elif str(showclassfit).startswith('II') :
            if 'ClassMC' in self.__dict__ :  
                bestFitModelII = self.ClassMC.maxLikeModels.II
                PII = self.ClassMC.PII
            elif 'ClassSim' in self.__dict__ : 
                if str(showclassfit).endswith('like'): bestFitModelII = self.maxLikeIIModel
                else : bestFitModelII = self.maxProbIIModel
                PII = self.PII
            else :  bestFitModelII = None               
            if bestFitModelII == None : 
                print("The max likelihood II model is not available.")
                print("Run doGridClassify or doClassifyMC + getMaxLikeModelsMC" )
            else : 
                bestFitModelII.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                               showlegend=False, autozoom=autozoom, 
                                               showsalt2fit=False, showclassfit=False,
                                               filled=True, marker=' ' )
                ax = p.gca()
                fig = p.gcf()
                if showclasstable :
                    toptxt = "P(Ia$|$D)=%.2f\nP(Ibc$|$D)=%.2f\nP(II$|$D)=%.2f\nz$_{SN}$=%.3f$\pm$%.3f"%(
                        self.PIa, self.PIbc, self.PII, self.zsnphot, self.zsnphoterr)
                    bottxt = "Best-fit Model:\n%s(%s)\nMJD$_{pk}$=%i\nz=%.3f\n$A_V$=%.2f\n$R_V$=%.1f\n$\Delta$m=%.1f\n$\chi^{2}_{\\nu}$=%.1f"%(
                            bestFitModelII.TYPE, bestFitModelII.TEMPLATE.replace('_','').replace('+',''), 
                            round(bestFitModelII.PEAKMJD),bestFitModelII.REDSHIFT,
                            bestFitModelII.COLORPAR,bestFitModelII.COLORLAW,
                            bestFitModelII.MAGOFF, bestFitModelII.chi2_ndof)
                    ax.text(0.95, 0.9, toptxt,
                            transform=ax.transAxes,ha='right',va='top' )
                    ax.text(0.95, 0.7, bottxt,
                            transform=ax.transAxes,ha='right',va='top' )

                mjdpk = bestFitModelII.mjdpk

        if bands=='all' : bands = ''.join(self.bandlist)
        magmin,magmax=27,22
        fluxmin,fluxmax=0,2
        for band in reversed(self.BANDORDER) : 
            if band not in bands : continue
            iband = np.where( self.FLT == band )[ 0 ]
            mag = self.MAG[ iband ] 
            magerr = self.MAGERR[ iband ] 
            flux = self.FLUXCAL[ iband ] 
            fluxerr = self.FLUXCALERR[ iband ] 
            mjd = self.MJD[ iband ] 
            camera = self.SURVEYDATA.band2camera( band )

            if mjdpk==None : mjdpk = self.pkmjd
            if xtype=='tobs' : x = mjd - mjdpk
            elif xtype=='trest' : x = ( mjd - mjdpk ) / (1+self.z)
            else : x = mjd
                       
            if ytype=='flux' : 
                y = flux
                yerr = fluxerr
                yUL = []
                if y.max() > fluxmax and y.max() != 99 : fluxmax = y.max()
                if y.min() < fluxmin and y.min() > -90 : fluxmin = y.min()
            elif ytype.startswith('chi2'): 
                if ytype.endswith('Ia'): maxLikeModel = self.maxLikeIaModel
                elif ytype.endswith('Ibc'): maxLikeModel = self.maxLikeIbcModel
                elif ytype.endswith('II'): maxLikeModel = self.maxLikeIIModel
                ibandchi2 = np.where( maxLikeModel.CHI2FLT == band )[0]
                if not len(ibandchi2) : continue
                y = maxLikeModel.CHI2VEC[ ibandchi2 ]
                yerr = np.zeros( len(y) )
                yUL = []
                mjd = maxLikeModel.CHI2MJD[ ibandchi2 ]
                if xtype=='tobs' : x = mjd - mjdpk
                elif xtype=='trest' : x = ( mjd - mjdpk ) / (1+self.z)
                else : x = mjd
            else : 
                iUpperLim = np.where( magerr<-8 )[0]
                iGoodMags = np.where( (magerr>-8) & (mag<32) )[0]
                y = mag[ iGoodMags ]
                yerr = magerr[ iGoodMags ]
                yUL = mag[ iUpperLim ]
                xUL = x[  iUpperLim ]
                x = x[ iGoodMags ]
                if len(y) : 
                    if y.max() > magmax : magmax = min(y.max(),28)
                    if y.min() < magmin : magmin = max(y.min(),13)

            lstyle,alpha = '-',1
            # if camera=='IR': lstyle='-'
            # elif camera=='ACS': lstyle='--'
            # elif camera=='UVIS': lstyle=':'
            if showsalt2fit or showclassfit : lstyle=' '
            if showlegend: label = self.SURVEYDATA.band2filter(band)
            else : label='_nolegend_'

            defaultarg = { 'label':label, 'ls':lstyle, 'marker':'o','alpha':alpha,'color':'k' }
            fillarg = {'alpha': 0.3}
            if 'BANDCOLOR' in self.SURVEYDATA.__dict__ : 
                defaultarg['color'] = self.SURVEYDATA.BANDCOLOR[band]
                fillarg['color'] = self.SURVEYDATA.BANDCOLOR[band]
            if 'BANDMARKER' in self.SURVEYDATA.__dict__ : 
                defaultarg['marker'] = self.SURVEYDATA.BANDMARKER[band] 
            plotarg = dict(defaultarg.items()+kwarg.items())
            if plotarg['marker']==' ': ax.plot( x, y, **plotarg )
            if filled : 
                if ytype.startswith('mag'): 
                    yerr = np.where( (yerr<8) & (yerr>-8), yerr, np.abs(yerr).min() )
                ax.fill_between( x, y-yerr, y+yerr, **fillarg)
            else : 
                if len(y) : 
                    ax.errorbar( x, y, yerr, **plotarg )
                if len(yUL) : 
                    if not len(y) : 
                        plotarg['marker']='_'
                        ax.plot( xUL, yUL, **plotarg )
                    for xx,yy in zip( xUL, yUL ):
                        plotarg['marker']='_'
                        plotarg['label']='_nolegend_'
                        plotarg['mew'] = p.rcParams['lines.linewidth'] * 1.5
                        ax.plot( xx, yy, **plotarg )
                        arr = FancyArrowPatch( [xx,yy], [xx,yy+0.75], arrowstyle='->', mutation_scale=15, ls='solid', fc=plotarg['color'], ec=plotarg['color'] )
                        ax.add_patch( arr )             
                        #color = self.SURVEYDATA.BANDCOLOR[band]
                        #ax.arrow( xx, yy,  0.0, 0.75, fc=color, ec="k",
                        #          head_width=2., head_length=0.1  )

        if ytype=='mag': 
            ax.set_ylabel( 'Vega mag' )
            if not autozoom : ax.set_ylim( magmax+0.1, magmin-0.1) 
            if not ax.yaxis_inverted() : ax.invert_yaxis()
        elif ytype=='flux':
            ax.set_ylabel( 'Flux' )
            if not autozoom : ax.set_ylim( fluxmin*0.9, fluxmax*1.1) 
        elif ytype.startswith('chi2'):
            ax.set_ylabel( r'$\chi^2$', rotation='horizontal' )

        if xtype=='mjd' : 
            ax.set_xlabel( 'obs frame time (MJD)' )
            if self.mjdpk > 0 and not autozoom : 
                ax.set_xlim( self.mjdpk - 20 * (1+self.z), self.mjdpk + 60 * (1+self.z) )
        elif xtype=='trest' : 
            ax.set_xlabel( 'rest frame time (days)' )
            ax.set_xlim( -20, 60 )
        elif xtype=='tobs' : 
            ax.set_xlabel( 'obs frame time (days rel. to peak)' )
        if showlegend : 
            leg = ax.legend( loc='upper left', frameon=False, numpoints=1, handlelength=0.2, handletextpad=0.4, labelspacing=0.2 )

        if savefig : p.savefig( savefig )
        p.draw()
    
   
    def plotColorMagAll( self, mjd='peak', Nsim=2000, classfractions='mid', dustmodel='mid',
                         showpia=True, linelevels=[0.95,0.68,0], clobber=False, verbose=False, **kwargs ):
        """ Make a composite color-mag plot with all available color vs mag combinations.
        mjd : the MJD observation date to plot.  Use 'peak' to plot the peak observed date, 
              use 'all' to plot every observed epoch in a separate figure.

        Nsim : number of SNe to simulate for each class (if a new sim is needed or clobbering is on)

        classfractions : the fraction of SNe in each of the 3 main sub-classes [ Ia, Ib/c, II ]
             e.g.  classfractions = [0.24,0.57,0.19]  (the z=0 fractions measured by Li+ 2011)
             Special options 'max','mid','min' : 
                use a redshift-dependent prior based on simple models for the CCSN and SNIa 
                rate vs z. The first uses the baseline rate models, the latter 2 use alternate
                rate models that ~minimize/maximize the fraction assigned to Type Ia at any z.
                   See hstsnpipe/tools/rates/priors.py for details. 

        dustmodel : distribution of host extinction ['high','mid','low']

        showpia : if False, don't show any SNIa classification probability numbers.
       
        Other keyword arguments are passed on to the self.plotColorMag function, and trickle 
        down to override defaults in the simplot.plotSimClouds function   e.g. 
           Nbins : the number of grid-steps along each axis for binning to make contours
           binrange : the data values of grid-limits for binning:  [ [xlow,xhigh], [ylow,yhigh] ]
           tsample : the size of the time-step (in obs-frame days) for sampling photometry
              from the simulated SN light curves
        """
        if mjd =='all' : 
            from matplotlib import pyplot as pl
            # define the epoch dates and use a recursive loop to cycle through all epochs
            epochlist = clusterfloats( self.MJD, dmax=8 )
            ifig = 0
            for mjd in epochlist : 
                ifig += 1
                pl.figure( ifig ) 
                self.plotColorMagAll( mjd=mjd, Nsim=Nsim, classfractions=classfractions, dustmodel=dustmodel,
                                      clobber=clobber, verbose=verbose, **kwargs )
                clobber=False  # no need to re-do the sims each time

        if mjd=='peak' : mjd = self.pkmjdobs
        imjd = np.where( np.abs(self.MJD - mjd) < 8 )[0]
        bandlist = self.FLT[imjd]

        Nbands = len(np.unique( bandlist ) )
        Nrow, Ncol = Nbands-1, Nbands-1
        axlist, plist,likelist = [], [], []
        irow = 0
        bluest = ''
        for band2 in self.BANDORDER : 
            if band2 not in bandlist: continue
            if not bluest : 
                bluest = band2
                continue
            irow += 1
            icol = 0
            for band1 in self.BANDORDER : 
                if band1 not in bandlist : continue
                ib1 = self.BANDORDER.find( band1 ) 
                ib2 = self.BANDORDER.find( band2 ) 
                if ib2 <= ib1 : continue
                icol += 1
                color = band1+'-'+band2
                mag = band2
                if icol == 1 : ax1 = p.subplot( Nrow, Ncol, (irow-1)*Ncol+icol)
                else : ax = p.subplot( Nrow, Ncol, (irow-1)*Ncol+icol, sharey=ax1)
                ax,pia,likes = self.plotColorMag( color, mag, mjd=mjd, Nsim=Nsim, 
                                                  classfractions=classfractions, dustmodel=dustmodel,
                                                  linelevels=linelevels, label=False, 
                                                  sidehist=False, clobber=clobber, verbose=verbose, **kwargs )
                ax = p.gca()
                ax.set_ylim( min(28,ax.get_ylim()[0]) , max(22,ax.get_ylim()[1]) )
                
                if showpia and False:   # deprecated !!
                    pcolorclass = self.getColorClassification( color, mag, mjd=mjd, Nsim=Nsim,
                                                               classfractions=classfractions, dustmodel=dustmodel,
                                                               clobber=False, verbose=verbose )
                    pia = pcolorclass[0]
                    if ((Nrow>1) | (Ncol>1)) : 
                        ax.text( 0.95,0.95,'%.2f'%(pia), fontsize='large', ha='right',va='top', backgroundcolor='w',transform=ax.transAxes)
                    
                axlist.append( ax ) 
                plist.append( pia )
                likelist.append( likes )
                clobber=False
        fig = p.gcf()
        fig.subplots_adjust( left=0.08,bottom=0.1,right=0.98,top=0.97,wspace=0.30,hspace=0.30)
        axbg = p.axes( [0.08,0.1, 0.9, 0.88], frameon=False ) 
        #p.setp( axbg.get_xticklabels(), visible=False)
        #p.setp( axbg.get_yticklabels(), visible=False)
        axbg.yaxis.set_visible( False )
        axbg.xaxis.set_visible( False )

        if showpia and False:   # deprecated !! 
            # axbg.text( 0.95, 0.95, '%s\nz=%.3f$\pm$%.3f\nMJD$_{pk}$=%.i$\pm$%i\nMJD$_{obs}$=%.1f\n$\\tilde{P}(Ia)=%.1f$\n$\mathcal{L}_{Ia}$=%.1f'%( # with likelihood
            #axbg.text( 0.95, 0.95, '%s\nz=%.3f$\pm$%.3f\nMJD$_{pk}$=%.i$\pm$%i\nMJD$_{obs}$=%.1f\n$\\tilde{P}(Ia)_{f}=%.1f$\n$\\tilde{P}(Ia)_{m}=%.1f$'%( # with PIa_flux
            axbg.text( 0.95, 0.95, '%s\nz=%.3f$\pm$%.3f\nMJD$_{pk}$=%.i$\pm$%i\nMJD$_{obs}$=%.1f\n$\\tilde{P}(Ia)=%.1f$'%(
                    self.name, self.z,self.zerr,int(self.mjdpk), int(self.mjdpkerr), mjd, round(np.median(plist),1) ), 
                       ha='right',va='top', transform=axbg.transAxes, fontsize='x-large' )
        else : 
            axbg.text( 0.95, 0.95, '%s\nz=%.3f$\pm$%.3f\nMJD$_{pk}$=%.i$\pm$%i\nMJD$_{obs}$=%.1f'%(
                self.name, self.z,self.zerr,int(self.mjdpk), int(self.mjdpkerr), mjd ), 
                   ha='right',va='top', transform=axbg.transAxes, fontsize='x-large' )

        return( axlist )

    def _magColorPlot( self, xaxis='W-H', yaxis='J-H', mjd='peak', Nsim=2000, label='top-right', 
                       classfractions='mid', dustmodel='mid', linelevels=[0.95,0.68,0.0], 
                       showpia=True, clobber=False, verbose=False, debug=False,  **kwargs ):
        """ make mag-mag, color-mag or color-color plot(s) at the given obs date
        xaxis : the color or mag to be plotted along the x axis. e.g. 'W-H' or 'J'
        yaxis : the color or mag to be plotted along the y axis. e.g. 'H' or 'J-H'
        mjd :  the obs-frame date(s) for sampling colors and mags. 
               Provide a scalar or list, or use 'all' for all dates with S/N>3 
               in the reddest band, or use 'peak' for the peak observed mjd 
               (the date with highest observed S/N). 

        classfractions : the fraction of SNe in each of the 3 main sub-classes [ Ia, Ib/c, II ]
             e.g.  classfractions = [0.24,0.57,0.19]  (the z=0 fractions measured by Li+ 2011)
             Special options 'max','mid','min' : 
                use a redshift-dependent prior based on simple models for the CCSN and SNIa 
                rate vs z. The first uses the baseline rate models, the latter 2 use alternate
                rate models that ~minimize/maximize the fraction assigned to Type Ia at any z.
                   See hstsnpipe/tools/rates/priors.py for details. 

        dustmodel : distribution of host extinction ['high','mid','low']

        Other keyword arguments are passed to simplot.plotColorMag 
           e.g.    plotstyle, Nbins, linelevels, classfractions, etc.
        """
        import simplot
        if debug: import pdb; pdb.set_trace()

        # run and/or read in a classification simulation
        self.getClassSim( simroot='%s_colormag'%self.SURVEYDATA.SURVEYNAME, Nsim=Nsim, simpriors=True, dustmodel=dustmodel, clobber=clobber, verbose=verbose ) 

        if xaxis.find('-')>0: band1,band2 = xaxis.split('-')
        else : band1,band2 = xaxis,xaxis
        if yaxis.find('-')>0: band3,band4 = yaxis.split('-')
        else : band3,band4 = yaxis,yaxis

        if mjd=='all' : 
            # find the dates where the signoise is >3 in the 'mag' band
            imjdlist = np.where( (self.signoise>=3) & (self.FLT==band3))[0]
        else : 
            # find all observations within 5 days of the requested mjd(s)
            if mjd=='peak' : mjd = self.pkmjdobs
            if type(mjd)==list : mjd = np.array(mjd)
            imjdlist = np.where( (np.abs(self.MJD - mjd) < self.SURVEYDATA.EPOCHSPAN) & (self.FLT==band1) )[0]
        if not len(imjdlist) : 
            print("No suitable observations found within 5 days of mjd = %.1f"%mjd)
            return([]) 

        ifig = 1
        axlist,plist,likelist = [],[],[]
        for imjd in imjdlist : 
            if len(imjdlist)>1 : fig = p.figure( ifig ) 
            ifig += 1

            # find observations in the color bands that are ~coincident with this observation date
            try : 
                mjdnow = self.MJD[imjd]
                i1 = np.where( ( np.abs(self.MJD - mjdnow)<5 ) & (self.FLT==band1) )[0][0]
                i2 = np.where( ( np.abs(self.MJD - mjdnow)<5 ) & (self.FLT==band2) )[0][0]
                i3 = np.where( ( np.abs(self.MJD - mjdnow)<5 ) & (self.FLT==band3) )[0][0]
                i4 = np.where( ( np.abs(self.MJD - mjdnow)<5 ) & (self.FLT==band4) )[0][0]
            except : 
                from matplotlib import pyplot as pl
                axlist.append( pl.gca() )
                plist.append( 0 )
                likelist.append( 0 )
                continue

            # Set up the snmags dict to plot the observed SN colors and mags
            snmags = {}
            snmags[band1] = self.MAG[i1]
            snmags[band2] = self.MAG[i2] 
            snmags[band3] = self.MAG[i3] 
            snmags[band4] = self.MAG[i4] 
            snmags['d'+band1] = self.MAGERR[i1]
            snmags['d'+band2] = self.MAGERR[i2] 
            snmags['d'+band3] = self.MAGERR[i3]
            snmags['d'+band4] = self.MAGERR[i4] 

            # define the time to sample photometry from for simulated SNe,
            # in observer-frame days rel. to peak
            mjdrange=[mjdnow,mjdnow]

            # TODO : handle mjd range appropriately for each class
            try: 
                ax, likeia  = simplot.plotSimClouds( self.ClassSim.II,  xaxis=xaxis, yaxis=yaxis, snmags=snmags, mjdrange=mjdrange, linelevels=linelevels, **kwargs )
                ax, likeibc = simplot.plotSimClouds( self.ClassSim.Ibc, xaxis=xaxis, yaxis=yaxis, snmags=snmags, mjdrange=mjdrange, linelevels=linelevels, **kwargs )
                ax, likeii  = simplot.plotSimClouds( self.ClassSim.Ia,  xaxis=xaxis, yaxis=yaxis, snmags=snmags, mjdrange=mjdrange, linelevels=linelevels, **kwargs )
            except Exception as e: 
                print( "Warning : exception when plotting" )
                print( e  ) 
                from matplotlib import pyplot as pl
                ax = pl.gca()

            pia = 0
            if label : 
                labeltext = '%s\nz=%.3f$\pm$%.3f\nMJD$_{pk}$=%.i$\pm$%i\nMJD$_{obs}$=%.1f'%(
                    self.name, self.z,self.zerr,int(self.mjdpk), int(self.mjdpkerr), mjdnow )
                if showpia: 
                    pcolorclass  = self.getColorClassification( xaxis=xaxis, yaxis=yaxis, mjd=mjd, Nsim=Nsim, 
                                                                classfractions=classfractions, dustmodel=dustmodel,
                                                                clobber=False, verbose=verbose )
                    pia = pcolorclass[0]
                    # labeltext += '\nP(Ia)$_{f}\sim$%.3f'%piaF
                    labeltext += '\nP(Ia)$\sim$%.3f'%pia
                    if showpia > 1 : 
                        labeltext += '\n$\mathcal{L}_{Ia}$=%.3f'%(likeia) 
                label = str( label )

                ha, va = 'right','top'
                hp, vp = 0.97, 0.97
                if label.find('left') >= 0 : ha,hp = 'left', 0.03
                elif label.find('right') >= 0 : ha,hp = 'right', 0.97
                if label.find('top') >= 0 : va, vp = 'top', 0.97
                elif label.find('bottom') >= 0 : va, vp = 'bottom', 0.03
                ax.text( hp, vp, labeltext, ha=ha,va=va, backgroundcolor='w', transform=ax.transAxes)

            axlist.append(ax)
            plist.append( pia )
            likelist.append( [likeia,likeibc,likeii] )
        if len(axlist)==1 : axlist = axlist[0]
        if len(plist)==1 : 
            plist = plist[0]
            likelist = likelist[0]
        return( axlist, plist, likelist )


    # Convenience functions for making color-mag and color-color plots with _magColorPlot
    def plotColorMag( self, color='W-H', mag='H', mjd='peak', Nsim=3000, 
                      classfractions='mid', dustmodel='mid', linelevels=[0.95,0.68,0.0], 
                      label=True, clobber=False,**kwargs ):
        return( self._magColorPlot( xaxis=color, yaxis=mag, mjd=mjd, Nsim=Nsim, 
                                    classfractions=classfractions, dustmodel=dustmodel, 
                                    linelevels=linelevels, label=label, clobber=clobber,**kwargs ) )

    def plotColorColor( self, color1='widest', color2='J-H', mjd='peak', Nsim=3000, 
                        classfractions='mid', dustmodel='mid', linelevels=[0.99, 0.68, 0.0],
                        label='top-left', showpia=False, clobber=False,**kwargs ):
        __doc__ = self._magColorPlot.__doc__
        for band1 in self.bandorder : 
            if band1 in self.bands : break
        for band2 in reversed(self.bandorder) : 
            if band2 in self.bands : break
        widest = band1+'-'+band2
        if color1 == 'widest' : color1=widest
        if color2 == 'widest' : color2=widest
            
        return( self._magColorPlot( xaxis=color1, yaxis=color2, mjd=mjd, Nsim=Nsim, 
                                    classfractions=classfractions, dustmodel=dustmodel,
                                    linelevels=linelevels, label=label, showpia=showpia, 
                                    clobber=clobber,**kwargs ) )

    setattr( plotColorMag, '__doc__', _magColorPlot.__doc__ )
    setattr( plotColorColor, '__doc__', _magColorPlot.__doc__ )


    def plotColorCurves( self, color='bluest-reddest',
                         tstep=2, Nsim=500, clobber=False, verbose=False):
        """ 
        NOT YET OPERATIONAL
        plot the observed band1-band2 color vs time, 
        overlaid on simulated color curves matching the redshift
        and peak-mjd ranges of this SuperNova. 
        color : a string giving the color to plot, e.g. 'J-H', 'W-H', etc.
                default is 'bluest-reddest' for the widest available color
        tstep : time-step to sample for the simulation
        Nsim  : total number of SNe to simulate 
        """
        print("Color Curve plotting not yet operational.")
        return( None ) 

        ax = p.axes([0.1,0.1,0.85,0.85])

        band1 = color.split('-')[0]
        band2 = color.split('-')[1]

        # set the bands used for color classification
        if band1=='bluest': 
            band1 = [band for band in self.SURVEYDATA.BLUEORDER
                     if band in self.bandlist ][0]
        if band2=='reddest': 
            band2 = [band for band in self.SURVEYDATA.REDORDER
                     if band in self.bandlist ][0]

        # run/read in a photometric classification SNANA simulation 
        self.classsim = getClassSim( self, tstep=tstep, Nsim=Nsim, clobber=clobber, verbose=verbose)
        nobs = self.classsim.NOBS[0]
    
        # TODO :  clean up below to speed up the simulated SN photometry sampling
        #   and/or to limit to the observed dates

        m1,m2,m3 = np.zeros([len(tobs),len(sim[2].__dict__['PTROBS_MIN'])]), np.zeros([len(tobs),len(sim[2].__dict__['PTROBS_MIN'])]), np.zeros([len(tobs),len(sim[2].__dict__['PTROBS_MIN'])])
        histup,histdown = [],[]
        tobsarr = []
        for i,to in zip(np.arange(len(tobs)),tobs):
            sim[2].samplephot(to)
            m1[i,:] = sim[2].__dict__['mag%s%s'%(band1, timestr(to))]
            m2[i,:] = sim[2].__dict__['mag%s%s'%(band2, timestr(to))]
            row=np.where((abs(m1[i,:])!=99) & (abs(m2[i,:])!=99))

            if row[0]!=[]:
                magsub = (m1[i,row]-m2[i,row])[0]
                magsub.sort()
                index1,index2=int(len(magsub)*0.16),int(len(magsub)*0.84)
                histup = histup + [magsub[index2]]
                histdown = histdown + [magsub[index1]] 
                tobsarr = tobsarr + [to]

        ax.fill_between(array(tobsarr)+peakmjd,histdown,histup,alpha=0.4,color='b')

        m1,m2,m3 = np.zeros([len(tobs),len(sim[1].__dict__['PTROBS_MIN'])]), np.zeros([len(tobs),len(sim[1].__dict__['PTROBS_MIN'])]), np.zeros([len(tobs),len(sim[1].__dict__['PTROBS_MIN'])])
        histup,histdown = [],[]
        tobsarr = []
        for i,to in zip(np.arange(len(tobs)),tobs):
            sim[1].samplephot(to)
            m1[i,:] = sim[1].__dict__['mag%s%s'%(band1, timestr(to))]
            m2[i,:] = sim[1].__dict__['mag%s%s'%(band2, timestr(to))]
            row=np.where((abs(m1[i,:])!=99) & (abs(m2[i,:])!=99))

            if row[0]!=[]:
                magsub = (m1[i,row]-m2[i,row])[0]
                magsub.sort()
                index1,index2=int(len(magsub)*0.16),int(len(magsub)*0.84)
                histup = histup + [magsub[index2]]
                histdown = histdown + [magsub[index1]] 
                tobsarr = tobsarr + [to]

        ax.fill_between(array(tobsarr)+peakmjd,histdown,histup,alpha=0.4,color='g')

        m1,m2,m3 = np.zeros([len(tobs),len(sim[0].__dict__['PTROBS_MIN'])]), np.zeros([len(tobs),len(sim[0].__dict__['PTROBS_MIN'])]), np.zeros([len(tobs),len(sim[0].__dict__['PTROBS_MIN'])])
        histup,histdown = [],[]
        tobsarr = []
        for i,to in zip(np.arange(len(tobs)),tobs):
            sim[0].samplephot(to)
            m1[i,:] = sim[0].__dict__['mag%s%s'%(band1, timestr(to))]
            m2[i,:] = sim[0].__dict__['mag%s%s'%(band2, timestr(to))]
            row=np.where((abs(m1[i,:])!=99) & (abs(m2[i,:])!=99))

            if row[0]!=[]:
                magsub = (m1[i,row]-m2[i,row])[0]
                magsub.sort()
                index1,index2=int(len(magsub)*0.16),int(len(magsub)*0.84)
                histup = histup + [magsub[index2]]
                histdown = histdown + [magsub[index1]] 
                tobsarr = tobsarr + [to]

        ax.fill_between(array(tobsarr)+peakmjd,histdown,histup,alpha=0.4,color='r')

        ax.set_ylim([min(mags)-0.5,max(mags)+0.5])
        ax.set_xlim([peakmjd+min(tobs),peakmjd+max(tobs)])

        ax.set_ylabel('%s - %s'%(band1,band2))
        ax.set_xlabel('MJD')
        ax.set_title('SN %s at z = %.2f'%(sn,z))

        from matplotlib.lines import Line2D
        for mag,mage,mj in zip(mags,magerr,mjd):
            l = Line2D([mj,mj],[mag-mage,mag+mage])
            ax.add_line(l)
            l = Line2D([mj,mj],[mag-mage,mag+mage])
            ax.add_line(l)
        ax.errorbar(mjd,mags,yerr=magerr,fmt='o',color='k',capsize=0)

        return


    def plotClassStatsMonteCarlo( self, showpriors=True, clobber=False, verbose=False, **kwargs  ) :
        """ plot a histogram of reduced chi2 values, and then plot the classification probabilities 
        as a function of redshift distance, extinction, and SALT2 parameters.  
        Keyword args are passed on to the doClassify  function if needed.
        """
        # TODO : add 2-d contour plots in z-mB and z-Av and Av-mB spaces
        # TODO : add an option to show priors on top of posterior probability curves

        # check if products from the probability computations are already present,
        # run getClassProb if needed: 
        if( 'ClassMC' not in self.__dict__ ): 
            print("ERROR: no classification products. Run doClassifyMC to get classification probability distributions.")
            return(None)

        # read in the .DUMP data and bin it up if needed
        if 'DUMPBIN' not in self.ClassMC.Ia.__dict__ : 
            self.ClassMC.Ia.bindumpdata()
        if 'DUMPBIN' not in self.ClassMC.Ibc.__dict__ : 
            self.ClassMC.Ibc.bindumpdata()
        if 'DUMPBIN' not in self.ClassMC.II.__dict__ : 
            self.ClassMC.II.bindumpdata()

        fig = p.figure( figsize=(10.5, 8) )
        fig.clf()
        
        simlist = self.ClassMC
        postProblist = [ ClassMC.PROB for ClassMC in self.ClassMC ]
        colorlist = ['r','g','b']        

        Ncol=0
        for isim,sim,postProb,color in zip( [1,2,3], simlist, postProblist, colorlist ) : 
            isCC = isim>1  # Is this a CC sim?

            # count up the parameters : z,x1,c/Av,beta/Rv,m_offset,peakmjd
            paramnamelist,paramveclist = [],[]
            for paramname, paramvec in zip( ['Z','COLORPAR','COLORLAW','LUMIPAR','MAGOFF','PEAKMJD'],
                                            [sim.z, sim.COLORPAR, sim.COLORLAW, sim.LUMIPAR, sim.MAGOFFSET, sim.PEAKMJD ]) :
                if not np.iterable(paramvec) : continue
                if len(paramvec)<=1 : continue
                if isCC and paramname=='LUMIPAR' : continue
                paramrange = paramvec.max()-paramvec.min()
                if paramrange<1e-6 : continue
                paramveclist.append( paramvec )
                paramnamelist.append( paramname )

            # panel 1 : histogram of chi2 values for chi2/nu < 10
            Ncol = len(paramveclist)+2
            iax=(isim-1)*Ncol+1
            ax1 = fig.add_subplot(3,Ncol,iax)
            chi2bins = np.arange(0,10,0.2)
            histChi2, edgeChi2 = np.histogram( sim.CHI2 / sim.NDOF, bins=chi2bins )
            ax1.plot( edgeChi2[:-1],  histChi2,  marker=' ', ls='steps-pre-', color=color)
            ax1.set_xlabel('$\chi^2$/$\\nu$' )
            ax1.text( 0.05, 0.95, 'N($\chi^2$/$\\nu$)', ha='left',va='top', transform=ax1.transAxes, color=color )

            # panel 2 : histogram of likelihood values for the bins contributing 90% of the total likelihood
            iax+=1
            ax2 = fig.add_subplot(3,Ncol,iax)
            ilikesorted = np.array([ i for i in reversed( sim.LIKE.argsort() )])
            likefrac = sim.LIKE[ilikesorted].cumsum() / sim.LIKE.sum()
            if likefrac[0] > 0.9 : i90 = 10
            else : i90 = max(10,np.where( likefrac < 0.9 )[0][-1])
            histLike, edgeLike = np.histogram( sim.LIKE[ilikesorted][:i90], bins=max( int(i90/10.), i90+1) )
            ax2.plot( edgeLike[:-1]/sim.LIKE.sum(),  histLike,  marker=' ', ls='steps-pre-', color=color)
            ax2.set_xlabel('p(D$|$%s) / total'%sim.simname.split('_')[-1])
            ax2.yaxis.set_tick_params( pad=-20 )
            ax2.text( 0.95, 0.95, 'N( p(D$|$%s) )'%sim.simname.split('_')[-1], ha='right',va='top', transform=ax2.transAxes, color=color )
            if isim==1 : ax1.text( 0.05, 1.05, '%s (%s)   P(Ia,Ib/c,II)= [ %.1f, %.1f, %.1f ]'%(self.name,self.nickname,self.ClassMC.PIa,self.ClassMC.PIbc,self.ClassMC.PII),
                                   ha='left', va='bottom', fontsize='x-large', transform=ax1.transAxes )
        

            for paramvec,paramname in zip(paramveclist,paramnamelist):
                # Define the label text and prior functions
                if paramname=='REDSHIFT' : partext, xlabel,prior = 'z', 'redshift',sim.PRIORz
                elif paramname=='Z' : partext, xlabel,prior = 'z', 'redshift',sim.PRIORz
                elif paramname=='COLORPAR' :
                    if isCC : partext, xlabel, prior = 'Av', 'Host Extinction',sim.PRIORAv
                    else : partext, xlabel, prior = 'c', 'SALT2 color', sim.PRIORc
                elif paramname=='COLORLAW' : 
                    if isCC : partext, xlabel, prior = 'Rv', 'Extinction Law', sim.PRIORRv
                    else : partext, xlabel, prior = r'$\beta$', r'SALT2 $\beta$', sim.PRIORBeta
                elif paramname=='LUMIPAR': partext, xlabel, prior = 'x1', 'SALT2 x1', sim.PRIORx1
                elif paramname=='PEAKMJD': partext, xlabel, prior = 'MJD$_{pk}$', 'Peak MJD', lambda mjd : np.ones(len(mjd))
                elif paramname=='MAGOFF': partext, xlabel,prior = 'm$_{offset}$', 'mag offset', sim.PRIORmagoffset

                # sort the posterior probability values into bins according to the parameter of interest,
                # then sum up each bin to get the marginalized posterior probability
                parambins = np.linspace(paramvec.min(),paramvec.max(),int(sim.nsim**(1./len(paramveclist))) )
                ibins = np.digitize( paramvec, bins=parambins )
                postProbBinned = np.array([ postProb[np.where(ibins==ithisbin)].sum() for ithisbin in range(len(parambins)) ])
                iax+=1 
                ax = fig.add_subplot(3,Ncol,iax)
                ax.plot( parambins, postProbBinned, ls='steps-mid-', color=color,zorder=10 )
                ax.text( 0.05, 0.95, 'p(%s$|$D)'%partext, ha='left',va='top', transform=ax.transAxes, color=color )
                ax.set_xlabel( xlabel )
                ax.set_yticklabels( [] )
                if showpriors: 
                    priorBinned = prior(parambins)
                    priorBinned = priorBinned * postProbBinned.max() / priorBinned.max()
                    ax.plot( parambins, priorBinned, ls='steps-mid--', color='k',zorder=20 )
                    ax.text( 0.05, 0.85, 'p(%s)'%partext, ha='left',va='top', transform=ax.transAxes, color='k' )

        fig.subplots_adjust( left=0.08, right=0.95,bottom=0.1, top=0.92, wspace=0.12, hspace=0.2 )
        p.draw()

        return( 0 )


    def plotClassStatsGrid( self, showpriors=True, savefig='', clobber=False, verbose=False, debug=False, interactive=True, **kwargs  ) :
        """ plot a histogram of reduced chi2 values, and then plot the classification probabilities 
        as a function of redshift distance, extinction, and SALT2 parameters.  
        Keyword args are passed on to the doClassify  function if needed.

        NOTE: this is currently a bit broken if you run doGridClassify with useLuminosityPrior=2.
        """
        from matplotlib import ticker
        if debug : import pdb; pdb.set_trace()
        # TODO : add 2-d contour plots in z-mB and z-Av and Av-mB spaces
        
        if not interactive : 
            p.ioff()

        # check if products from the probability computations are already present,
        if( clobber or 'PIa' not in self.__dict__  
            or 'PIbc' not in self.__dict__ 
            or 'PII' not in self.__dict__ ): 
            if verbose: print("ERROR: No classification products. Run doGridClassify to get classification probability distributions.")
            return(None)

        fig = p.figure( figsize=(11, 8.5) )
        fig.clf()

        simlist = self.ClassSim
        postProblist = [ self.postProbIa, self.postProbIbc, self.postProbII ] 

        from hstsnpipe.tools.figs import colors
        colorlist = [colors.red,colors.green,colors.blue]        

        for isim,sim,postProb,color in zip( [1,2,3], simlist, postProblist, colorlist ) : 
            isCC = isim>1  # Is this a CC sim?

            Npar = ((len(sim.z)>1) + (len(sim.COLORPAR)>1) + (len(sim.COLORLAW)>1) + 
                    (len(sim.LUMIPAR)>1) + (len(sim.PKMJD)>1) + (len(sim.LUMIPAR)>1) )
            Ncol = Npar+2
            iax=(isim-1)*Ncol+1

            # panel 1 : histogram of chi2 values for chi2/nu < 10
            ax1 = fig.add_subplot(3,Ncol,iax)
            chi2bins = np.arange(0,10,0.2)
            histChi2, edgeChi2 = np.histogram( sim.CHI2 / sim.NDOF, bins=chi2bins )
            ax1.plot( edgeChi2[:-1],  histChi2,  marker=' ', ls='steps-pre-', color=color, lw=1.5 )
            ax1.set_xlabel('$\chi^2$/$\\nu$' )
            ax1.text( 0.05, 0.95, 'N($\chi^2$/$\\nu$)', ha='left',va='top', transform=ax1.transAxes, color=color)
            ax1.locator_params(axis = 'x', nbins = 4)
            ax1.locator_params(axis = 'y', nbins = 8)
            ax1.set_xlim( -0.1, 10.2 )
            ax1.xaxis.set_major_locator( ticker.MultipleLocator( 2 ) )
            ax1.xaxis.set_minor_locator( ticker.MultipleLocator( 1 ) )
            ax1.set_xticks( [1,3,5,7,9] )

            # panel 2 : histogram of likelihood values for the bins contributing 90% of the total likelihood
            iax+=1
            ax2 = fig.add_subplot(3,Ncol,iax)
            ilikesorted = np.array([ i for i in reversed( sim.LIKE.argsort() )])
            likefrac = sim.LIKE[ilikesorted].cumsum() / sim.LIKE.sum()
            if likefrac[0] > 0.9 : i90 = 10
            else : i90 = max(10,np.where( likefrac < 0.9 )[0][-1])
            histLike, edgeLike = np.histogram( sim.LIKE[ilikesorted][:i90], bins=max( int(i90/10.), i90+1) )
            ax2.plot( edgeLike[:-1]/sim.LIKE.sum(),  histLike,  marker=' ', ls='steps-pre-', color=color, lw=1.5)
            ax2.set_xlabel('p(D$|$%s) / total'%sim.simname.split('_')[-1])
            ax2.yaxis.set_tick_params( pad=-20 )
            ax2.text( 0.05, 0.95, 'N( p(D$|$%s) )'%sim.simname.split('_')[-1], ha='left',va='top', transform=ax2.transAxes, color=color )
            if isim==1 : ax1.text( 0.05, 1.05, '%s (%s)   P(Ia,Ib/c,II)= [ %.1f, %.1f, %.1f ]'%(self.name,self.nickname,self.PIa,self.PIbc,self.PII),
                                   ha='left', va='bottom', fontsize='x-large', transform=ax1.transAxes )
            ax2.yaxis.set_ticks_position('right')
            ax2.yaxis.set_ticks_position('both')
            ax2.locator_params(axis = 'x', nbins = 4)
            ax2.locator_params(axis = 'y', nbins = 4)
            ax2.set_xlim( -0.1*ax2.get_xlim()[1], ax2.get_xlim()[1] )

            # plot priors and posteriors marginalized down to a single axis, for all relevant grid 
            # dimensions (i.e. we skip over any axis with length 1) 
            if sim.NMAGOFFSET>1 : 
                ppgrid = postProb.reshape( sim.NMAGOFFSET, len(sim.PKMJD),  len(sim.LUMIPAR), 
                                            len(sim.COLORLAW), len(sim.COLORPAR), len(sim.LOGZ))
            else : 
                ppgrid = postProb.reshape( len(sim.PKMJD),  len(sim.LUMIPAR), 
                                           len(sim.COLORLAW), len(sim.COLORPAR), len(sim.LOGZ))


            paramveclist = [sim.z, sim.COLORPAR, sim.COLORLAW, sim.LUMIPAR, sim.PKMJD-self.mjdpk, sim.MAGOFFSET ]
            for iparam, paramvec in zip([0,1,2,3,4,5], paramveclist ) : 
                if len(paramvec)<=1 : continue
                # if isCC and iparam==3 : continue
                if sim.USELUMPRIOR and not isCC and iparam==5 : continue
                iax+=1 

                # Define the parameter name and x-axis title strings, 
                # and get the prior values
                if iparam==0 : 
                    parname, xlabel = 'z', 'redshift'
                    prior = sim.zPriorVec
                    xlim = [ self.z-1.2*self.zerr, self.z+1.2*self.zerr]
                    xtmaj = np.round(3*self.zerr/4.,3)
                    xtmin = np.round(xtmaj/2., 4 )
                elif iparam==1 : 
                    if isCC : 
                        parname, xlabel = 'Av', 'Host Extinction'
                        prior = sim.AvPriorVec
                        dx = (sim.AV.max()-sim.AV.min())/len(sim.AV)
                        xlim = [ sim.AV.min()-dx, sim.AV.max()+dx ]
                        xtmaj,xtmin = 2,1
                    else : 
                        parname, xlabel = 'c', 'SALT2 color'
                        prior = sim.cPriorVec
                        dx = (sim.c.max()-sim.c.min())/len(sim.c)
                        xlim = [ sim.c.min()-dx, sim.c.max()+dx ]
                        xlim = [ -0.7, 1.2 ]
                        xtmaj,xtmin = 0.5,0.25
                elif iparam==2 : 
                    if isCC : 
                        parname, xlabel = 'Rv', 'Extinction Law'
                        prior = sim.RvPriorVec
                        xlim=None
                    else : 
                        parname, xlabel = r'$\beta$', r'SALT2 $\beta$'
                        prior = sim.BetaPriorVec
                        xlim=None
                elif iparam==3 and 'x1PriorVec' in sim.__dict__ : 
                    parname, xlabel = 'x1', 'SALT2 x1'
                    prior = sim.x1PriorVec
                    dx = (sim.x1.max()-sim.x1.min())/float(len(sim.x1))
                    xlim = [ -3.6, 3.8 ]
                    xtmaj,xtmin = 2,1
                elif iparam==3 :
                    if 'Ib' in sim.SIM_SUBTYPE : 
                        parname, xlabel = 'tmp', 'Ib/c template'
                    else : 
                        parname, xlabel = 'tmp', 'II template'
                    Ntemp = len( sim.LUMIPAR )
                    prior = np.ones( Ntemp )
                    paramvec = np.arange( Ntemp )
                    xlim= [-1.5, Ntemp+1.5 ]
                    xtmaj,xtmin = 5,1
                elif iparam==4 : 
                    parname, xlabel = 't$_{pk}$', '$\Delta$ MJD peak'
                    prior = np.ones( len(sim.PKMJD) )
                    xlim = None 
                elif iparam==5 : 
                    parname, xlabel = '$\Delta$m', 'mag offset'  
                    xlim=None

                    # The mag offset parameter is a Special case : 
                    # If sim.USELUMPRIOR == 2 : we have sampled the lum.functions at NMAGOFF steps
                    #      and each sub-type has its own lum.function, so the magoffset steps are different
                    #    sim.USELUMPRIOR == 1 : we have assigned the optimal flux scaling for CCSN and applied a prior. 
                    #    sim.USELUMPRIOR == False : assigned the optimal flux scaling but applied no luminosity prior
                    # In each of these cases, we do not have no pre-defined grid of magoffset values, 
                    # so we bin up and plot histograms 
                    paramvec = np.arange(-3,3.1,0.2)
                    ibinsMagOff = np.digitize( sim.MAGOFFSET, bins=paramvec )
                    ppgridMarginalized = np.array([ sim.postProb[np.where(ibinsMagOff==ithisbin)].sum() 
                                                    for ithisbin in range(len(paramvec)) ])
                    prior = np.ma.masked_invalid( np.array([ sim.MAGOFFSETPRIOR[np.where(ibinsMagOff==ithisbin)].mean() 
                                                             for ithisbin in range(len(paramvec)) ]) ).filled( 0 )
                        
                if iparam != 5 : 
                    # For any other parameter (z, x1, c, MJDpk...), we simply
                    # marginalize over all other parameters to isolate the one of interest
                    if sim.USELUMPRIOR>1 and isCC : Naxes = 6 
                    else : Naxes = 5
                    axes = range(Naxes)
                    axes.remove( Naxes-1-iparam )
                    ppgridMarginalized = ppgrid
                    for igax in reversed(axes) : 
                        ppgridMarginalized = ppgridMarginalized.sum( axis=igax )

                ax = fig.add_subplot(3,Ncol,iax)
                postProb1d = ppgridMarginalized / ppgridMarginalized.max()
                isorted = paramvec.argsort()

                ax.plot( paramvec[isorted], postProb1d[isorted], ls='steps-mid-', color=color,zorder=15, lw=1.5 )
                if showpriors: 
                    ax.plot( paramvec[isorted], prior[isorted] / prior.max() , ls='steps-mid--', color='k',zorder=20, lw=0.7 )

                ax.text( 0.05, 0.95, 'p(%s$|$D)'%parname, ha='left',va='top', transform=ax.transAxes, color=color )
                ax.text( 0.98, 0.95, 'p(%s)'%parname, ha='right',va='top', transform=ax.transAxes, color='k' )
                ax.set_xlabel( xlabel )
                ax.set_yticklabels( [] )
                ax.set_ylim( 0, 1.2 )
                if xlim : 
                    ax.set_xlim( xlim )
                    if xtmaj>0: ax.xaxis.set_major_locator( ticker.MultipleLocator( xtmaj ) )
                    if xtmin>0: ax.xaxis.set_minor_locator( ticker.MultipleLocator( xtmin ) )
                else : 
                    ax.locator_params(axis = 'x', nbins = 4)
                ax.yaxis.set_major_locator( ticker.MultipleLocator( 0.5 ) )
                ax.yaxis.set_minor_locator( ticker.MultipleLocator( 0.25 ) )

            # ax.set_yticklabels( [0.5, 1.0] )
            ax.yaxis.set_label_position('right')
            if 'Ia' in sim.SIM_SUBTYPE : ax.set_ylabel( 'Type Ia', rotation=-90, color=colors.red )
            elif 'Ibc' in sim.SIM_SUBTYPE : ax.set_ylabel( 'Type Ib/c', rotation=-90, color=colors.green )
            elif 'IIP' in sim.SIM_SUBTYPE : ax.set_ylabel( 'Type II', rotation=-90, color=colors.blue )

        fig.subplots_adjust( left=0.04, right=0.95,bottom=0.08, top=0.92, wspace=0.0, hspace=0.3 )
        p.draw()

        if savefig : p.savefig( savefig )
        return( 0 )

    def plotMaxLikeModels( self, ytype='flux', xtype='mjd', showchi2=False, usegrid=True, clobber=False, verbose=False, **kwargs ) :
        """ plot the max likelihood light curves as filled contours
        showchi2 : alongside each light curve plot, show the contribution to the total 
           chi2 for each point
        additional keyword args are passed on to plotLightCurve and thence to the 'plot()'
        command (e.g. marker='o', ls=' ', lw=2 )
        """
        self.plotBestFitModels( ytype=ytype, xtype=xtype, maximize='likelihood', showchi2=showchi2, usegrid=usegrid, clobber=clobber, verbose=verbose, **kwargs )

    def plotMaxProbModels( self, ytype='flux', xtype='mjd', clobber=False, verbose=False, **kwargs ) :
        """ plot the max posterior probability light curves as filled contours
        showchi2 : alongside each light curve plot, show the contribution to the total 
           chi2 for each point
        additional keyword args are passed on to plotLightCurve and thence to the 'plot()'
        command (e.g. marker='o', ls=' ', lw=2 )

        NOTE: showchi2 is not an option for the max-probability models.
        """
        self.plotBestFitModels( ytype=ytype, xtype=xtype, maximize='probability', showchi2=False, usegrid=True, clobber=clobber, verbose=verbose, **kwargs )

    def plotBestFitModels( self, ytype='flux', xtype='mjd', maximize='likelihood', showchi2=False, usegrid=True, clobber=False, verbose=False, **kwargs ) :
        """ plot the max likelihood light curves as filled contours
        showchi2 : alongside each light curve plot, show the contribution to the total 
           chi2 for each point
        additional keyword args are passed on to plotLightCurve and thence to the 'plot()'
        command (e.g. marker='o', ls=' ', lw=2 )
        """
        fig = p.gcf()
        fig.clf()
        fignum = fig.number

        if 'maxLikeIaModel' not in self.__dict__ and 'ClassMC' not in self.__dict__: 
            print("No max likelihood models are defined.  Run .doGridClassify or doClassifyMC + getMaxLikeModelsMC")
            return(None)

        if showchi2 : ax1 = fig.add_subplot(321)
        else : ax1 = fig.add_subplot(311)
        self.plotLightCurve( ytype, xtype, showclassfit='Ia.%s'%maximize[:4], **kwargs )
        ax1.text(0.05,0.95, '%s\n max %s models\nP(Ia,Ibc,II)=%.2f,%.2f,%.2f'%(self.nickname,maximize[:4],self.PIa, self.PIbc, self.PII), transform=ax1.transAxes, ha='left',va='top' )

        if showchi2 : ax2 = fig.add_subplot(323, sharex=ax1, sharey=ax1)
        else: ax2 = fig.add_subplot(312, sharex=ax1, sharey=ax1)
        self.plotLightCurve( ytype, xtype, showclassfit='Ibc.%s'%maximize[:4], showlegend=True, **kwargs )

        if showchi2 : ax3 = fig.add_subplot(325, sharex=ax1, sharey=ax1)
        else : ax3 = fig.add_subplot(313, sharex=ax1, sharey=ax1)
        self.plotLightCurve( ytype, xtype, showclassfit='II.%s'%maximize[:4], **kwargs )
        p.setp( ax1.get_xticklabels(), visible=False )
        p.setp( ax2.get_xticklabels(), visible=False )

        fig.subplots_adjust( left=0.08, bottom=0.08, right=0.98, top=0.98, hspace=0, wspace=0.2 )

        if showchi2 and maximize.startswith('like') : 
            ax4 = fig.add_subplot(322, sharex=ax1 )
            self.plotLightCurve( 'chi2Ia', xtype, ls=' ')#, marker='o' )
            ax5 = fig.add_subplot(324, sharex=ax1 )
            self.plotLightCurve( 'chi2Ibc', xtype, ls=' ')#, marker='o' )
            ax6 = fig.add_subplot(326, sharex=ax1 )
            self.plotLightCurve( 'chi2II', xtype, ls=' ')#, marker='o' )
            p.setp( ax4.get_xticklabels(), visible=False )
            p.setp( ax5.get_xticklabels(), visible=False )
            ax4.yaxis.set_ticks_position('right')
            ax4.yaxis.set_ticks_position('both')
            ax4.yaxis.set_label_position('right')
            ax5.yaxis.set_ticks_position('right')
            ax5.yaxis.set_ticks_position('both')
            ax5.yaxis.set_label_position('right')
            ax6.yaxis.set_ticks_position('right')
            ax6.yaxis.set_ticks_position('both')
            ax6.yaxis.set_label_position('right')
            fig.subplots_adjust( left=0.08, bottom=0.08, right=0.9, top=0.98, hspace=0, wspace=0 )
            return(ax1, ax2, ax3, ax4, ax5, ax6)

        return(ax1, ax2, ax3)

    def chi2likeSingle( self, simsn, bands='all', trestrange=[-15,30], 
                        verbose=True, debug=False):
        """ compute the chi2 statistic and the posterior likelihood from comparison of the 
        light curve for this observed SuperNova against the single synthetic SN lightcurve provided
        in 'simsn'.
        bands      : a string listing the bands to fit, e.g. 'HJW'. Use 'all' for all bands (default)
        trestrange : fit only observations within this rest-frame time window (rel. to peak)  
        """
        if debug : import pdb; pdb.set_trace()

        # make a list of obs indices for observations 
        # that we should include in the calculation
        if bands=='all' : bands = self.bands
        igoodobs = []
        for iobs in range(self.nobs) : 
            if self.FLT[iobs] not in bands : continue
            mjdobs = self.MJD[iobs]
            trest = (mjdobs-simsn.mjdpk)/(1+simsn.z)
            if trest < trestrange[0] : continue
            if trest > trestrange[1] : continue
            igoodobs.append( iobs )
        # extract the usable observed fluxes and errors
        fluxobs = self.FLUXCAL[igoodobs]
        fluxerrobs =self.FLUXCALERR[igoodobs]

        # get a list of indices for the simulated SN observations
        # that most closely match the actual obs dates and filters.
        igoodsim = []
        for iobs in igoodobs : 
            iband = np.where( simsn.FLT == self.FLT[iobs] )[0]
            dist = np.abs(simsn.MJD[iband] - self.MJD[iobs])
            imatch = iband[ dist.argmin() ]
            igoodsim.append( imatch )
        # extract the matching simulated fluxes and errors
        fluxsim = simsn.FLUXCAL[igoodsim]
        fluxerrsim = simsn.FLUXCALERR[igoodsim]
        Ndof = max(1, len( igoodsim ) - 3)

        # chi2vec.append( (fluxobs-fluxsim)**2 / (fluxerrobs**2+fluxerrsim**2) )
        chi2vec = (fluxobs-fluxsim)**2 / (fluxerrobs**2+fluxerrsim**2)
        chi2flt = self.FLT[igoodobs]
        chi2 = np.sum(chi2vec)
        alpha = np.product(1 / np.sqrt(2 * np.pi * (fluxobs**2 + fluxsim**2)))
        likelihood = alpha*np.exp(-chi2/2.)                       
                      
        return( chi2vec, chi2flt, Ndof, likelihood )


    def getChi2LikelihoodMC( self, classname, Nsim=2000, bands='all', trestrange=[-15,30], 
                             modelerror=0.1, errfloor='auto', useLuminosityPrior=True, 
                             verbose=True, clobber=False, debug=False):
        """ Generate arrays containing the chi2 statistic and the posterior probability 
        distribution functions, found by comparison of the light curve for this observed 
        SuperNova against each of the synthetic SN lightcurves provided in the SimTable 
        object 'simset'.  

        classname  :  the SN class to compare against ('Ia', 'Ibc', or 'II')
                     This selects which snana.SimTable object to use (presumably generated 
                     in a classification simulation, so that it contains 2-D phot data arrays 
                     holding all the simulated light curves for that class)
        bands      : a string listing the bands to fit, e.g. 'HJW'. Use 'all' for all bands (default)
        trestrange : fit only observations within this rest-frame time window (rel. to peak)  
        modelerror : fractional flux error to apply to each SN model for chi2 calculation
        errfloor   : minimum flux error for the model (e.g. for zero-flux extrapolations)
                     With the default 'auto' setting, the errfloor is automatically set on a 
                     filter-by-filter basis, using the getErrFloor function.
        useLuminosityPrior  : 
                   if False : allow a free parameter for scaling the flux of each simulated SN so that
                     it most closely matches the observed fluxes (i.e. remove all priors on cosmology
                     and luminosity functions that have been baked in to the simulations) 

        STORED RESULTS : 
          self.chi2<class> : 1-d array of chi2 values (one for each synthetic SN)
          self.like<class> : 1-d array of likelihood values (one for each synthetic SN)

        RETURNS :
          chi2vector  : the 1-d array of chi2 values, one for each comparison model in simtab 
          postProb    : the posterior probability, with one element for each comparison SN model in simtab.
        """
        if debug : import pdb; pdb.set_trace()

        # read in the classification simulation metadata 
        if clobber>2 or 'ClassMC' not in self.__dict__ :
            self.getClassSim( simroot='HST_classifyMC', Nsim=Nsim, objectname='ClassMC',
                              simpriors=False, dustmodel='flat', 
                              verbose=verbose, clobber=clobber)

        # read in the simulated light curves for the class of interest
        if classname=='Ia': 
            if clobber>1 or 'MJD' not in self.ClassMC.Ia.__dict__ : 
                if verbose: print("Reading in %i simulated Type Ia light curves."%self.ClassMC.Ia.nsim)
                self.ClassMC.Ia.getLightCurves( verbose=verbose, clobber=clobber )
            simtab = self.ClassMC.Ia
        elif classname=='Ibc': 
            if clobber>1 or 'MJD' not in self.ClassMC.Ibc.__dict__ : 
                if verbose: print("Reading in %i simulated Type Ib/c light curves."%self.ClassMC.Ibc.nsim)
                self.ClassMC.Ibc.getLightCurves( verbose=verbose, clobber=clobber )
            simtab = self.ClassMC.Ibc
        elif classname=='II' : 
            if clobber>1 or 'MJD' not in self.ClassMC.II.__dict__ : 
                if verbose: print("Reading in %i simulated Type II light curves."%self.ClassMC.II.nsim)
                self.ClassMC.II.getLightCurves( verbose=verbose, clobber=clobber )
            simtab = self.ClassMC.II
        else : 
            raise exceptions.RuntimeError("I don't know how to classify for type %s"%classname)

        # sort the observed and simulated SN light curves by MJD
        isortobs = self.MJD.argsort()
        isortsim = simtab.MJD[0].argsort()
        mjdobs = self.MJD[ isortobs ]
        trestobs = (mjdobs - self.mjdpk)/(1+self.z) 
        fltobs = self.FLT[isortobs]
        fluxobs = self.FLUXCAL[isortobs]
        fluxerrobs = self.FLUXCALERR[isortobs]

        # make a list of obs indices (from the mjd-sorted list) for 
        # observations  that we should include in the calculation
        if bands=='all' : bands = self.bands
        igoodobs = np.where( (trestobs>trestrange[0]) & 
                             (trestobs<trestrange[1]) & 
                             ( np.array([ flt in bands for flt in fltobs ] )) )[0]

        # define matrices of identical shape for the flux and fluxerr from the 
        # observed SN and the simulated Sne. 
        fluxobs  = np.array([ fluxobs[igoodobs].tolist() for i in np.arange( simtab.nsim ) ])
        fluxerrobs  = np.array([ fluxerrobs[igoodobs].tolist() for i in np.arange( simtab.nsim ) ])

        # compute the 2-d chi2 matrix: 
        #  each row compares a single simulated SN to our observed SN,
        #  and each column along the row corresponds to a different observation
        #  date and filter.
        fluxsim = simtab.FLUXCAL[:,isortsim][:,igoodobs]
        if useLuminosityPrior==False  :
            # determine the flux scaling factor A for each synthetic SN that minimizes
            #   the flux difference relative to the observed SN.
            A = optimalFluxScale( fluxsim, fluxobs, fluxerrobs, modelerror ) 
            fluxsim = fluxsim * A
            self.ClassMC.__dict__[classname].FLUXSCALE = A
        else : 
            self.ClassMC.__dict__[classname].FLUXSCALE = 1

        # set the model uncertainty error floor filter by filter:
        self.ClassMC.__dict__[classname].errfloor = errfloor
        if errfloor=='auto': 
            if self.zerr<0.01 : zsim = self.z
            else : zsim = simtab.z
            fltsim = simtab.FLT[0][igoodobs]
            errfloor1 = getErrFloor( zsim, fltsim, classname )
            errfloor = np.array( [ errfloor1 for i in range(simtab.nsim) ] )

        fluxscale = self.ClassMC.__dict__[classname].FLUXSCALE
        if fluxscale >0 : magoffset = -2.5*np.log10(fluxscale)
        else : magoffset = np.nan
        self.ClassMC.__dict__[classname].MAGOFFSET = magoffset
        if modelerror : fluxerrsim = (fluxsim * modelerror + errfloor)
        else : fluxerrsim = simtab.FLUXCALERR[:,isortsim][:,igoodobs] + errfloor
        chi2matrix = ( fluxsim - fluxobs)**2 / ( fluxerrsim**2 + fluxerrobs**2 )

        # reduce the 2-d chi2 matrix to a 1-d vector (summing across each light curve)
        chi2vector = chi2matrix.sum( axis=1 ) 

        # compute the 1-d likelihood vector, assuming gaussian errors, i.e.  e^(-chi2/2)
        likevector = np.exp(-chi2vector/2) / np.sqrt( 2 * np.pi * (fluxerrobs**2 + fluxerrsim**2).sum(axis=1) )

        zrange = simtab.SIM_REDSHIFT.max()-simtab.SIM_REDSHIFT.min()
        cprange = simtab.COLORPAR.max()-simtab.COLORPAR.min()
        clrange = simtab.COLORLAW.max()-simtab.COLORLAW.min()
        lprange = simtab.LUMIPAR.max()-simtab.LUMIPAR.min()
        trange = simtab.SIM_PEAKMJD.max()-simtab.SIM_PEAKMJD.min()
        self.ClassMC.__dict__[classname].NDOF = max(1,len(igoodobs) - (zrange>0.01) - (clrange>1) - (cprange>0.01) - (lprange>0.01) - (trange>1))
        self.ClassMC.__dict__[classname].CHI2 = chi2vector
        self.ClassMC.__dict__[classname].LIKE = likevector
        self.ClassMC.__dict__[classname].FLUXSCALED = fluxsim
        self.ClassMC.__dict__[classname].FLUXERRSCALED = fluxerrsim
        self.ClassMC.__dict__[classname].modelerror = modelerror

        return( None )


class SALT2fit(object):
    """ object class for results from a SNANA SALT2 fit
    as recorded in a .fitres output file """

    def __init__( self, fitresfile ):
        """ initialize the object and read in the fit results """
        self.VARNAMES = []
        self.NSN = 0
        if os.path.isfile( fitresfile ): 
            self.fitsuccess = self.rdfitres( fitresfile )
        else : 
            print("No such .fitres file : %s"%fitresfile )
            self.fitsuccess = False

    def rdfitres( self, fitresfile ): 
        """ read in the SNANA fitting results from a .fitres file """
        fin = open(fitresfile)
        fitreslines = fin.readlines()
        fin.close()

        for line in fitreslines : 
            if not len(line.strip()): continue
            key,valstr = line.strip().split(':')
            if key=='NVAR' : Nvar = int(valstr)
            elif key=='VARNAMES': 
                self.VARNAMES = valstr.split()
                for var in self.VARNAMES : 
                    self.__dict__[var] = []
            elif key=='SN': 
                datlist = []
                ivar = range(len(self.VARNAMES))
                for i,v in zip(ivar, valstr.split()) : 
                    var = self.VARNAMES[i]
                    self.__dict__[var].append( str2num(v) )
                self.NSN += 1

        ivar = range(len(self.VARNAMES))
        try : 
            for v in self.VARNAMES : 
                if len(self.__dict__[v])>1 : 
                    self.__dict__[v] = np.array( self.__dict__[v] )
                else : 
                    self.__dict__[v] = self.__dict__[v][0]
        except IndexError :
            print("Error in SALT2 fitting. Missing/corrupt data in %s"%fitresfile)
            return( False )

        return( True )


    def getsnfitdat( self, CID ):
        """ retrieve the SNANA fit data for a single SN from the
        fitres variable attributes and return it as a dict """
        sndat = {}
        for isn in range(self.NSN): 
            if self.CID[isn] == CID : 
                for var in self.VARNAMES : 
                    sndat[var] = self.__dict__[var][isn]
                break
        return( sndat )

    @property 
    def z( self ):
        if 'Z' in self.__dict__ : 
            return( self.Z )
        elif 'ZSPEC' in self.__dict__ : 
            return( self.ZSPEC )
        elif 'ZHOST' in self.__dict__ : 
            return( self.ZHOST )
        elif 'ZPHOT' in self.__dict__ : 
            return( self.ZPHOT )

    @property 
    def zerr( self ):
        if 'ZERR' in self.__dict__ : 
            return( self.ZERR )
        elif 'ZSPECERR' in self.__dict__ : 
            return( self.ZSPECERR )
        elif 'ZHOSTERR' in self.__dict__ : 
            return( self.ZHOSTERR )
        elif 'ZPHOTERR' in self.__dict__ : 
            return( self.ZPHOTERR )

class SuperNovaSet( Sequence ) :
    """ simple container for holding a set of three SuperNova 
    models, corresponding to each of the primary sub-classes
    (e.g. for holding maximum likelihood models)
       Ia :  the SNIa model
       Ibc : the SNIb/c model
       II  : the SNII model
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


class SurveyData(object): 
    def __init__( self ):
        return( None )

    def filter2band( self, filtername ):
        """ convert a full filter string into its 
        single-digit alphabetic code.  e.g:
          filter2band('F125W')  ==>  'J'
        """
        return( self.FILTER2BAND[ filtername ] ) 

    def filter2alpha( self, filtername ):
        return( self.FILTER2BAND[ filtername ] ) 

    def band2filter( self, band ):
        """ convert a single-digit alphabetic bandpass code 
        into a full filter string. e.g:
          band2filter('J') ==> 'F125W'
        """
        import exceptions
        for filtername in self.FILTER2BAND.keys(): 
            if self.FILTER2BAND[filtername] == band : 
                return( filtername )
        else : 
            raise exceptions.RuntimeError(
                "Unknown filter %"%band )

    def alpha2filter( self, alpha ) : 
        return( self.band2filter(alpha) )

    def band2camera( self, band ) :
        """ convert a single-digit alphabetic bandpass code 
        into a single-letter camera code
          band2cam('J') ==> 'i'
        """
        for camera in self.CAMERAS : 
            if band in self.CAMERABANDLIST[camera] : 
                return( camera ) 
        return('x')

    def band2cam( self, band ) :
        return( self.camera2cam( self.band2camera( band ) ) )

    def filter2cam( self, filtername ) :
        """ convert a full hst filter name
        into a single-letter camera code
          filter2cam('F125W') ==> 'i'
        """
        return( self.band2cam( self.filter2band( filtername ) ) )

    def camera2cam( self, camera ) :
        """ convert a camera string (e.g. from an apt file) 
        into a single-letter camera code
          camera2cam('WFC3/IR') ==> 'i'
        """
        return( camera.lower()[0] ) 

    def zptsnana( self, camera='IR', band='H', etime=1000 ) :
        """ SNANA wants exposure-time-corrected zeropoints for a defined aperture 
        i.e. the mags that produce 1 total count in the aperture, in a given etime.
        This function converts traditional astronomical (etime-independent) 
        zero points into the  SNANA style 
        """
        from math import log10
        if etime==0 : return( np.nan )
        if 'ZPTSNANA' in self.__dict__ : 
            return( self.ZPTSNANA[camera][band] )
        elif self.band2filter(band) in self.ZEROPOINT[camera] : 
            return( self.ZEROPOINT[camera][self.band2filter(band)] - 2.5*log10( 1./etime ) )
        elif band in self.ZEROPOINT[camera] : 
            return( self.ZEROPOINT[camera][band] - 2.5*log10( 1./etime ) )

    def skynoise( self, camera='IR', band='H', etime=1000  ) :
        """ Compute and report the sky-noise in standard SNANA units: 
           total ADU / pix for a given exposure time
        Note: IR noise values include thermal noise.
        TODO : include dark current (culled from ETC?)
        """
        from math import sqrt

        # SHORTCUT for SNLS :
        if 'SKYNOISE' in self.__dict__ : 
            return(self.SKYNOISE[camera][band] )

        # Calculate for HST : 
        skynoise = sqrt( etime * self.SKYCPS[self.band2filter(band)] / self.GAIN[camera] )
        return( skynoise )




            
def str2num(s) :
    """ convert a string to an int or float, as appropriate.
    If neither works, return the string"""
    try: return int(s)
    except exceptions.ValueError:
        try: return float(s)
        except exceptions.ValueError: return( s )


def bifgauss( x, mean, sigmaL, sigmaR ): 
    """ return the value of a bifurcated gaussian distribution 
    at the given abscissa.  The bifurcated gaussian is comprised of
    the left half of a gaussian distribution with width sigmaL and 
    the right half of a gaussian with width sigmaR, both having a 
    common mean value. 
    The two halves are normalized to a value of unity at the join.
    """
    from numpy import sqrt, pi, exp, iterable, array, ndarray, append, where
    #gauss = lambda xval,mu,sig : (1/(sqrt(2*pi)*sig)) * exp(-(xval-mu)**2 / (2*sig**2))
    gauss = lambda xval,mu,sig : exp(-(xval-mu)**2 / (2*sig**2))
    if not iterable( x ) : 
        if x <= mean : return( gauss( x, mean, sigmaL ) )
        else : return( gauss( x, mean, sigmaR ) )

    if type(x) != ndarray : x = array( x )
    #ileft = where(x<=mean) 
    #iright = where(x>mean) 
    #yleft = gauss( x[ileft], mean, sigmaL ) 
    #yright = gauss( x[iright], mean, sigmaR )
    return( where( x<=mean, gauss(x,mean,sigmaL), gauss(x,mean,sigmaR) ) )


def luminosityFunctionFluxScales( sntype, nsteps, nsigma=5 ):
    """  Defines the flux scaling vectors needed to clone a simulated 
    SN into a family of light curves that sample the luminosity function.
    INPUT : 
       sntype : <scalar or 1D list of SN subtypes>
           (e.g.  ['IIP','IIP','IIL','IIN'])
       nsteps : <int> the number of positions to use for sampling
           the luminosity function 
       nsigma : <int> how far out into the wings to sample 
 
    RETURNS :  one numpy.NDarray with shape [ len(sntype), nsteps ].  
    Each row gives an array of flux scaling factors that sample the 
    luminosity function at nsteps positions out to +-nsigma """

    # dictionary of gaussian standard deviations that define the luminosity 
    # function in magnitude space for each SN sub-class
    sigmaMdict = {'Ia' :0.47,
                  'IIP':0.80,
                  'IIN':1.00,
                  'IIL':0.42,
                  'Ib' :0.90,
                  'Ic' :0.60,
                  'Ibc':1.1}

    # Define dictionaries to hold the 1-d array of flux scaling factors 
    # for each possible SN sub-class
    fluxscaledict = {}
    for snclass,sigmaM in sigmaMdict.iteritems() : 
        magoffarray  = np.linspace( -nsigma*sigmaM, nsigma*sigmaM, nsteps )
        fluxscaledict[snclass]  = 10**(0.4*magoffarray) 

    # now construct the 2-D output arrays, which have one row for each 
    # entry in the input sntype array (presumed to be a 1D array or a scalar) 
    if not np.iterable(sntype) : sntype = [ sntype ]
    fluxscalegrid = np.array([  fluxscaledict[t] for t in sntype ])
    return( fluxscalegrid )


def luminosityFunctionPriors( sntype, nsteps, nsigma=5 ):
    """  Defines a luminosity function prior for each SN 
    sub-class entry in the input 'sntype' array.   The prior
    is gaussian in magnitude space (therefore log-normal in 
    flux space)

    INPUT : 
       sntype : <scalar or 1D list of SN subtypes>
           (e.g.  ['IIP','IIP','IIL','IIN'])
       nsteps : <int> the number of positions to use for sampling
           the luminosity function 
       nsigma : <int> how far out into the wings to sample 
 
    RETURNS :  Two numpy.NDarrays with shape [ len(sntype), nsteps ].  
    The first has in each row a vector of prior probabilities sampling 
    the luminosity function at nsteps positions over +-nsigma 
    The second gives the sampling step size in magnitudes.""" 
    
    gauss = lambda x,mu,sig : (1/(np.sqrt(2*np.pi)*sig)) * np.exp(-(x-mu)**2 / (2*sig**2))

    # dictionary of gaussian standard deviations that define the luminosity 
    # function in magnitude space for each SN sub-class
    sigmaMdict = {'Ia' :0.47,
                  'IIP':0.80,
                  'IIN':1.00,
                  'IIL':0.42,
                  'Ib' :0.90,
                  'Ic' :0.60,
                  'Ibc':1.1}

    # Define dictionaries to hold the 1-d array of 
    # prior probabilities for each possible SN sub-class
    priordict,fluxscaledict,dMagoffdict = {},{},{}
    for snclass,sigmaM in sigmaMdict.iteritems() : 
        magoffarray  = np.linspace( -nsigma*sigmaM, nsigma*sigmaM, nsteps )
        fluxscalearray = 10**(0.4*magoffarray) 
        priordict[snclass] = gauss( magoffarray, 0, sigmaM )
        dMagoffdict[snclass] = np.diff( magoffarray )[0]

    # now construct the 2-D output array, which has one row for each 
    # entry in the input sntype array (presumed to be a 1D array or a scalar) 
    if not np.iterable(sntype) : sntype = [ sntype ]
    priorgrid = np.array([  priordict[i] for i in range( len(sntype)/nsteps ) ]).ravel()
    dMagoffgrid = np.array([  dMagoffdict[i] for i in range( len(sntype)/nsteps ) ]).ravel()
    return( priorgrid, dMagoffgrid )



def magOffsetPrior( sntype, istep, nsteps, nsigma=3 ):
    """  Returns the value of the luminosity function prior 
    for the SN sub-class given in 'sntype', at step number 
    'istep' out of 'nsteps' positions that sample the 
    luminosity function over +-nsigma from the mean. 
    The prior is gaussian in magnitude space (therefore 
    log-normal in flux space).

    INPUT : 
       sntype : <str>  name of the  SN subtype (e.g.  'IIP' or 'Ib' )

       istep  : <int> the sampling step number at which to evaluate
           the prior. Sampling starts at istep=0.

       nsteps : <int> the number of positions to use for sampling
           the luminosity function 

       nsigma : <int> the number of standard deviations away from
          the mean to put the end-points of the sampling range.
          (e.g. nsigma=5 means the sampling spans +- 5sigma from
           the mean of the luminosity function) 
 
    RETURNS :  Two scalars giving the value of the  prior probability
    at the given sampling position, and the size of each sampling 
    step (in magnitudes)
    """ 
    gauss = lambda x,mu,sig : (1/(np.sqrt(2*np.pi)*sig)) * np.exp(-(x-mu)**2 / (2*sig**2))

    # dictionary of gaussian standard deviations that define the luminosity 
    # function in magnitude space for each SN sub-class
    sigmaMdict = {'Ia' :0.47,
                  'IIP':0.80,
                  'IIN':1.00,
                  'IIL':0.42,
                  'Ib' :0.90,
                  'Ic' :0.60,
                  'Ibc':1.1}

    sigmaM = sigmaMdict[ sntype ]
    magoffarray= np.linspace( -nsigma*sigmaM, nsigma*sigmaM, nsteps )
    magoffset  = magoffarray[istep]
    dMagoffset = np.diff( magoffarray )[0]
    fluxscalearray = 10**(0.4*magoffset) 
    priorval = gauss( magoffset, 0, sigmaM )
    return( priorval, dMagoffset ) 


def magOffsetFluxScale( sntype, istep, nsteps, nsigma=3 ):
    """  Returns the flux scaling factor that produces the 
    desired magnitude offset for the given SN sub-class 'sntype'
    at step number  'istep' out of 'nsteps' positions that sample the 
    luminosity function over +-'nsigma' from the mean. 

    INPUT : 
       sntype : <str>  name of the  SN subtype (e.g.  'IIP' or 'Ib' )

       istep  : <int> the sampling step number at which to evaluate
           the prior. Sampling starts at istep=0.

       nsteps : <int> the number of positions to use for sampling
           the luminosity function 

       nsigma : <int> the number of standard deviations away from
          the mean to put the end-points of the sampling range.
          (e.g. nsigma=5 means the sampling spans +- 5sigma from
           the mean of the luminosity function) 
 
    RETURNS :  The flux scaling factor that corresponds to the 
     appropriate magnitude offset for this class and step position.
    """ 
    # dictionary of gaussian standard deviations that define the luminosity 
    # function in magnitude space for each SN sub-class
    sigmaMdict = {'Ia' :0.47,
                  'IIP':0.80,
                  'IIN':1.00,
                  'IIL':0.42,
                  'Ib' :0.90,
                  'Ic' :0.60,
                  'Ibc':1.1}
    sigmaM = sigmaMdict[ sntype ]
    magoffarray= np.linspace( -nsigma*sigmaM, nsigma*sigmaM, nsteps )
    fluxscale = 10**(-0.4*magoffarray[istep])
    return( fluxscale ) 


def optimalFluxScale(sim,obs,sigmaobs,modelerror):
    """takes arrays of observed and simulated SN fluxes
    and finds the scaling factor to minimize chi2"""
    # the simulated SN uncertainty, sigmasim, depends on the scaled
    # flux of the model, and therefore on the flux scaling factor we
    # are about to compute. Thus, we first estimate the flux scaling
    # factor using only the current values of sigmaobs and sigmasim,
    # then use that estimated scaling factor to revise sigmasim, 
    # and finally re-calculate the scaling factor.
    sumax = max(0,len(sim.shape)-1)
    sigmasim = sim * modelerror
    num0=np.sum(sim*obs/(sigmaobs**2 + sigmasim**2), axis=sumax)
    denom0=np.sum(sim**2./(sigmaobs**2 + sigmasim**2), axis=sumax)
    if not np.iterable(denom0): 
        if denom0==0 : return(1.)
    else : 
        denom0[np.where(denom0==0)] = 1
    a0 = num0 / denom0
    if np.iterable(a0) : a0 = a0.reshape( len(a0), 1 )

    sigmasim = sim*a0*modelerror
    num=np.sum(sim*obs/(sigmaobs**2 + sigmasim**2), axis=sumax)
    denom=np.sum(sim**2./(sigmaobs**2 + sigmasim**2), axis=sumax)
    a=num/denom
    if np.iterable(denom): 
        denom[np.where(denom==0)] = 1

    if np.iterable(a) : a = a.reshape( len(a0), 1 )
    elif np.isnan( a ) : import pdb; pdb.set_trace()
    return a


def optimalFluxScalePrior( fluxscale, sntype  ):
    """ Returns the value of the luminosity function prior for flux
    scaling factor 'fluxscale', for the SN sub-class given in
    'sntype'.  The prior is drawn from a gaussian probability
    distribution in magnitude space (therefore log-normal in flux
    space).

    INPUT : 
       fluxscale : <float> the flux scaling factor  
       sntype : <str>  name of the  SN subtype (e.g.  'IIP' or 'Ib' )

    RETURNS :  <float> giving The value of the  prior probability
    at the given flux scaling value 
    """ 
    # dictionary of gaussian standard deviations that define the luminosity 
    # function in magnitude space for each SN sub-class
    gauss = lambda x,mu,sig : (1/(np.sqrt(2*np.pi)*sig)) * np.exp(-(x-mu)**2 / (2*sig**2))
    sigmaMdict = {'Ia' :0.15, 
                  'Ib' :0.90,'Ic' :0.60,'Ibc':1.1, 
                  'IIP':0.80,'IIN':1.00,'IIL':0.42, }
    if isinstance( sntype, basestring  ) : 
        sigmaM = sigmaMdict[ sntype ]
    else : 
        sigmaM = np.array([sigmaMdict[ snt ] for snt in np.ravel(sntype) ] )
        sigmaM = sigmaM.reshape( fluxscale.shape ) 
    if fluxscale>0 : 
        magoffset = -2.5*np.log10( fluxscale ) 
        prior = gauss( magoffset, 0, sigmaM )
    else :  
        sigmaF = 10**(-0.4*(sigmaM))
        prior = gauss( fluxscale, 1, sigmaF )
    return( prior )


def avpriorexp( Av, tau=0.7) :
    """ simple exponential prior for host galaxy extinction 
    p(Av) ~ exp(-Av/tau)  
    For negative values of Av, returns unity.
    """
    from numpy import iterable, exp, ndarray, array, ones, where, append
    if not iterable( Av ) : 
        if Av<0 : return(1)
        else : return( exp(-Av/tau) )

    if type(Av) != ndarray : Av = array( Av )

    ileft = where(Av<0)
    iright = where(Av>=0)

    if len(iright) == 1 and iterable(iright[0]) : 
        ileft = ileft[0]
        iright = iright[0] 

    yleft = ones( len(ileft) )
    yright = exp( -Av[iright] / tau ) 
    return( append( yleft, yright ) )

def zpriorFromFile( pdzfile ):
    """Returns a function that evaluates the redshift prior 
    at any given redshift, as defined by a two-column text file 
    that has values of redshift in column 1, and P(z) in column 2. 
    Typically used to set the z prior from a photo-z PDF"""
    from scipy import interpolate
    redshift,Pz = np.loadtxt(pdzfile,unpack=True,usecols=[0,1])
    zprior = interpolate.interp1d( redshift, Pz, bounds_error=False, fill_value=0 )
    return( zprior )


def zpriorFromHost( z, dz=None, z95=None, z68=None ):
    """Returns a function that evaluates the redshift prior 
    at any given redshift, as defined by the 68% and 95% photoz 
    confidence limits, or by a simple uncertainty dz, 
    (typically derived from the host galaxy spec-z or photz

    WARNING : this is a bit kludgy. """
    from scipy import interpolate

    if  z95 : z95min, z95max = z95
    elif z68 : z95min,z95max = 0.5*z68[0],1.5*z68[0],
    elif dz : z95min,z95max = max(0.1*z,z-(2.5*dz)), min(z+(2.5*dz),2.99)

    #if  z68 : z68min, z68max = z68
    #elif z95 : z68min, z68max = 0.5*(z95[0]+z), 0.5*(z95[1]+z)
    #elif dz : z68min, z68max = z-dz, z+dz

    a = z95min
    b = z 
    c = z95max

    A = 0.05/a
    C = 0.05/(15-c)
    B = (1.9-A*(b-a)-C*(c-b)) / (c-a)

    redshift = [ 0, a, b, c, 15 ]
    Pz = [ 0, A, B, C, 0 ]
    zprior = interpolate.interp1d( redshift, Pz, bounds_error=False, fill_value=0 )
    return( zprior )
    
def clusterfloats( floatlist, dmax=5, fmin=None, fmax=None  ) :
    """ Given a list of floats, group into clusters where each member
    is no more than dmax from the mean of the group.  
    Returns a sorted list of the mean float value for each cluster. 

    fmin and fmax provide a min and max threshold: all floats
     that land beyond these thresholds are put together into 
     the first/last grouping.

    (NOTE: this simplistic algorithm is probably not robust 
    for arbitrary lists of numbers, but it is useful for 
    grouping a list of well-spaced MJDs into distinct epochs)
    """
    import numpy as np
    
    floatlist = np.array( floatlist ) 
    isort = floatlist.argsort() 

    groupFloatlists = []
    gotfirstgroup, gotlastgroup = False, False
    for flt in floatlist[isort] : 
        newgroup=True
        if fmin!=None and flt <= fmin : 
            gotfirstgroup = True
            if len(groupFloatlists)==0 : 
                groupFloatlists.append( [ flt ] )
            else : 
                groupFloatlists[0].append( flt )
            continue
        if fmax!=None and flt >= fmax : 
            if not gotlastgroup : 
                gotlastgroup = True
                groupFloatlists.append( [ flt ] )
            else : 
                groupFloatlists[-1].append( flt )
            continue
        for floatlist in groupFloatlists : 
            if abs( np.median(floatlist) - flt) < dmax : 
                floatlist.append( flt ) 
                newgroup=False
                break
        if newgroup : groupFloatlists.append( [flt] )

    # define the date of each group as the mean of its members
    groupMeanFloatlist = np.array( [ np.mean( floatlist ) for floatlist in groupFloatlists ] )  
    groupMeanFloatlist.sort()

    # except for the first/last  one (usually the template epoch) where we define the date 
    # as the max/min date of its members 
    if gotfirstgroup : 
        groupMeanFloatlist[0] = max( groupFloatlists[0] )
    if gotlastgroup : 
        groupMeanFloatlist[-1] = min( groupFloatlists[-1] )

    # assign a group number to each float
    # groupindex = np.array( [ np.argmin( np.abs( flt-groupMeanFloatlist ) ) for flt in floatlist ] )

    return( groupMeanFloatlist )



def getClassFractions( z, IaClassFraczModel='mid', iafractions='snIaFractionPrior.dat') : 
    """ read in the Ia class fraction prior from the specified data table.
    Interpolate to the given value(s) of z. 

    INPUTS : 
       z     : redshifts. may be a scalar or a numpy array 
       IaClassFraczModel : 'high', 'mid', 'low', 'flat'
      
    RETURNS : Three scalars or arrays (matching the input z) giving the fraction 
      of SN in the Ia, Ibc, and II classes, respectively. 

      NOTE: The 'flat' model assumes (unrealistically) that any observed SN is 
        a priori equally likely to  be a Ia or a Ib/c or a II at any z. 
        For this model no interpolation is done and the function 
        simply returns 0.333 for each class at every redshift

    """
    import sys
    import os
    from exceptions import RuntimeError
    from scipy import interpolate as scint
    
    if IaClassFraczModel=='flat' : 
        if np.iterable( z ) : 
            fIa = np.zeros( len(z) ) + 0.333 
            fIbc = np.zeros( len(z) ) + 0.333 
            fII = np.zeros( len(z) ) + 0.333 
        else : 
            fIa = 0.333 
            fIbc = 0.333 
            fII = 0.333 
    else : 
        # locate and read in the data for SNIa fraction vs z
        if not os.path.isfile( iafractions ): 
            thisfile = sys.argv[0]
            if 'ipython' in thisfile : thisfile = __file__
            thispath = os.path.abspath( os.path.dirname( thisfile ) )
            iafractions = os.path.join( thispath, iafractions )
        if not os.path.isfile( iafractions ) :
            raise( RuntimeError( "can't locate %s"%os.path.basename( iafractions ) ) )
        zdat, fIaMin, fIa, fIaMax = np.loadtxt( iafractions, unpack=True )

        # interpolate to get the Ia fraction at the desired z
        if IaClassFraczModel=='high' : 
            interpIa = scint.interp1d( zdat, fIaMax, bounds_error=False, fill_value=fIaMax[-1] ) 
        elif IaClassFraczModel=='low' : 
            interpIa = scint.interp1d( zdat, fIaMin, bounds_error=False, fill_value=fIaMin[-1] ) 
        elif IaClassFraczModel=='mid' : 
            interpIa = scint.interp1d( zdat, fIa, bounds_error=False, fill_value=fIa[-1] ) 
        else : 
            raise exceptions.RuntimeError("IaClassFraczModel must be one of ['high','mid','low']")
        fIa = interpIa( z )
        
        if not np.iterable( fIa ) : fIa = float( fIa )

        # split up the CCSN fraction into II and Ib/c (following Li+ 2011)
        fCC = 1-fIa
        fII = fCC * 0.57/0.76
        fIbc = fCC * 0.19/0.76

    # return the three class fractions as values in a dictionary 
    # keyed by the class name 
    return( fIa, fIbc, fII )


def getErrFloor( surveydatadict, z, flt, snclass='Ia' ):
    """ Define the minimum model uncertainty for the given snclass in
    the obs-frame filter flt, at redshift z.  The uncertainty is
    returned in FLUXCAL units (i.e. flux for ZPT=27.5)
    
    The flt input can be either a scalar or a 1D array giving the list
    of observed filters (the same for each simulated SN).  The z input
    can be a scalar (if all simulated SNe are at the same redshift),
    or a 1-D array listing many redshifts."""
    FLTWAVE = surveydatadict.FLTWAVE

    # minimum fluxcal uncertainty for each sub-class
    f0 = { 'Ia':0.025, 'CC':0.05, 'Ibc':0.05,  'II':0.05, 
           'Ib':0.05, 'Ic':0.05, 
           'IIP':0.05, 'IIL':0.05, 'IIN':0.05, 
           'II-P':0.05, 'II-L':0.05, 'II-N':0.05 }[ snclass ]
    errfloor = lambda w : np.where( w<4800, f0 + f0*( ((w-4800)/1500.) **2 ), f0 + f0*( ((w-4800)/15000.) **2 ) )

    # Get a dict of obs-frame wavelengths for the HST 
    # filters and convert to rest-frame wavelengths
    # TODO : read this info from the SNANA filter files
    if not np.iterable( flt ) : flt = np.array( [flt] )
    fltwaveobs = np.array( [ FLTWAVE[f] for f in flt ] )
    if np.iterable( z ) : 
        fltwaverest = np.array([ fltwaveobs / (1.+zz) for zz in z ])
    else : 
        fltwaverest = fltwaveobs / (1.+z)

    # return the FLUXCAL error floor for each filter at each redshift
    return( errfloor( fltwaverest ) )
    
