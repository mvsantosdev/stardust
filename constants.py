from colors import *

# Factor for converting 1 square arcmin to steradians
arcmin2steradian = 8.461594994075237e-08 # 1 / (60*180/pi)^2 

# DEFAULT COSMOLOGY : Sullivan et al 2011,
OMEGA_MATTER = 0.27  # from SN + WMAP7 + SDSS DR7 LRGs
OMEGA_LAMBDA = 0.73  # assuming flatness
W0_LAMBDA = -1.0  # assumed constant
H0 = 70.0  

# SNANA codes for SN types
SNTYPEDICT = {1:'Ia',10:'Ia',2:'II',3:'Ibc',32:'Ib',33:'Ic',20:'IIP',21:'IIn',22:'IIL' }

# SNANA NON1A template id conversion table
IBCMODELS = {
    '103': ['Ib' ,'CSP-2004gv' ],
    '104': ['Ib' ,'CSP-2006ep' ],
    '105': ['Ib' ,'CSP-2007Y'  ],
    '202': ['Ib' ,'SDSS-000020'],
    '203': ['Ib' ,'SDSS-002744'],
    '212': ['Ib' ,'SDSS-014492'],
    '234': ['Ib' ,'SDSS-019323'],
    '021': ['Ibc','SNLS-04D1la'],
    '022': ['Ic' ,'SNLS-04D4jv'],
    '101': ['Ic' ,'CSP-2004fe' ],
    '102': ['Ic' ,'CSP-2004gq' ],
    '205': ['Ic' ,'SDSS-004012'],
    '207': ['Ic' ,'SDSS-013195'],
    '211': ['Ic' ,'SDSS-014475'],
    '217': ['Ic' ,'SDSS-015475'],
    '218': ['Ic' ,'SDSS-017548'],
    }
       
IIMODELS = {
    '201':['IIP','SDSS-000018'],
    '204':['IIP','SDSS-003818'],
    '208':['IIP','SDSS-013376'],
    '210':['IIP','SDSS-014450'],
    '213':['IIP','SDSS-014599'],
    '214':['IIP','SDSS-015031'],
    '215':['IIP','SDSS-015320'],
    '216':['IIP','SDSS-015339'],
    '219':['IIP','SDSS-017564'],
    '220':['IIP','SDSS-017862'],
    '221':['IIP','SDSS-018109'],
    '222':['IIP','SDSS-018297'],
    '223':['IIP','SDSS-018408'],
    '224':['IIP','SDSS-018441'],
    '225':['IIP','SDSS-018457'],
    '226':['IIP','SDSS-018590'],
    '227':['IIP','SDSS-018596'],
    '228':['IIP','SDSS-018700'],
    '229':['IIP','SDSS-018713'],
    '230':['IIP','SDSS-018734'],
    '231':['IIP','SDSS-018793'],
    '232':['IIP','SDSS-018834'],
    '233':['IIP','SDSS-018892'],
    '235':['IIP','SDSS-020038'],
    '206':['IIN','SDSS-012842'],
    '209':['IIN','SDSS-013449'],
    '002':['IIL','Nugent+ScolnicIIL'],
    }

CCMODELS = dict( IBCMODELS.items() + IIMODELS.items() )


# Convert the SIM_NON1a model codes into human-readable sub-type strings.
SUBTYPES = {
    0:'Ia',
    103:'Ib' ,104:'Ib' ,105:'Ib' ,202:'Ib' ,203:'Ib' ,212:'Ib' ,234:'Ib' ,
     21:'Ibc', 22:'Ic' ,101:'Ic' ,102:'Ic' ,205:'Ic' ,207:'Ic' ,211:'Ic' ,217:'Ic' ,218:'Ic' ,
    201:'IIP',204:'IIP',208:'IIP',210:'IIP',213:'IIP',214:'IIP',215:'IIP',216:'IIP',219:'IIP',
    220:'IIP',221:'IIP',222:'IIP',223:'IIP',224:'IIP',225:'IIP',226:'IIP',227:'IIP',228:'IIP',
    229:'IIP',230:'IIP',231:'IIP',232:'IIP',233:'IIP',235:'IIP',206:'IIN',209:'IIN',  2:'IIL', }

SUBTYPEDICT = SUBTYPES



# ============================================================
# SURVEY DEFINITIONS
from __init__ import SurveyData
from copy import deepcopy
    
HST = SurveyData()
HST.SURVEYNAME='HST'
HST.FIELDNAME = 'default'
HST.KCORFILE = 'HST/kcor_HST_AB.fits' # no k-corrections. Usable with SALT2 and NON1A only
HST.MAGREF = 'AB'
HST.IAMODEL = 'SALT2.Guy10_UV2IR' # or mlsc2k2 or snoopy, etc.
HST.NONIAMODEL = 'NON1A'   # NON1A or non1a
HST.CAMERAS = ['ACS','UVIS','IR']  # Full-length camera names
HST.CAMERABANDLIST={'ACS':'BGVRXIZ','UVIS':'DSTUWC78','IR':'YMJNHLOPQEF'}
HST.ALLBANDS = 'DSTUWC78BGVRXIZYMJNHLOPQEF'
HST.BANDORDER = HST.ALLBANDS 
HST.BANDMARKER = {
    'S':'^','T':'^','U':'^','C':'^','W':'o','7':'1','8':'2',
    'B':'<','G':'h','V':'<','R':'^','X':'o','I':'>','Z':'s',
    'Y':'o','M':'v','J':'s','N':'H','H':'D',
    'E':'o','F':'o',
    'L':'1','O':'2','P':'3','Q':'4' }

HST.BANDCOLOR =  {
    # color-blind barrier free pallette from 
    #    http://people.apache.org/~crossley/cud/cud.html
    'S':'DarkMagenta','T':'DarkOrchid','U':'DeepPink','C':'HotPink',
    '7':'Teal','8':'DarkOrange',
    'B':'DarkSlateBlue','G':'ForestGreen','R':maroon,'X':khaki,
    'E':'DarkCyan','F':'DarkRed','M':khaki, 
    'L':'DarkTurquoise','O':'DarkSlateGrey','P':'SeaGreen','Q':'Brown',
    'V':lightblue,'I':teal,'Z':pink,
    'W':'k','J':green,'H':maroon,
    'Y':darkblue,'N':olivegreen, }
HST.DARKCURRENT={'ACS':0.006,'UVIS':0.0005,'IR':0.05}# e- /pix /sec
HST.CAMERABANDLIST = {'ACS':['B','G','V','R','X','I','Z'], 
                      'IR':['H','J','Y','M','N','L','O','P','Q','E','F'],
                      'UVIS':['D','S','T','U','W','C','7','8'] }
HST.FILTER2BAND = { 'F218W':'D','F225W':'S','F275W':'T',#'F300X':'E',
                    'F336W':'U','F390W':'C','F350LP':'W',
                    'F763M':'7','F845M':'8',
                    'F435W':'B','F475W':'G','F606W':'V','F625W':'R',#'F555W':'F',
                    'F775W':'X','F814W':'I','F850LP':'Z',
                    'F125W':'J','F160W':'H','F125W+F160W':'A',
                    'F105W':'Y','F110W':'M','F140W':'N',
                    'N110W':'E','N160W':'F', # NICMOS
                    'F098M':'L','F127M':'O','F139M':'P','F153M':'Q',
                    'G141':'4','G102':'2','blank':'0'
                    }
HST.SKYCPS = {'F218W':0.0005,'F225W':0.0066,'F275W':0.0037,'F336W':0.0018,
              'F350LP':0.1077,'F390W':0.0098,
              'F850LP':0.039,'F775W':0.078,'F625W':0.083,'F606W':0.127,
              'F555W':0.054,'F475W':0.057,'F435W':0.030,'F814W':0.102,
              'F098M':0.6106,'F105W':1.0150,'F110W':1.6611,'F125W':1.112,
              'F127M':0.2697,'F139M':0.2391,'F140W':1.1694,
              'F153M':0.2361,'F160W':0.943, 
              'N110W':1.6611,'N160W':0.943, }
# Detector "Gain" (really Inverse Gain) for each camera, in e- / ADU
HST.GAIN = {'a':2.0, 'u':1.6, 'i':2.5, 
            'ACS':2.0, 'UVIS':1.6, 'IR':2.5 }
# From instrument handbooks, read noise per pixel:
# HST.RDNOISE = {'ACS':4.8,'UVIS':3.2,'IR':17}   # e-/pix 
# Read noise in 0.4" aperture (from ETC)
HST.RDNOISE = {'ACS':60.22,'UVIS':75.96,'IR':114.28}   # e- 
HST.PSF_FWHM_ARCSEC = { 'ACS': 0.13,'UVIS':0.07, 'IR':0.15 } # PSF FWHM in arcsec
HST.PIXSCALE = { 'ACS': 0.05, 'UVIS':0.04, 'IR':0.13, 'default':0.13 } # Native pixel scale in arcsec per pixel
# NOTE : NIC2 F160W and F110W zero points below were computed following the DHB
# http://www.stsci.edu/hst/nicmos/documents/handbooks/DataHandbookv8/nic_ch5.9.3.html#328526
# Alternate NIC2 Vegamag zeropoints of unknown provenance (possibly from A.Riess? )
#    'N110W':22.77  ,'N160W':21.95 },
HST.ZEROPOINT_AB = {  # infinite  aperture, AB mags
    'UVIS':{'F350LP':26.9435,'F218W':22.9641,'F225W':24.0403,
            'F275W':24.1305,'F336W':24.6682,'F390W':23.3562,
            'F689M':24.4641,'F763M':24.2070,'F845M':23.7811 },
    'IR':{'F105W':26.2687,'F110W':26.8223,'F125W':26.2303,
          'F140W':26.4524,'F160W':25.9463,'F098M':25.6674,
          'F127M':24.6412,'F139M':24.4793,'F153M':24.4635,
          'F126N':22.8609,'F128N':22.9726,'F130N':22.9900,
          'F132N':22.9472,'F164N':22.9089,'F167N':22.9568,
          'N110W':23.69195,'N160W':23.462779 },
    'ACS':{'F435W':25.665,'F475W':26.056,'F550M':24.857,
           'F555W':25.711,'F606W':26.493,'F625W':25.899,
           'F775W':25.662,'F814W':25.947,'F850LP':24.857}
    }
HST.ZEROPOINT_VEGA = {  # infinite  aperture, Vega mags
    'ACS':{'F435W':25.76695,'F475W':26.16252,'F555W':25.72747,'F606W':26.40598,
           'F625W':25.74339,'F775W':25.27728,'F814W':25.51994,'F850LP':24.3230},
    'IR':{'F105W':25.6236,'F110W':26.0628,'F125W':25.3293,
          'F140W':25.3761,'F160W':24.6949,'F098M':25.1057,
          'F127M':23.6799,'F139M':23.4006,'F153M':23.2098,
          'F126N':21.9396,'F128N':21.9355,'F130N':22.0138,
          'F132N':21.9499,'F164N':21.5239,'F167N':21.5948,
          'N110W':22.9643,'N160W':22.15325 },
    'UVIS':{'F350LP':26.7874,'F218W':21.2743,'F225W':22.3808,
            'F275W':22.6322,'F336W':23.4836,'F390W':25.1413,
            'F689M':24.1873,'F763M':23.8283,'F845M':23.2809 }
    }
if HST.MAGREF=='AB' or HST.KCORFILE.endswith('AB.fits') :
    HST.ZEROPOINT = HST.ZEROPOINT_AB
else :
    HST.ZEROPOINT = HST.ZEROPOINT_VEGA
HST.SOLIDANGLE = {  # Field size in square arcmin converted to steradians
    'gnd':77.54*arcmin2steradian,'gnw':(38.7 + 41.4)*arcmin2steradian,
    'gsd':66.5*arcmin2steradian, 'gsw':39.4*arcmin2steradian,
    'cos':196.8*arcmin2steradian,'uds':207.1*arcmin2steradian,
    'egs':199.1 * arcmin2steradian,'default':0.000000393 * 100 }
HST.COORD = { # approximate coordinates of field centers 
    'gnd':{'RA':189.228621,'DEC':62.238572,},
    'gnw':{'RA':188.228621,'DEC':61.238572,}, # exaggerated offset from deep
    'gna':{'RA':189.228621,'DEC':62.238572,},
    'gsd':{'RA':53.122751,'DEC':-27.805089,},
    'gsw':{'RA':52.122751,'DEC':-26.805089,}, # exaggerated offset from deep
    'gsa':{'RA':53.122751,'DEC':-27.805089,},
    'cos':{'RA':150.116321,'DEC':2.2009731,},
    'egs':{'RA':214.825000,'DEC':52.825000,},
    'uds':{'RA':34.40650,'DEC':-5.200000,},
    'default':{'RA':0.0,'DEC':0.0,},
    }
HST.EPOCHLIST = {  # MJD of search epochs, including first epoch (i.e. including decliners)
    # NOTE: gnw and egs have two independent halves. We treat them as a single search, and 
    #  so only record the first pair of epochs here.
    'gnd':[56020,56073,56126,56183,56238,56297,56348,56402,56458,56511],
    'gnw':[56020,56073],
    'gna':[56017,56070,56123,56175,56230,56288,56343,56393,56450,56510],
    'gsd':[55480,55528,55578,55624,55722,55774,55821,55860,55921,55974],
    'gsw':[55573,55621],
    'gsa':[55480,55528,55578,55624,55647,55722,55774,55821,55860,55921,55974,55573,55621],
    'cos':[55905,55953],
    'egs':[56384,56435],
    'uds':[55512,55562], 
    'default':[]}
HST.EPOCHSPAN = 5 # width of an obs epoch, in days 
HST.SEARCHBANDS = ['J','H'] # primary search filters
HST.SEARCHETIMES = [1000,1200] # exposure times for each search band
HST.SEARCHEFF_PIPELINE_FILE = 'SEARCHEFF_PIPELINE_CANDELS.DAT' # in $SNDATA_ROOT/models/searcheff/

# HST Filter order, blue to red and red to blue 
HST.BLUEORDER = 'STUCBGVRXIZWLYEMJOPNQHF'
HST.REDORDER  = 'FHQNPOJMEYLWZIXRVGBCUTS'
HST.CAMERABANDLIST={'ACS':'BGVRXIZ','UVIS':'DSTUWC78','IR':'YEMJNHFLOPQ'}

# central wavelengths for HST Filters (Angstroms)
HST.FLTWAVE = { 'S':2250, 'T':2750, 'U':3360, 'C':3900,
                'W':3500, '7':7630, '8':8450,
                'B':4350, 'G':4750, 'V':6060, 'R':6250,
                'X':7750, 'I':8140, 'Z':8500,
                'J':12500,'H':16000,
                'Y':10500,'M':11000,'N':14000,
                'L':9800,'O':12700,'P':13900, 'Q':15300,
                'E':11000,'F':16000,
                }

#============================================================

CANDELS = deepcopy( HST ) 
CANDELS.SURVEYNAME='CANDELS'
CANDELS.BANDMARKER.update( {
    'S':'^','T':'^','U':'^','C':'^','W':'o',
    'B':'s','G':'s','R':'s','X':'o',
    'V':'<','I':'>','Z':'^',
    'Y':'p','M':'o','J':'s','N':'H','H':'D',
    'E':'o','F':'o',
    'L':'1','O':'2','P':'3','Q':'4' } )
CANDELS.BANDCOLOR.update({
    # color-blind barrier free pallette from 
    #    http://people.apache.org/~crossley/cud/cud.html
    'S':'DarkMagenta','T':'DarkOrchid','U':'DeepPink','C':'HotPink',
    'B':'DarkSlateBlue','G':'ForestGreen','R':'FireBrick','X':'RoyalBlue',
    'E':'DarkCyan','F':'DarkRed','M':'DarkBlue', 
    'L':'DarkTurquoise','O':'DarkSlateGrey','P':'SeaGreen','Q':'Brown',
    'V':'#009e73','I':'#cc79a7','Z':'#e69f00',
    'W':'k','J':'#0072b2','H':'#d55e00',
    'Y':'#56b4e9','N':'#009e73', })

#============================================================

FRONTIER = deepcopy( HST )
FRONTIER.SURVEYNAME='FRONTIER'
FRONTIER.FIELDNAME = 'WFC3-CORE'
FRONTIER.BANDCOLOR.update({
    'S':'DarkMagenta','T':'DarkOrchid','U':'DeepPink','C':'HotPink','W':'k',
    'B':'DarkSlateBlue','G':'ForestGreen','V':'Teal','R':'FireBrick',
    'X':'RoyalBlue','I':'Orange','Z':'Red',
    'Y':'DarkCyan','M':'DarkBlue','J':'Chocolate','N':'LimeGreen','H':'MediumOrchid',
    'E':'DarkCyan','F':'DarkRed',
    'L':'DarkTurquoise','O':'DarkSlateGrey','P':'SeaGreen','Q':'Brown' })
FRONTIER.BANDMARKER.update({
    'S':'^','T':'^','U':'^','C':'^','W':'^',
    'B':'s','G':'s','V':'s','R':'s',
    'X':'o','I':'s','Z':'s',
    'Y':'o','M':'o','J':'o','N':'>','H':'d',
    'E':'o','F':'o',
    'L':'d','O':'d','P':'d','Q':'d' })
FRONTIER.SOLIDANGLE = {  # Field size in square arcmin converted to steradians
    'WFC3-CORE': 6 * 4.6 * 8.461595e-08,
    'ACS-WINGS': 6 * 6.4 * 8.461595e-08,
    'default': 6 * 4.6 * 8.461595e-08,
    }
FRONTIER.COORD = { # approximate coordinates of field centers 
    'WFC3-CORE':{'RA':0.0,'DEC':0.0,},
    'ACS-WINGS':{'RA':0.0,'DEC':0.0,},
    'default':{'RA':0.0,'DEC':0.0,},
    }
FRONTIER.EPOCHLIST = {  # MJD of search epochs, including first epoch (i.e. including decliners)
    # NOTE: gnw and egs have two independent halves. We treat them as a single search, and 
    #  so only record the first pair of epochs here.
    'WFC3-CORE':[ 56600, 56600, 56600, # epoch 01 : Nov 04, 2013 : START Orient A: ACS on cluster
                  56610, 56610, 56610, 
                  56620, 56620, 56620, 
                  56630, 56630, 56630, 
                  56640, 56640, 56640, # epoch 5 : Dec 14, 2013 : END Orient A (40 day window)
                  56780, 56780, 56780, 56780,  # epoch 6: May 04, 2014 : START Orient B: WFC3 on cluster
                  56790, 56790, 56790, 56790,  
                  56800, 56800, 56800, 56800,  
                  56810, 56810, 56810, 56810,  
                  56820, 56820, 56820, 56820,  # epoch 10 : Jun 12, 2014 : END Orient B (40 day window)
                  ],
    'ACS-WINGS':[ 56600, 56600, 56600, # epoch 01 : Nov 04, 2013 : START Orient A: ACS on cluster
                  56610, 56610, 56610, 
                  56620, 56620, 56620, 
                  56630, 56630, 56630, 
                  56640, 56640, 56640, # epoch 5 : Dec 14, 2013 : END Orient A (40 day window)
                  56780, 56780, 56780, 56780,  # epoch 6: May 04, 2014 : START Orient B: WFC3 on cluster
                  56790, 56790, 56790, 56790,  
                  56800, 56800, 56800, 56800,  
                  56810, 56810, 56810, 56810,  
                  56820, 56820, 56820, 56820,  # epoch 10 : Jun 12, 2014 : END Orient B (40 day window)
                  ], 
    'default':[] }
FRONTIER.EPOCHSPAN = 2 # width of an obs epoch, in days 
FRONTIER.SEARCHBANDS = ['B','V','I','J','N','H'] # primary search filters
FRONTIER.SEARCHETIMES = [10000,10000,10000,10000,10000,10000] # exposure times for each search band
FRONTIER.SEARCHEFF_PIPELINE_FILE = 'SEARCHEFF_PIPELINE_FRONTIER.DAT' # in $SNDATA_ROOT/models/searcheff/


# ============================================================



SNLS = SurveyData()
SNLS.SURVEYNAME = 'SNLS'
SNLS.FIELDNAME = 'default'
SNLS.TELESCOPE = 'CFHT'
SNLS.CAMERAS = ['MEGACAM',]
SNLS.FILTERS = 'griz'
SNLS.KCORFILE = 'SNLS3year/kcor_EFFMEGACAM_BD17.fits'
SNLS.MAGREF = 'BD17'
SNLS.IAMODEL = 'SALT2.Guy10_UV2IR'
SNLS.NONIAMODEL = 'NON1A'   
SNLS.ALLBANDS = 'griz'
SNLS.CAMERAS = ['MEGACAM',]
SNLS.CAMERABANDLIST = {'MEGACAM':'griz'}
SNLS.BANDORDER = SNLS.ALLBANDS
SNLS.FILTER2BAND = {'g':'g', 'r':'r','i':'i','z':'z'}
SNLS.COORD = {'default':{'RA':0.0,'DEC':0.0,},}
SNLS.SOLIDANGLE = {'default':1,} # placeholder for SNLS tile size in steradians
SNLS.PIXSCALE = {'MEGACAM':0.187, 'default':0.187}
SNLS.ZEROPOINT = {'MEGACAM':{'g':27.1, 'r':26.325, 'i':25.934, 'z':24.854 },}
SNLS.ZPTSNANA = SNLS.ZEROPOINT
SNLS.SKYNOISE = { 'MEGACAM':{'g':19.0,  'r':25.8, 'i':46.7, 'z':45.9 }}
SNLS.GAIN = {'MEGACAM':1.62 }
# median MEGACAM readnoise across the mosaic
#SNLS.RDNOISE = {'MEGACAM':4.2}  # e- / pixel 
SNLS.RDNOISE = {'MEGACAM':60.4}  # total e- noise in a 0.4" aperture
SNLS.EPOCHSPAN = 2 # width of an obs epoch, in days 
SNLS.SEARCHEFF_PIPELINE_FILE = 'SEARCHEFF_PIPELINE_SNLS.DAT' # in $SNDATA_ROOT/models/searcheff/


# SNLS Filter order, blue to red and red to blue 
SNLS.BLUEORDER = 'griz'
SNLS.REDORDER  = 'zirg'
SNLS.CAMERABANDLIST={'MEGACAM':'griz'}

# central wavelengths for SNLS Filters (Angstroms)
SNLS.FLTWAVE = { 'u':3360, 'g':4750, 'r':6250, 'i':8140, 'z':8500 }


#============================================================

DES = SurveyData()
DES.SURVEYNAME = 'DES'
DES.FIELDNAME = 'default'
DES.TELESCOPE = 'CTIO'
DES.CAMERAS = ['DECAM',]
DES.FILTERS = 'grizY'
DES.KCORFILE = 'DES/kcor_DES_grizY.fits'
DES.MAGREF = 'AB'
DES.IAMODEL = 'SALT2.Guy10_UV2IR'
DES.NONIAMODEL = 'NON1A'   
DES.ALLBANDS = 'grizY'
DES.CAMERAS = ['DECAM',]
DES.CAMERABANDLIST = {'DECAM':'griz'}
DES.BANDORDER = DES.ALLBANDS
DES.FILTER2BAND = {'g':'g', 'r':'r','i':'i','z':'z','Y':'Y'}
DES.COORD = {'default':{'RA':0.0,'DEC':0.0,},}
DES.SOLIDANGLE = {'default':1,} # placeholder for DES tile size in steradians
DES.PIXSCALE = {'DECAM':0.187, 'default':0.187}
DES.ZEROPOINT = {'DECAM':{'g':27.5, 'r':27.5, 'i':27.5, 'z':27.5, 'Y':27.5 },}
DES.ZPTSNANA = DES.ZEROPOINT
DES.SKYNOISE = { 'DECAM':{'g':0.0,  'r':0.0, 'i':0.0, 'z':0.0 }}
DES.GAIN = {'DECAM':1.0 }
DES.RDNOISE = {'DECAM':0.0}  # total e- noise in a 0.4" aperture
DES.EPOCHSPAN = 2 # width of an obs epoch, in days 
DES.SEARCHEFF_PIPELINE_FILE = 'SEARCHEFF_PIPELINE_DES.DAT' # in $SNDATA_ROOT/models/searcheff/
# DES Filter order, blue to red and red to blue 
DES.BLUEORDER = 'grizY'
DES.REDORDER  = 'Yzirg'
DES.CAMERABANDLIST={'DECAM':'grizY'}
# central wavelengths for DES Filters (Angstroms)
DES.FLTWAVE = { 'g':4750, 'r':6250, 'i':8140, 'z':8500, 'Y':10500 }
DES.BANDMARKER = {'g':'^','r':'s','i':'D','z':'o','Y':'H'}
DES.BANDCOLOR =  {
    'g':green,'r':red,'i':blue,
    'z':magenta, 'Y':'k' }



# ==================================================

# This dictionary of SurveyData objects keyed by survey name
# is used by __init__.py to define SuperNova and SimTable properties
SurveyDataDict = { 'HST':HST,'CANDELS':CANDELS,'CLASH':HST,'FRONTIER':FRONTIER,
                   'SNLS':SNLS, 'SNLS3':SNLS, 'SNLS3year':SNLS, 'DES':DES }


# ==================================================
