# 2013.10.31
# S.Rodney
# an implementation of the galsnid host-galaxy-based classification 
# approach (Foley & Mandel 2013) for use with STARDUST

def classify( morphology=None, sedtype=None, 
              priorIa=0.24, priorIbc=0.19, priorII=0.57, 
              returnErrors=False ):
    """ 
    Compute three values giving the (posterior) probability that 
    a SN is of type Ia, Ibc, or II, for a host galaxy observed 
    with the given morphology and SED type. 

    INPUTS:
      Morphology : 's', 'sd', 'd', 'di', 'i'
      SEDtype    : 'early', 'late', 'starburst' 
           or a number in the range  1 - 6
           corresponding to the GOODZ template set
      priorIa/Ibc/II : the prior probability that this 
           SN is of Type Ia/Ibc/II 
             (defaults use the fractions in a volume-limited 
               sample at z=0 from Li et al 2011a)
      returnErrors : also propagate the uncertainties on 
          the galsnid likelihoods into errors on the 
          galsnid posterior probabilities and return them
    
    RETURNS:  [ p(Ia|D), p(Ibc|D), p(II|D) ]

      or, if returnErrors==True : 
         [ [ p(Ia|D), dp(Ia|D)_plus, dp(Ia|D)_minus ] ,
           [ p(Ibc|D), dp(Ibc|D)_plus, dp(Ibc|D)_minus ] ,
           [ p(II|D), dp(II|D)_plus, dp(II|D)_minus ] ]

    NOTE: if the user does not specify either a morphology or 
      an sedtype, then the returned values simply reflect 
      back the priors.

    """
    #import exceptions
    #if not (morphology or sedtype) :
    #    raise( exceptions.RuntimeError( "Must specify host galaxy morphology and/or SED type!") )
    from numpy import sqrt

    if priorIa == priorIbc and priorIa == priorII : 
        priorIa, priorIbc, priorII = [0.33,0.33,0.33]

    priorCC = (1-priorIa)

    # get the likelihoods : 
    pMIa, pMIbc, pMII, pMCC = 1, 1, 1, 1
    if morphology not in [None, 'None'] : 
        pMIa,dpMIaPlus,dpMIaMinus = pMorphIa( morphology )
        pMCC,dpMCCPlus,dpMCCMinus = pMorphCC( morphology )
    
    pSIa, pSIbc, pSII, pSCC = 1, 1, 1, 1
    if sedtype not in [None, 'None'] : 
        pSIa,dpSIaPlus,dpSIaMinus = pSEDIa( sedtype )
        pSCC,dpSCCPlus,dpSCCMinus = pSEDCC( sedtype )

    # define the normalization factor
    k = (priorIa * pMIa * pSIa) + (priorCC * pMCC * pSCC)

    # compute the posterior probabilities,
    # dividing the CC posterior probability into Ib/c and II 
    # according to the relative fraction in each sub-class
    postIa =  priorIa * pMIa * pSIa / k
    postCC =  priorCC * pMCC * pSCC / k
    fIbc = priorIbc / priorCC
    fII = priorII / priorCC

    postIbc, postII = postCC*fIbc, postCC*fII
    if not returnErrors : return( postIa, postIbc, postII  )

    # propagate the galsnid likelihood errors into errors 
    # on the galsnid posterior probabilities 
    # NOTE : this is really Q+D and ignores covariance !!
    dPIaPlus = sqrt(  (dpMIaPlus * (postIa/pMIa)**2)**2 + 
                      (dpSIaPlus * (postIa/pSIa)**2)**2 )
    dPCCPlus = sqrt(  (dpMCCPlus * (postCC/pMCC)**2)**2 + 
                      (dpSCCPlus * (postCC/pSCC)**2)**2 )
    dPIaMinus = sqrt( (dpMIaMinus * (postIa/pMIa)**2)**2 + 
                      (dpSIaMinus * (postIa/pSIa)**2)**2 )
    dPCCMinus = sqrt( (dpMCCMinus * (postCC/pMCC)**2)**2 + 
                      (dpSCCMinus * (postCC/pSCC)**2)**2 )
    dPIbcPlus = dPCCPlus * fIbc
    dPIbcMinus = dPCCMinus * fIbc
    dPIIPlus = dPCCPlus * fII
    dPIIMinus = dPCCMinus * fII

    return( [  [postIa,dPIaPlus,-dPIaMinus],   
               [postIbc,dPIbcPlus,-dPIbcMinus], 
               [postII,dPIIPlus,-dPIIMinus] ] )

def getPIaSDerrors( pIaSD, morphology=None, sedtype=None, priorIa=0.24, priorIbc=0.19, priorII=0.57 ) :
    """ propagate the  uncertainties on galsnid priors into an error 
    on the galsnid+STARDUST posterior classification probability for the 
    SN Ia class.
    
    INPUTS : 
     pIaSD  : the type Ia posterior classification probability
         computed by STARDUST, using galsnid for the priors
     morphology, sedtype, priorIa/Ibc/II : the inputs to galsnid.classify()
        for defining the galsnid posterior probabilities (which you have 
        used as the priors for STARDUST)

    RETURNS :  
      dP(Ia|D)_plus, dP(Ia|D)_minus  : the positive and negative 
         uncertainties on the posterior classification probabilities.

    WARNING : this is Q+D.  Should be reviewed and improved.
    ACTUALLY, it doesn't even work.  Needs overhaul
    """
    from numpy import sqrt

    # get the galsnid posterior probabilities (and their uncertainties):: 
    galsnidPosteriors = classify( 
        morphology=morphology, sedtype=sedtype, priorIa=priorIa, 
        priorIbc=priorIbc, priorII=priorII, returnErrors=True )

    pIaGal, dpIaPlus, dpIaMinus = galsnidPosteriors[0]
    pIbcGal,dpIbcPlus,dpIbcMinus = galsnidPosteriors[1]
    pIIGal, dpIIPlus, dpIIMinus = galsnidPosteriors[2]
    
    pCCGal = pIbcGal + pIIGal 
    dpCCPlus = dpIbcPlus + dpIIPlus
    dpCCMinus = dpIbcMinus + dpIIMinus
    
    dpIaSDGalPlus = sqrt( (dpIaPlus*(pIaSD/pIaGal)**2)**2 + (dpCCPlus*((1-pIaSD)/pCCGal)**2)**2 ) 
    dpIaSDGalMinus = sqrt( (dpIaMinus*(pIaSD/pIaGal)**2)**2 + (dpCCMinus*((1-pIaSD)/pCCGal)**2)**2 )
    
    return( dpIaSDGalPlus, dpIaSDGalMinus )

def pIaMorph( morphology, priorIa ):
    """ 
    The probability that a SN is of type Ia,
    given that host galaxy has the observed morphology, 
    with PIa as the prior probability that this SN
    is of type Ia. (e.g. the probability derived from
    light curve matching with STARDUST)
    """
    # get the likelihoods : 
    pDIa = pMorphIa( morphology )[0]
    pDCC = pMorphCC( morphology )[0]

    # define the normalization factor
    k = priorIa * pDIa + (1-priorIa) * pDCC

    # return the posterior probability
    return(  priorIa * pDIa / k )


def pCCMorph( morphology, priorCC ):
    """ 
    The probability that a SN is a CC SN,
    given that host galaxy has the observed morphology, 
    with priorCC as the prior probability that this SN
    is of type CC. (e.g. the probability derived from
    light curve matching with STARDUST)
    """
    # get the likelihoods : 
    pDIa = pMorphIa( morphology )[0]
    pDCC = pMorphCC( morphology )[0]

    # define the normalization factor
    k = priorCC * pDCC + (1-priorCC) * pDIa

    # return the posterior probability
    return(  priorCC * pDCC / k )



def pMorphIa( morphology ):
    """ P(D|Ia) : The probability that one would observe
    a SN host galaxy to have the given morphology, assuming 
    that the SN is of type Ia 

    Morphology may be specified as a hubble type :
      [ 'E', 'S0', 'Sa', 'Sb', 'Sbc', 'Sc', 'Scd', 'Irr' ]
    or using the CANDELS visual classification metrics 
    for spheroid/disk/irregular and mixed morphologies : :
      [ 's', 'sd',  'd', 'di', 'i' ]

    RETURNS :  P(D|Ia), errPplus, errPminus
    """
    if morphology   == 'E'  : return( 0.141, 0.021, -0.018 ) 
    elif morphology == 'S0' : return( 0.217, 0.026, -0.023 )
    elif morphology == 'Sa' : return( 0.149, 0.022, -0.019 )
    elif morphology == 'Sb' : return( 0.177, 0.023, -0.021 )
    elif morphology == 'Sbc': return( 0.117, 0.019, -0.017 )
    elif morphology == 'Sc' : return( 0.120, 0.019, -0.017 )
    elif morphology == 'Scd': return( 0.076, 0.016, -0.013 )
    elif morphology == 'Irr': return( 0.003, 0.004, -0.002 )

    elif morphology == 's'  : return( 0.253, 0.028, -0.025 ) 
    elif morphology == 'sd' : return( 0.255, 0.028, -0.025 )
    elif morphology == 'd'  : return( 0.353, 0.032, -0.030 )
    elif morphology == 'di' : return( 0.103, 0.018, -0.015 )   
    elif morphology == 'i'  : return( 0.035, 0.011, -0.009 )
    elif morphology == 'u'  : return( 1, 0.1, -0.1 )
    else : return( 1, 0.1, -0.1 )

def pMorphCC( morphology ):
    """ P(D|Ia) : The probability that one would observe
    a SN host galaxy to have the given morphology, assuming 
    that the SN is a CC SN.

    Morphology may be specified as a hubble type :
      [ 'E', 'S0', 'Sa', 'Sb', 'Sbc', 'Sc', 'Scd', 'Irr' ]
    or using the CANDELS visual classification metrics 
    for spheroid/disk/irregular and mixed morphologies : :
      [ 's', 'sd',  'd', 'di', 'i' ]

    RETURNS :  P(D|CC), errPplus, errPminus
    """
    if morphology   == 'E'  : return( 0.002, 0.003, -0.001 ) 
    elif morphology == 'S0' : return( 0.017, 0.007, -0.005 )
    elif morphology == 'Sa' : return( 0.142, 0.017, -0.015 )
    elif morphology == 'Sb' : return( 0.188, 0.020, -0.018 )
    elif morphology == 'Sbc': return( 0.231, 0.022, -0.020 )
    elif morphology == 'Sc' : return( 0.218, 0.021, -0.019 )
    elif morphology == 'Scd': return( 0.188, 0.020, -0.018 )
    elif morphology == 'Irr': return( 0.015, 0.006, -0.004 )

    elif morphology == 's'  : return( 0.009, 0.005, -0.003 ) 
    elif morphology == 'sd' : return( 0.151, 0.018, -0.016 )
    elif morphology == 'd'  : return( 0.529, 0.032, -0.030 )
    elif morphology == 'di' : return( 0.199, 0.020, -0.019 )   
    elif morphology == 'i'  : return( 0.112, 0.015, -0.014 )
    elif morphology == 'u'  : return( 1, 0.1, -0.1 )
    else : return( 1, 0.1, -0.1 )


def pColorIa( color ):
    """ returns the likelihood of observing  
    host galaxy with the given rest-frame 
    B-K color, assuming the SN is a Ia 

    RETURNS :  P(B-K|Ia)
    """
    if color < 3 :   return( 0.240, 0.05, 0.05 )
    elif color < 4 : return( 0.578, 0.05, 0.05 )
    else :           return( 0.183, 0.05, 0.05 )

def pColorCC( color ):
    """ returns the likelihood of observing  
    host galaxy with the given rest-frame 
    B-K color, assuming the SN is a CC

    RETURNS :  P(B-K|CC)
    """
    if color < 3 :   return( 0.484, 0.05, 0.05 )
    elif color < 4 : return( 0.485, 0.05, 0.05 )
    else :           return( 0.032, 0.05, 0.05 )


def pSEDIa( sedtype ):
    """ returns the likelihood of observing  
    host galaxy with the given rest-frame 
    B-K color, assuming the SN is a Ia.
    The SED type is from the GOODZ SED template 
    set (Dahlen et al 2010).  
    1=E, 2=Sbc, 3=Scd, 4=Irr, 5,6=Starburst
    plus 4 interpolations between each. 

    RETURNS :  P(sedtype|Ia), dPplus, dPminus
    """
    if sedtype in [None, 'None'] : return( 1, 0., 0. ) # unknown

    if not type(sedtype)==str : 
        if sedtype > 3.5 :   sedtype = 'SB'
        elif sedtype > 1.5 : sedtype = 'A'
        elif sedtype <= 1.5: sedtype = 'P'
        
    sedtype = sedtype.lower()
    if sedtype in ['starburst','sb'] : return( 0.129, 0.05, -0.05 ) # starburst
    elif sedtype in ['late','a']  :    return( 0.521, 0.05, -0.05 ) # late
    elif sedtype in ['early','p']  :   return( 0.351, 0.05, -0.05 ) # Early 
    else : return( 1, 0., 0. ) # unknown


def pSEDCC( sedtype ):
    """ returns the likelihood of observing  
    a host galaxy with the given SED type,
    assuming the SN is a CC.
    The SED type is from the GOODZ SED template 
    set (Dahlen et al 2010).  
    1=E, 2=Sbc, 3=Scd, 4=Irr, 5,6=Starburst
    plus 4 interpolations between each. 

    RETURNS :  P(sedtype|CC),  dPplus, dPminus
    """
    if sedtype in [None, 'None'] : return( 1, 0., 0. ) # unknown
    if not type(sedtype)==str : 
        if sedtype > 3.5 :   sedtype = 'SB'
        elif sedtype > 1.5 : sedtype = 'A'
        elif sedtype <= 1.5: sedtype = 'P'

    sedtype = sedtype.lower()
    if sedtype in ['starburst','sb'] : return( 0.303, 0.05, -0.05 ) # starburst
    elif sedtype in ['late','a']  :    return( 0.618, 0.05, -0.05 ) # late
    elif sedtype in ['early','p']  :   return( 0.080, 0.05, -0.05 ) # Early 
    else : return( 1, 0., 0. ) # unknown

      
def shortcut( sn ) : 
    """ For a given snana.SuperNova object sn, 
    quickly convert the posterior probabilities computed 
    using the 'mid' class fractions prior into the 
    posterior probabilities you get when adopting the 
    'galsnid' prior.   
    NOTE: we do not account for redshift dependence here,
    so this only works exactly if we have a known spec-z


    !!!  NOT  FUNCTIONAL !!!
    """
    return(0)
    import numpy as np
    k =  sn.bayesNormFactor

    priorIa = np.mean( sn.ClassSim.Ia.priorModel )
    priorIbc = np.mean( sn.ClassSim.Ibc.priorModel )
    priorII = np.mean( sn.ClassSim.II.priorModel )

    priorIaNew, priorIbcNew, priorIINew = pClassMorph( sn.HOST_TYPE, priorIa=priorIa, priorIbc=priorIbc, priorII=priorII )
    kNew = ( (priorIaNew/priorIa) * sn.PIa  +  (priorIbcNew/priorIbc) * sn.PIbc  + (priorIINew/priorII) * sn.PII )

    PIaNew = priorIaNew * sn.likeIa.sum() / kNew
    PIbcNew = priorIbcNew * sn.likeIbc.sum() / kNew
    PIINew = priorIINew * sn.likeII.sum() / kNew

    return(  PIaNew, PIbcNew , PIINew )



def mkCANDELStable():
    """ Q+D function to construct a modified table of P(D|Ia) 
    values, converting from the hubble sequence into the 
    CANDELS s/d/i morphology classes.
     CANDELS = HubbleSeq
       s   = E/S0 
       s+d = S0/Sa
       d   = Sb/Sbc/Sc
       d+i = Sc/Scd
       i   = Scd/Irr

      CANDELS  =  Rest-frame (B-K)_Vega
     type < 1.5       ~       B-K > 3.7   
    1.5 < type < 3.5  ~   2.76 < B-K < 3.7
     type > 3.5       ~       B-K < 2.76
    """

    import numpy as np

    # count of SNe of each type from Leaman et al 2011
    # (for our purposes here, we could just as well assume that N=1000
    #  or N=1 for each class, but these may be useful in the future 
    #  if we want to do more rigorous definition of the uncertainties, 
    #  though even better would be to repeat the Monte Carlo tests that 
    #  Foley+Mandel did)
    NIa = 274
    NIbc = 116
    NII = 324
    NCC = NIbc + NII

    #                     0    1       2      3      4      5      6      7
    #                     E    S0     Sa     Sb     Sbc    Sc     Scd    Irr
    PDIa = np.array( [ 0.141, 0.217, 0.149, 0.177, 0.117, 0.120, 0.076, 0.003 ] ) 
    PDCC = np.array( [ 0.002, 0.017, 0.142, 0.188, 0.231, 0.218, 0.188, 0.015 ] ) 

    NDIa = PDIa * NIa
    NDCC = PDCC * NCC
    # NDIa = array([ 38.634,  59.458,  40.826,  48.498,  32.058,  32.88 ,  20.824, 0.822])

    # s : spheroid  = E + S0 / 2.
    NsIa = NDIa[0] + NDIa[1] / 2.
    PsIa = NsIa / NIa
    NsCC = NDCC[0] + NDCC[1] / 2.
    PsCC = NsCC / NCC

    # sd : spheroid+disk = S0 / 2. + Sa
    NsdIa = NDIa[1] / 2. + NDIa[2] / 2.
    PsdIa = NsdIa / NIa
    NsdCC = NDCC[1] / 2. + NDCC[2] / 2.
    PsdCC = NsdCC / NCC

    # d : disk = Sb + Sbc + Sc/2.
    NdIa = NDIa[3] + NDIa[4] + NDIa[5] / 2.
    PdIa = NdIa / NIa
    NdCC = NDCC[3] + NDCC[4] + NDCC[5] / 2.
    PdCC = NdCC / NCC

    # di : disk+irregular = Sc/2. + Scd/2.
    NdiIa = NDIa[5]/2. + NDIa[6]/2.
    PdiIa = NdiIa / NIa
    NdiCC = NDCC[5]/2. + NDCC[6]/2.
    PdiCC = NdiCC / NCC

    # i : irregular = Scd/2 + Irr
    NiIa = NDIa[6]/2. + NDIa[7]
    PiIa = NiIa / NIa
    NiCC = NDCC[6]/2. + NDCC[7]
    PiCC = NiCC / NCC

    # NOTE : as a shortcut, for the colors
    #  I do assume NIa = NCC = 1 and simply 
    #  add up the likelihoods to get my composite bins
    # SB: blue, Starbursty : B-K < 2.75
    PbIa = 0.026 + 0.023 + 0.037 + 0.043
    PbCC = 0.044 + 0.075 + 0.069 + 0.115

    # A : green, Active (Late) Type : 2.75 < B-K < 3.75
    PgIa =   0.111 + 0.131 + 0.154 + 0.125
    PgCC =   0.181 + 0.185 + 0.137 + 0.115

    # P : red, Passive (Early) Type : B-K > 3.75
    PrIa =   0.168 + 0.103 + 0.080 
    PrCC =   0.048 + 0.022 + 0.010  

    tabletxt = """
  Table of likelihoods :
Morphology Type    P(D|Ia)    P(D|CC)
   spheroid         %5.3f      %5.3f
   spheroid+disk    %5.3f      %5.3f
   disk             %5.3f      %5.3f
   disk+irregular   %5.3f      %5.3f
   irregular        %5.3f      %5.3f

  SED type        B-K Color   P(D|Ia)    P(D|CC)
    Starburst     <2.75       %5.3f      %5.3f
    Active      2.75-3.75       %5.3f      %5.3f
    Passive       >3.75       %5.3f      %5.3f
"""%( PsIa, PsCC, PsdIa, PsdCC, PdIa, PdCC, PdiIa, PdiCC, PiIa, PiCC, 
      PbIa, PbCC, PgIa, PgCC, PrIa, PrCC  )
    print( tabletxt )


    tabletxt = """
\n\n  Table of posterior probabilities :
Morphology         P(Ia|D)    P(Ibc|D)   P(II|D)
   spheroid         %5.3f      %5.3f      %5.3f
   spheroid+disk    %5.3f      %5.3f      %5.3f
   disk             %5.3f      %5.3f      %5.3f
   disk+irregular   %5.3f      %5.3f      %5.3f
   irregular        %5.3f      %5.3f      %5.3f
"""%( classify( 's' ) + classify( 'sd' ) + \
          classify( 'd' ) + classify( 'di' ) + \
          classify( 'i' ) )
    print( tabletxt )


