import numpy as np
import sys
if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz

def magnitude_to_radius(teffs: list[float], ratios: list[float], band, obsmag, zp_system, Tfrac=0.9,
    filterdir='filter_transmissions/'):



    '''Estimate the radius of the star given a temperature,
    photometric filter and observed (dereddened) absolute
    magnitude.

    Input:
     - teff1: model effective temperature in K (float)
     - band: name of the photometric band (string), see section
       'Available photometric bands' at the start of of this
       functions for which ones are included, and the
       description below on how to add more.
     - obsmag: the observed absolute magnitude in the given
       band (float)
     - Tfrac: fraction of the effective temperature that used
       used for calculating the 'theoretical SED' aka black
       body curve (float)
     - zp_system: choose from 'vega', 'AB', 'ST' (string)
     - filterdir: specify (relative) path to the directory
       where the filter information is stored (string)
     - teff2: model effective temperature of companion in K (float), 0 corresponds to the single star case
     - R: model radius ratio R2/R1. 0 corresponds to the single star case

    Output:
     - Estimated stellar radius in solar units (float) of both companions, for single stars, only the first return value should be used (the other will be 0)

    NOTE ON ADDING NEW FILTERS

    A useful resource for filter information is the
    SVO Filter Profile Service:
       http://svo2.cab.inta-csic.es/theory/fps/

    When adding a new filter, please do the following:
    1. Place an asci file with wavelengths and transmissions in
       the filter directory (specified in the parameter
       'filterdir'. In this file, lines with columns names or
       headers should start with a '#'. Wavelengths can be in
       Angstrom or nm (see next point).
    2. Add an 'elif' statement in the code below under 'Available
       photometric bands', in which you give the filter a clear
       and descriptive name, and point to the transmission file.
       Wavelength units in the data file can be either nanometers
       or Angstrom, specify which one is used in the file in the
       parameter 'waveunit' in the elif statement.
    3. Add zero points to the file 'zero_points.dat' in the
       filterdirectory. In the first column give the name of
       the filter: use the same name as in point 2.

    NOTE ON THE 'THEORETICAL SED'

    The "theoretical SED" on which the radius estimate is based
    is a Planck function. The temperature used for this can be
    scaled with Tfrac, is now set default to 0.9, this is as done in
    Mokiem 2005, who follows Markova 2004.

    #FIXME it would be interesting to check whether this
    chosen value for Tfrac from Markova 2004 gives the best
    approximation by comparing for calculated models the
    real SEDs with the 0.9*teff black body spectrum, and see
    how those radii compare

    '''



    ##########################################################
    ###            Available photometric bands             ###
    ##########################################################

    if band == 'SPHERE_Ks':
        filterfile = 'SPHERE_IRDIS_B_Ks.dat'
        waveunit = 'nm'
    elif band == 'HST_555w':
        filterfile = 'HST_ACS_HRC.F555W.dat'
        waveunit = 'angstrom'
    elif band == '2MASS_Ks':
        filterfile = '2MASS_Ks.dat'
        waveunit = 'angstrom'
    elif band == 'VISTA_Ks':
        filterfile = 'Paranal_VISTA.Ks.dat'
        waveunit = 'angstrom'
    elif band == 'Johnson_V':
        filterfile = 'GCPD_Johnson.V.dat'
        waveunit = 'angstrom'
    elif band == "Johnson_J":
        filterfile = "Generic_Johnson.J.dat"
        waveunit = 'angstrom'
    else:
        print('Unknown value for <band>, exiting')
        sys.exit()

    ##########################################################
    ###             Computation starts here                ###
    ##########################################################

    # Read transmission profile and convert units if necessary
    filterfile = filterdir + filterfile
    wave, trans = np.loadtxt(filterfile, comments='#', unpack=True)

    if waveunit == 'nm':
        nm_to_Angstrom = 10
        wave = wave * nm_to_Angstrom
    elif waveunit == 'angstrom':
        pass
    else:
        print('Unknown value for <waveunit>, exiting')

    # Get filter zero point
    zpfile = filterdir + 'zero_points.dat'
    zp_values = np.genfromtxt(zpfile, comments='#', dtype=str)
    the_zero_point = ''
    for afilter in zp_values:
        if afilter[0] == band:
            if zp_system == 'vega':
                the_zero_point = float(afilter[1])
            elif zp_system == 'AB':
                the_zero_point = float(afilter[2])
            elif zp_system == 'ST':
                the_zero_point = float(afilter[3])
            else:
                print('Unknown value for <zp_system>, exiting')
                sys.exit()
    if the_zero_point == '':
        print('Zero point for band ' + band + ' not found, exiting')
        sys.exit()

    tBBs = [teff * Tfrac for teff in teffs]

    # Integration over angles results in the factor of pi
    F_lambdas = [np.pi * planck_wavelength(wave, tBB) for tBB in tBBs]

    rsun = 6.96e10
    parsec_cm = 3.08567758e18

    filtered_fluxes = [trapezoid(trans*F_lambda, wave)/trapezoid(trans, wave) for F_lambda in F_lambdas]

    obsflux = magnitude_to_flux(obsmag, the_zero_point)

    d = 10 * parsec_cm / rsun

    denom = 0
    for i in range(len(ratios)):
        filtered_flux = filtered_fluxes[i]
        ratio = ratios[i]
        denom += filtered_flux * ratio**2
    R = d*np.sqrt(obsflux/denom)
    radii = [R * ratio for ratio in ratios]

    return radii

def planck_wavelength(wave_angstrom, temp):
    ''' Calculate the Planck function as function of temperature,
    and wavelengt (in Angstrom, output is then also in Angstrom).
    '''

    angstrom_to_cm = 1e-8
    wave = wave_angstrom * angstrom_to_cm

    # All units in cgs
    hh = 6.6260755e-27 #Planck constant;
    cc = 2.99792458e10 #speed of light in a vacuum;
    kk = 1.380658e-16 #Boltzmann constant;

    prefactor = 2.0 * hh * cc**2 / (wave**5)
    exponent = (hh * cc / kk) / (wave * temp)
    Blambda = prefactor * (1.0 / (np.exp(exponent)-1))

    #Blambda from per cm to per angstrom
    Blambda = Blambda * angstrom_to_cm

    return Blambda

def magnitude_to_flux(magnitude, zpflux):
    ''' Calculate observed flux from magnitude and zeropoint flux'''
    obsflux = zpflux * 10**(-magnitude/2.5)
    return obsflux

def get_synthetic_magnitude(teffs: list[float], radii: list[float], band, zp_system, Tfrac = 0.9, filterdir='filter_transmissions/'):
    if band == 'SPHERE_Ks':
        filterfile = 'SPHERE_IRDIS_B_Ks.dat'
        waveunit = 'nm'
    elif band == 'HST_555w':
        filterfile = 'HST_ACS_HRC.F555W.dat'
        waveunit = 'angstrom'
    elif band == '2MASS_Ks':
        filterfile = '2MASS_Ks.dat'
        waveunit = 'angstrom'
    elif band == 'VISTA_Ks':
        filterfile = 'Paranal_VISTA.Ks.dat'
        waveunit = 'angstrom'
    elif band == 'Johnson_V':
        filterfile = 'GCPD_Johnson.V.dat'
        waveunit = 'angstrom'
    elif band == "Johnson_J":
        filterfile = "Generic_Johnson.J.dat"
        waveunit = 'angstrom'
    else:
        print('Unknown value for <band>, exiting')
        sys.exit()

    ##########################################################
    ###             Computation starts here                ###
    ##########################################################

    # Read transmission profile and convert units if necessary
    filterfile = filterdir + filterfile
    wave, trans = np.loadtxt(filterfile, comments='#', unpack=True)

    if waveunit == 'nm':
        nm_to_Angstrom = 10
        wave = wave * nm_to_Angstrom
    elif waveunit == 'angstrom':
        pass
    else:
        print('Unknown value for <waveunit>, exiting')

    # Get filter zero point
    zpfile = filterdir + 'zero_points.dat'
    zp_values = np.genfromtxt(zpfile, comments='#', dtype=str)
    the_zero_point = ''
    for afilter in zp_values:
        if afilter[0] == band:
            if zp_system == 'vega':
                the_zero_point = float(afilter[1])
            elif zp_system == 'AB':
                the_zero_point = float(afilter[2])
            elif zp_system == 'ST':
                the_zero_point = float(afilter[3])
            else:
                print('Unknown value for <zp_system>, exiting')
                sys.exit()
    if the_zero_point == '':
        print('Zero point for band ' + band + ' not found, exiting')
        sys.exit()

    tBBs = [teff * Tfrac for teff in teffs]

    # Integration over angles results in the factor of pi
    F_lambdas = [np.pi * planck_wavelength(wave, tBB) for tBB in tBBs]

    rsun = 6.96e10
    parsec_cm = 3.08567758e18

    filtered_fluxes = [trapezoid(trans*F_lambda, wave)/trapezoid(trans, wave) for F_lambda in F_lambdas]

    d = 10 * parsec_cm / rsun

    denom = 0
    for i in range(len(radii)):
        filtered_flux = filtered_fluxes[i]
        radius = radii[i]
        denom += filtered_flux * radius**2

    obsflux = denom/d**2
    magnitude = -2.5*np.log10(obsflux/the_zero_point)
    return magnitude