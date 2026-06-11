# Functions for GA analysis script part of Kiwi-GA
# Created by Sarah Brands @ 29 July 2022
import copy
import functools
import os
import sys
import math
import tarfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import fastwind_wrapper as fw
import magnitude_to_radius as m2r
import population as pop
from epoch import Epoch

RMSEA_THRESHOLD = 1.5

def load_snapshot(savedmoddir, gen):
    return pop.Population.from_file(savedmoddir + str(gen).zfill(4) + '/' + str(gen).zfill(4) + '_snapshot.pkl')

def get_luminosity(Teff, radius):
    '''Calculate L in terms of log(L/Lsun), given Teff (K)
    and the radius in solar radii'''

    sigmaSB = 5.67051e-5
    Lsun = 3.9e33
    Rsun = 6.96e10

    radius_cm = radius * Rsun
    luminosity_cgs  = 4*math.pi * sigmaSB * Teff**4 * radius_cm**2
    luminosity = np.log10(luminosity_cgs / Lsun)

    return luminosity

def get_mass(logg, Rstar):
    ''' Gives Mstar in solar units, given logg and Rstar in solar units'''

    Msun = 1.99e33
    Rsun = 6.96e10
    Gcgs = 6.67259e-8

    g = 10**logg
    Rstar = Rstar*Rsun

    Mstar = g * Rstar**2 / Gcgs

    return Mstar / Msun

# def get_fx(mdot, vinf):
#     """ Estimates fx based on the Mdot and vinf, based on the
#     power law of Kudritzki, Palsa, Feldmeier et al. (1996). This power law
#     is extrapolated also outside where Kudritzki+96 have data points.
#     """
#
#     mdot = 10**mdot / 10**(-6)
#     logmdotvinf = np.log10(mdot/vinf)
#
#     # Relation from Kudritzki, Palsa, Feldmeier et al. (1996)
#     logfx = -5.45 - 1.05*logmdotvinf
#     # fx = 10**(logfx)
#
#     return logfx

def get_Gamma_Edd(Lum, Mass, kappa_e=0.344):
    """
    * Lum = luminosity in solor luminosity (no log)
    * Mass = stellar mass in solar mass

    kappa_e default from Bestenlehner (2020) page 3942
    """

    Lsun = 3.9e33
    Msun = 1.99e33
    ccgs = 2.99792458*10**10 #cm/s
    Gcgs = 6.67259e-8

    Lum = Lum*Lsun
    Mass = Mass*Msun

    GammE = Lum * kappa_e / (4.*np.pi*ccgs*Gcgs*Mass)

    return GammE

def get_vesc_eff(mass, radius, GammE):
    Rsun = 6.96e10
    Msun = 1.99e33
    Gcgs = 6.67259e-8

    mass_eff = mass*(1-GammE)

    if mass_eff < 0:
        mass_eff = 0

    vesc_cms = np.sqrt((2*Gcgs*mass_eff*Msun)/(radius*Rsun))
    vesc_kms = vesc_cms*1e-5

    return vesc_kms

def get_fw_fluxcont(fwdir):

    rsun = 6.957e10 #cm
    fluxcont = fwdir + 'FLUXCONT'
    indat = fwdir + 'INDAT.DAT'

    Rmax_fw_model = float(open(indat, 'r').readlines()[4].strip().split()[0])
    rstar = float(open(indat, 'r').readlines()[3].strip().split()[-1])
    mdot = float(open(indat, 'r').readlines()[5].strip().split()[0])

    stellar_surface = 4*np.pi*(rsun*rstar)**2

    # Look up the number of useful lines in the FLUXCONT
    lcount = -2
    for aline in open(fluxcont, 'r').readlines():
        lcount = lcount+1
        if len(aline.split()) == 1:
            break

    # Get FASTWIND spectrum
    lam, logFnu = np.genfromtxt(fluxcont, max_rows=lcount,
        skip_header=1, delimiter='').T[1:3]
    fnu = 10**logFnu # ergs/s/cm^2/Hz / RMAX^2
    flam = 3.00* 1e18 * fnu / (lam**2) # ergs/s/cm^2/A /RMAX^2
    flam = flam * stellar_surface # ergs/s/A /RMAX^2
    flam = flam * Rmax_fw_model**2 # ergs/s/A

    sorting = lam.argsort()
    lam = lam[sorting]
    flam = flam[sorting]

    return lam, flam

def magnitude_to_radius_SED(sed_wave, sed_flam, band, obsmag, zp_system,
    filterdir='filter_transmissions/'):

    '''Compute the radius of the star based on a fastwind model
    and observed (dereddened) absolute magnitude.

    Input:
     - band: name of the photometric band (string), see section
       'Available photometric bands' at the start of of this
       functions for which ones are included, and the
       description below on how to add more.
     - obsmag: the observed absolute magnitude in the given
       band (float)
     - zp_system: choose from 'vega', 'AB', 'ST' (string)
     - filterdir: specify (relative) path to the directory
       where the filter information is stored (string)

    Output:
     - Stellar radius in solar units (float)

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

    sed_ip = interp1d(sed_wave, sed_flam)
    F_lambda = sed_ip(wave)

    rsun = 6.96e10
    parsec_cm = 3.08567758e18
    radius_ratio = 10*parsec_cm / rsun

    filtered_flux = np.trapz(trans*F_lambda, wave)/np.trapz(trans, wave)
    obsflux = m2r.magnitude_to_flux(obsmag, the_zero_point)
    bolflux_10pc = obsflux/filtered_flux
    luminosity = bolflux_10pc * (10*parsec_cm / rsun)**2
    radius_rsun = luminosity**0.5

    return radius_rsun

def more_parameters(populations: list[pop.Population]):
    plists = []
    example = populations[-1].population[0]
    for comp in example.components:
        plist = {'logL', 'radius', 'Mspec', 'Gamma_Edd', 'vesc_eff', 'logq0', 'logQ0', 'logq1', 'logQ1', 'logq2', 'logQ2'}
        for epoch_name, vrad in comp.vrads.items():
            plist.add(epoch_name.split('/')[-1])
        all_params = comp.parameters.keys()
        for param in all_params:
            if not param in comp.template.variables.keys():
                if param in comp.template.fixed.keys():
                    orig_val = comp.template.fixed[param]
                    if orig_val == 'T' or orig_val == 'F':
                        pass
                    elif '.' in orig_val:
                        orig_val = float(orig_val)
                    else:
                        orig_val = int(orig_val)
                    if orig_val != comp.parameters[param]:
                        plist.add(param)
                else:
                    plist.add(param)
        if 'xlum' in comp.template.variables.keys():
            plist.add('logxlum')
        if 'vinf' in all_params:
            plist.add('vinf_vesc')
        if 'windturb' in all_params and 'vinf' in all_params:
            plist.add('windturb_kms')
        if 'mdot' in all_params and 'fclump' in all_params:
            plist.add('mdot_fclump')
        plists.append(sorted(list(plist)))

    for generation in populations:
        for individual in generation.population:
            for comp, plist in zip(individual.components, plists):
                for epoch_name, vrad in comp.vrads.items():
                    comp.parameters[epoch_name.split('/')[-1]] = float(vrad)
                comp.parameters['mdot'] = np.log10(float(comp.parameters['mdot']))
                comp.parameters['radius'] = comp.radius
                # Always get the luminosity and spectroscopic mass and Eddington factor
                comp.parameters['logL'] = get_luminosity(comp.parameters['teff'], comp.parameters['radius'])
                comp.parameters['Mspec'] = get_mass(comp.parameters['logg'], comp.parameters['radius'])
                comp.parameters['Gamma_Edd'] = get_Gamma_Edd(10**comp.parameters['logL'], comp.parameters['Mspec'])
                comp.parameters['vesc_eff'] = get_vesc_eff(comp.parameters['Mspec'], comp.parameters['radius'],comp.parameters['Gamma_Edd'])

                # If X-rays are given and variable then include them
                if 'logxlum' in plist:
                    if float(comp.parameters['xlum']) <= 0:
                        comp.parameters['xlum'] = 1e-20
                    comp.parameters['logxlum'] = np.log10(float(comp.parameters['xlum']))

                # Other derived parameters are only computed when relevant.
                if 'vinf_vesc' in plist:
                    comp.parameters['vinf_vesc'] = comp.parameters['vinf']/comp.parameters['vesc_eff']
                if 'windturb_kms' in plist:
                    comp.parameters['windturb_kms'] = comp.parameters['windturb'] * comp.parameters['vinf']
                if 'mdot_fclump' in plist:
                    comp.parameters['mdot_fclump'] = np.log10(10**comp.parameters['mdot'] * np.sqrt(comp.parameters['fclump']))

    return plists


def calculateP(chi2, best_chi2, degreesFreedom, normalize):
    """
    Based on the chi2 value of a model, compute the P-value
    Before this is done, all chi2 values are normalised by the lowest
    chi2 value of the run.
    """

    if normalize:
        scaling = best_chi2
    else:
        scaling = degreesFreedom

    # In principle, don't use this correction factor (keep set to 1.0)
    # Can be used to make error bars artificially larger
    correction_factor = 1.0
    if correction_factor != 1.0:
        print("!"*70)
        print("\n\n\n       WARNING!!!!!!!! chi2 correction\n\n\n")
        print("!"*70)
        print("chi2 of all models artificially lowered in order to enlarge")
        print("uncertainties\n\n\n")

    chi2 =  correction_factor * (chi2 * degreesFreedom) / scaling
    probs = stats.chi2.sf(chi2, degreesFreedom)
    return probs

def calculateP_noncent(chi2, degreesFreedom, lambda_nc):
    """
    Based on the chi2 value of a model, compute the P-value assuming a
    non-central chi2 distribution
    """

    # scaling = np.min(chi2)
    # chi2 = chi2 /scaling * degreesFreedom

    probs = np.zeros_like(chi2)
    try:
        for i in range(len(chi2)):
            probs[i] = stats.ncx2.sf(chi2[i], degreesFreedom, lambda_nc)
    except:
        chi2 = chi2.values
        for i in range(len(chi2)):
            probs[i] = stats.ncx2.sf(chi2[i], degreesFreedom, lambda_nc)
    return probs

def update_magnitude(m_name_orig, m_value_orig, m_system_orig,
    the_runname):
    ''' Look up runname to check if a different magniude should be adopted'''

    fname_muptdate = "lum_anchor_update.dat"
    if os.path.isfile(fname_muptdate):
        the_rnames = np.genfromtxt(fname_muptdate, dtype='str', usecols=[0]).T
        if not the_runname in the_rnames:
            return m_name_orig, m_value_orig, m_system_orig
        updatefile = open(fname_muptdate, 'r')
        allines_mupdate = updatefile.readlines()
        for magline in allines_mupdate[1:]:
            maglinelist = magline.strip().split()
            if len(maglinelist) == 4:
                runname0 = maglinelist[0]
                if runname0 == the_runname:
                    m_name_new = maglinelist[1]
                    m_value_new = float(maglinelist[2])
                    m_system_new = maglinelist[3]
                    print('Adopting new luminosity anchor')
                    return m_name_new, m_value_new, m_system_new
            # Extra option to include an uncertainty on the magnitude
            elif len(maglinelist) == 5:
                runname0 = maglinelist[0]
                if runname0 == the_runname:
                    m_name_new = maglinelist[1]
                    m_value_new = float(maglinelist[2])
                    m_value_error = float(maglinelist[3])
                    m_system_new = maglinelist[4]
                    print('Adopting new luminosity anchor with uncertainty')
                    return m_name_new, m_value_new,\
                           m_system_new, m_value_error
            else:
                errstr = "ERROR IN " + fname_muptdate + '!'
                errstr = errstr + '\n press enter to use UNCHANGED values.'
                input(errstr)
    else:
        return m_name_orig, m_value_orig, m_system_orig


def radius_correction(df, fw_path, runname, thecontrolfile, theradiusfile,
    datapath, outpath, comp_fw):
    radcorrfile = outpath + 'radius_correction.txt'
    if os.path.isfile(radcorrfile):
        os.remove(radcorrfile)

    xbest = pd.Series.idxmin(df['rchi2'])
    best_model_name = df['#run_id'][xbest]
    print('Best model:', best_model_name)
    best_gen_name = best_model_name.split('_')[0]
    bestmod_fw = fw_path + runname + '_' + best_model_name + '/'
    savemoddir = datapath + 'saved/' + best_gen_name + '/'
    bestmoddir = savemoddir + best_model_name.split('_')[1] + '/'
    if not os.path.isdir(bestmoddir + 'combined/'):
        fw.mkdir(bestmoddir + 'combined/')
        fw.untar(bestmoddir + 'combined.tar.gz', bestmoddir + 'combined/')
    params = pd.read_csv(bestmoddir + 'combined/params.csv')
    if not os.path.isdir(bestmoddir + '0/'):
        for index, row in params.iterrows():
            newloc = bestmoddir + str(index) + '/'
            name = row['run_id']
            compparts = name.split('_')
            compname = str(index)
            if len(compparts) == 4:
                compname += '_' + compparts[-1]
            fw.mkdir(newloc)
            fw.untar(bestmoddir + compname + '.tar.gz', newloc)
    the_best_indats = []
    for index, row in params.iterrows():
        newloc = bestmoddir + str(index) + '/'
        the_best_indats.append(newloc + 'INDAT.DAT')

    if False:
        if not os.path.isfile(bestmod_fw + 'FLUXCONT'):
            if comp_fw:
                for best_indat in the_best_indats:
                    os.system('mkdir -p ' + bestmod_fw)
                    moddir = savemoddir + best_model_name + '/'
                    modtar = savemoddir + best_model_name + '.tar.gz'
                    if not os.path.isdir(moddir):
                        os.system('mkdir -p ' + moddir)
                        os.system('tar -xzf ' + modtar + ' -C ' + moddir + '/.')
                    os.system('cp ' + best_indat + ' ' + fw_path + '.')
                    fwindat = fw_path + 'INDAT.DAT'
                    pnlte_logfile = (fw_path + runname + '_' + best_model_name
                        + '.pnltelog')

                    with open(fwindat) as f:
                        lines = f.readlines()
                    lines[0] = "'" + runname + '_' + best_model_name + "'\n"
                    with open(fwindat, "w") as f:
                        f.writelines(lines)

                    for acontrl in np.genfromtxt(thecontrolfile,dtype='str'):
                        if acontrl[0] == 'modelatom':
                            modelatom = acontrl[1]
                            break
                    currentdir = os.getcwd()
                    os.chdir(fw_path)
                    print('Start FASTWIND Computation ...', 3*'\n...')
                    print('    ... pnlte output here: ' + pnlte_logfile)
                    runpnlte = './pnlte_' + modelatom + '.eo > ' + pnlte_logfile
                    os.system(runpnlte)
                    os.chdir(currentdir)
                    if os.path.isfile(bestmod_fw + 'FLUXCONT'):
                        corr_ready = True
                        os.system('rm ' + pnlte_logfile)

                    else:
                        print('\n\n\nERROR! fw model could not compute, check!\n\n\n')
                        corr_ready = False
            else:
                corr_ready = False
        else:
            corr_ready = True

        if corr_ready:
            lam, flam = get_fw_fluxcont(bestmod_fw)

            fwindat = fw_path + 'INDAT.DAT'

            rsun = 6.96e10 # cm
            tefffact = 0.9
            mod_rstar = float(open(fwindat, 'r').readlines()[3].strip().split()[-1])
            stellar_surface = 4*np.pi*(rsun*mod_rstar)**2

            m_name = np.genfromtxt(theradiusfile, dtype='str')[0]
            m_value = np.genfromtxt(theradiusfile)[1]
            m_system = np.genfromtxt(theradiusfile, dtype='str')[2]

            m_name, m_value, m_system = update_magnitude(m_name, m_value, m_system,
                runname)[:3]

            new_rad = magnitude_to_radius_SED(lam, flam/stellar_surface,
                m_name, m_value, m_system,
                filterdir='filter_transmissions/')

            radius_ratio = new_rad/mod_rstar

            df['Q_radius_old'] = (10**df['mdot'])/(df['radius'])**(3./2.)

            # Correct all radii with the perc. correction from the best fit model.
            df['radius'] = df['radius']*radius_ratio

            # Correct mass loss rates by assuming a fixed Q value (Puls+96)
            df['mdot'] = np.log10(df['Q_radius_old']*(df['radius'])**(3./2.))

            df['q0'] = 10**df['logq0']
            df['logQ0'] = np.log10(df['q0']*4*np.pi*(rsun*df['radius'])**2)
            df['q1'] = 10**df['logq1']
            df['logQ1'] = np.log10(df['q1']*4*np.pi*(rsun*df['radius'])**2)
            df['q2'] = 10**df['logq2']
            df['logQ2'] = np.log10(df['q2']*4*np.pi*(rsun*df['radius'])**2)

            with open(radcorrfile, 'w') as f:
                f.write('# Radius corretion (= corrected radius/estimated radius\n')
                f.write(str(radius_ratio) + '\n')

        else:
            print("WARNING! No radius correction done. ")
    else:
        print("WARNING! No radius correction done. Currently not implemented")
    return df, the_best_indats, best_model_name, bestmoddir

def get_uncertainties(best_model: pop.Individual, populations: list[pop.Population], npspec, paramspaces: list[pop.Template], deriv_pars, incl_deriv=True):

    best_rchi2 = best_model.fitting_params['rchi2']
    if best_rchi2 > RMSEA_THRESHOLD:
        which_statistic = 'RMSEA' # 'Pval_chi2' or 'Pval_ncchi2' or 'RMSEA'
    else:
        which_statistic = 'Pval_chi2'

    best_model.fitting_params['RMSEA'] = np.sqrt(
        (best_model.fitting_params['chi2'] - best_model.fitting_params['dof']) / (
                    best_model.fitting_params['dof'] * (npspec - 1)))
    minRMSEA = best_model.fitting_params['RMSEA']
    # closefit_RMSEA = minRMSEA + 0.005
    closefit_RMSEA = minRMSEA
    lambda_nc = (closefit_RMSEA) ** 2 * best_model.fitting_params['dof'] * (npspec - 1)

    min_p_1sig = 0.0
    min_p_2sig = 0.0
    if which_statistic in ('Pval_ncchi2', 'Pval_chi2'):
        min_p_1sig = 0.317
        min_p_2sig = 0.0455
    elif which_statistic == 'RMSEA':
        min_p_1sig = minRMSEA * 1.04
        min_p_2sig = minRMSEA * 1.09

    ind_1sig: list[pop.Individual] = []
    ind_2sig: list[pop.Individual] = []
    for generation in populations:
        for ind in generation.population:
            # Assign P-vaues and compute inverse reduced chi2
            ind.fitting_params['invrchi2'] = 1./ind.fitting_params['rchi2']
            ind.fitting_params['norm_rchi2'] = ind.fitting_params['rchi2']/best_rchi2

            ind.fitting_params['RMSEA'] = np.sqrt((ind.fitting_params['chi2']-ind.fitting_params['dof'])/(ind.fitting_params['dof']*(npspec-1)))

            if which_statistic == 'Pval_ncchi2':
                ind.fitting_params['P-value'] = calculateP_noncent(ind.fitting_params['chi2'], ind.fitting_params['dof'], lambda_nc)
            else:
                # ORIGINAL P-VALUE
                ind.fitting_params['P-value'] = calculateP(ind.fitting_params['chi2'], best_model.fitting_params['chi2'], ind.fitting_params['dof'], normalize=True)

            if which_statistic == 'RMSEA':
                if ind.fitting_params['RMSEA'] <= min_p_1sig:
                    ind_1sig.append(ind)
                if ind.fitting_params['RMSEA'] <= min_p_2sig:
                    ind_2sig.append(ind)
            else:
                if ind.fitting_params['P-value'] >= min_p_1sig:
                    ind_1sig.append(ind)
                if ind.fitting_params['P-value'] >= min_p_2sig:
                    ind_2sig.append(ind)


    # Store the best fit parameters and 1 and 2 sig uncertainties in a dict
    # shape of the resulting data:
    #[{param1: [low, up, val], param2: [low, up, val], ...}, {param1: [low, up, val], param2: [low, up, val], ...}, ...]
    errors_1sig = []
    errors_2sig = []
    for comp_idx in range(len(paramspaces)):
        params_error_1sig = {}
        params_error_2sig = {}
        for key, value in paramspaces[comp_idx].variables.items():
            the_step_size = float(value[2])
            params_1sig = []
            params_2sig = []
            for ind in ind_1sig:
                params_1sig.append(ind.components[comp_idx].parameters[key])
            for ind in ind_2sig:
                params_2sig.append(ind.components[comp_idx].parameters[key])
            params_error_1sig[key] = [min(params_1sig)-the_step_size,
                                      max(params_1sig)+the_step_size,
                                      best_model.components[comp_idx].parameters[key]]
            params_error_2sig[key] = [min(params_2sig)-the_step_size,
                                      max(params_2sig)+the_step_size,
                                      best_model.components[comp_idx].parameters[key]]
        errors_1sig.append(params_error_1sig)
        errors_2sig.append(params_error_2sig)
    if incl_deriv:
        deriv_errors_1sig = []
        deriv_errors_2sig = []
        for comp_idx in range(len(paramspaces)):
            deriv_params_error_1sig = {}
            deriv_params_error_2sig = {}
            for key in deriv_pars[comp_idx]:
                params_1sig = []
                params_2sig = []
                for ind in ind_1sig:
                    params_1sig.append(ind.components[comp_idx].parameters[key])
                for ind in ind_2sig:
                    params_2sig.append(ind.components[comp_idx].parameters[key])
                deriv_params_error_1sig[key] = [float(min(params_1sig)),
                                                float(max(params_1sig)),
                                                float(best_model.components[comp_idx].parameters[key])]
                deriv_params_error_2sig[key] = [float(min(params_2sig)),
                                                float(max(params_2sig)),
                                                float(best_model.components[comp_idx].parameters[key])]
            deriv_errors_1sig.append(deriv_params_error_1sig)
            deriv_errors_2sig.append(deriv_params_error_2sig)

    # Read best model names (for plotting of line profiles)
    bestfamily = ind_2sig

    n1sig = len(ind_1sig)
    n2sig = len(ind_2sig)

    if incl_deriv:
        best_uncertainty = (best_model, bestfamily, errors_1sig,
            errors_2sig, deriv_errors_1sig, deriv_errors_2sig,
            which_statistic)
    else:
        best_uncertainty = (best_model, bestfamily, errors_1sig,
            errors_2sig, which_statistic)

    return best_uncertainty, n1sig, n2sig

def get_local_uncertainties(generation: pop.Population, npspec, paramspaces: list[pop.Template], deriv_pars, best_so_far, incl_deriv=True):

    best_model = best_so_far
    best_rchi2 = best_model.fitting_params['rchi2']
    for ind in generation.population:
        if ind.fitting_params['rchi2'] <= best_rchi2:
            best_model = ind
            best_rchi2 = ind.fitting_params['rchi2']

    if best_rchi2 > RMSEA_THRESHOLD:
        which_statistic = 'RMSEA' # 'Pval_chi2' or 'Pval_ncchi2' or 'RMSEA'
    else:
        which_statistic = 'Pval_chi2'
    best_model.fitting_params['RMSEA'] = np.sqrt(
        (best_model.fitting_params['chi2'] - best_model.fitting_params['dof']) / (
                    best_model.fitting_params['dof'] * (npspec - 1)))
    minRMSEA = best_model.fitting_params['RMSEA']
    # closefit_RMSEA = minRMSEA + 0.005
    closefit_RMSEA = minRMSEA
    lambda_nc = (closefit_RMSEA) ** 2 * best_model.fitting_params['dof'] * (npspec - 1)

    min_p_1sig = 0.0
    min_p_2sig = 0.0
    if which_statistic in ('Pval_ncchi2', 'Pval_chi2'):
        min_p_1sig = 0.317
        min_p_2sig = 0.0455
    elif which_statistic == 'RMSEA':
        min_p_1sig = minRMSEA * 1.04
        min_p_2sig = minRMSEA * 1.09

    ind_1sig: list[pop.Individual] = []
    ind_2sig: list[pop.Individual] = []
    if best_model == best_so_far:
        ind_1sig.append(best_model)
        ind_2sig.append(best_model)
    for ind in generation.population:
        # Assign P-vaues and compute inverse reduced chi2
        ind.fitting_params['invrchi2'] = 1./ind.fitting_params['rchi2']
        ind.fitting_params['norm_rchi2'] = ind.fitting_params['rchi2']/best_rchi2

        ind.fitting_params['RMSEA'] = np.sqrt((ind.fitting_params['chi2']-ind.fitting_params['dof'])/(ind.fitting_params['dof']*(npspec-1)))

        if which_statistic == 'Pval_ncchi2':
            ind.fitting_params['P-value'] = calculateP_noncent(ind.fitting_params['chi2'], ind.fitting_params['dof'], lambda_nc)
        else:
            # ORIGINAL P-VALUE
            ind.fitting_params['P-value'] = calculateP(ind.fitting_params['chi2'], best_model.fitting_params['chi2'], ind.fitting_params['dof'], normalize=True)

        if which_statistic == 'RMSEA':
            if ind.fitting_params['RMSEA'] <= min_p_1sig:
                ind_1sig.append(ind)
            if ind.fitting_params['RMSEA'] <= min_p_2sig:
                ind_2sig.append(ind)
        else:
            if ind.fitting_params['P-value'] >= min_p_1sig:
                ind_1sig.append(ind)
            if ind.fitting_params['P-value'] >= min_p_2sig:
                ind_2sig.append(ind)

    # Store the best fit parameters and 1 and 2 sig uncertainties in a dict
    # shape of the resulting data:
    #[{param1: [low, up, val], param2: [low, up, val], ...}, {param1: [low, up, val], param2: [low, up, val], ...}, ...]
    errors_1sig = []
    errors_2sig = []
    for comp_idx in range(len(paramspaces)):
        params_error_1sig = {}
        params_error_2sig = {}
        for key, value in paramspaces[comp_idx].variables.items():
            the_step_size = float(value[2])
            params_1sig = []
            params_2sig = []
            for ind in ind_1sig:
                params_1sig.append(ind.components[comp_idx].parameters[key])
            for ind in ind_2sig:
                params_2sig.append(ind.components[comp_idx].parameters[key])
            if params_1sig:
                params_error_1sig[key] = [min(params_1sig)-the_step_size,
                                          max(params_1sig)+the_step_size,
                                          best_model.components[comp_idx].parameters[key]]
                params_error_2sig[key] = [min(params_2sig)-the_step_size,
                                          max(params_2sig)+the_step_size,
                                          best_model.components[comp_idx].parameters[key]]
            else:
                params_error_1sig[key] = [0, 0, 0]
                params_error_2sig[key] = [0, 0, 0]
        errors_1sig.append(params_error_1sig)
        errors_2sig.append(params_error_2sig)
    if incl_deriv:
        deriv_errors_1sig = []
        deriv_errors_2sig = []
        for comp_idx in range(len(paramspaces)):
            deriv_params_error_1sig = {}
            deriv_params_error_2sig = {}
            for key in deriv_pars[comp_idx]:
                params_1sig = []
                params_2sig = []
                for ind in ind_1sig:
                    params_1sig.append(ind.components[comp_idx].parameters[key])
                for ind in ind_2sig:
                    params_2sig.append(ind.components[comp_idx].parameters[key])
                deriv_params_error_1sig[key] = [float(min(params_1sig)),
                                                float(max(params_1sig)),
                                                float(best_model.components[comp_idx].parameters[key])]
                deriv_params_error_2sig[key] = [float(min(params_2sig)),
                                                float(max(params_2sig)),
                                                float(best_model.components[comp_idx].parameters[key])]
            deriv_errors_1sig.append(deriv_params_error_1sig)
            deriv_errors_2sig.append(deriv_params_error_2sig)

    # Read best model names (for plotting of line profiles)
    bestfamily = ind_2sig

    n1sig = len(ind_1sig)
    n2sig = len(ind_2sig)

    if incl_deriv:
        best_uncertainty = (best_model, bestfamily, errors_1sig,
            errors_2sig, deriv_errors_1sig, deriv_errors_2sig,
            which_statistic)
    else:
        best_uncertainty = (best_model, bestfamily, errors_1sig,
            errors_2sig, which_statistic)

    return best_uncertainty, n1sig, n2sig


def propagate_uncertainty(value_dict, param_name, radius, delta_radius, power,
                          log=False):
    """
    Propagates the uncertainty on the radius to other parameters.
    Assumes the additional error term is normally distributed and small enough
    to be considered symmetric in powerlaws.
    value_dict assumes for each key the shape (lower_limit, upper_limit, best)
    param_name indicates the parameter to propagate error to.
    radius and delta_radius are the radius and extra uncertainty on it. Note
    that the delta_radius has to correspond to the desired nsigma.
    power is the power dependence on the radius for the parameter.
    """
    if log:
        best = 10**value_dict[param_name][2]
        low = best - 10**value_dict[param_name][0]
        up = 10**value_dict[param_name][1] - best
    else:
        best = value_dict[param_name][2]
        low = best - value_dict[param_name][0]
        up = best - value_dict[param_name][1]

    # The additional uncertainty term:
    extra_err = best * (delta_radius / radius) * power

    new_low = (extra_err**2 + low**2)**0.5
    new_up = (extra_err**2 + up**2)**0.5

    if log:
        value_dict[param_name][0] = np.log10(best - new_low)
        value_dict[param_name][1] = np.log10(best + new_up)
    else:
        value_dict[param_name][0] = best - new_low
        value_dict[param_name][1] = best + new_up

    return value_dict


#not currently implemented
def add_anchor_magnitude_uncertainty(df, runname, best_uncertainty,
                                     fw_path, theradiusfile):
    """
    not currently implemented
    Adds the uncertainty from the error in the absolute magnitude. The error is
    taken from the lum_anchor_update file. Performs error propagation to all
    relevant parameters.
    """
    # Get the original magnitude first
    m_name = np.genfromtxt(theradiusfile, dtype='str')[0]
    m_value = np.genfromtxt(theradiusfile)[1]
    m_system = np.genfromtxt(theradiusfile, dtype='str')[2]
    # try to update
    new_mag = update_magnitude(m_name, m_value, m_system, runname)
    if len(new_mag) == 4:
        m_name, m_value, m_system, m_error = new_mag
    else:
        print("Not adding anchor magnitude uncertainties!")
        return best_uncertainty

    # Check if a SED has been calculated for this run. Use that if available.
    # Otherwise use the default approximation for the radius.
    xbest = pd.Series.idxmin(df['rchi2'])
    best_model_name = df['run_id'][xbest]
    bestmod_fw = fw_path + runname + '_' + best_model_name + '/'
    if os.path.isfile(bestmod_fw + 'FLUXCONT'):
        lam, flam = get_fw_fluxcont(bestmod_fw)
        fwindat = fw_path + 'INDAT.DAT'

        rsun = 6.96e10 # cm
        mod_rstar = float(open(fwindat, 'r').readlines()[3].strip().split()[-1])
        stellar_surface = 4*np.pi*(rsun*mod_rstar)**2

        new_rad = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value, m_system)
        max_rad1 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value - m_error, m_system)
        max_rad2 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value - 2 * m_error, m_system)
        min_rad1 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value + m_error, m_system)
        min_rad2 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value + 2 * m_error, m_system)

    else:
        print("No SED for radius correction found, using approximation")
        new_rad = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value, m_system)[0]
        max_rad1 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value - m_error, m_system)[0]
        max_rad2 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value - 2*m_error, m_system)[0]
        min_rad1 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value + m_error, m_system)[0]
        min_rad2 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value + 2*m_error, m_system)[0]

    delta_radius1 = (max_rad1 - min_rad1) * 0.5
    delta_radius2 = (max_rad2 - min_rad2) * 0.5

    # Unpack uncertainties
    best_model_name, bestfamily_name, pars_err1, pars_err2, d_pars_err1,\
        d_pars_err2, which_statistic = best_uncertainty

    # All the parameters that need to have their uncertainty updated based on
    # the changed radius uncertainty. The number are the power of the dependence
    # on the radius. The bool indicates if the parameter is used in log
    all_derived_parameters = (("radius", 1.0, False),
                              ("logL", 2.0, True),
                              ("logQ0", 2.0, True),
                              ("logQ1", 2.0, True),
                              ("logQ2", 2.0, True),
                              ("Mspec", 2.0, False),
                              ("vesc_eff", 0.5, False),
                              ("vinf_vesc", 0.5, False),
                              ("mdot_fclump", 1.5, True))

    for dpar, power, log in all_derived_parameters:
        if dpar in d_pars_err1:
            d_pars_err1 = propagate_uncertainty(d_pars_err1, dpar, new_rad,
                                                delta_radius1, power, log=log)
            d_pars_err2 = propagate_uncertainty(d_pars_err2, dpar, new_rad,
                                                delta_radius2, power, log=log)

    # mdot is done separately, because it is not a derived parameter.
    if "mdot" in pars_err1:
        pars_err1 = propagate_uncertainty(pars_err1, "mdot", new_rad,
                                          delta_radius1, 1.5, log=True)
        pars_err2 = propagate_uncertainty(pars_err2, "mdot", new_rad,
                                          delta_radius2, 1.5, log=True)

    # returns the same best_uncertainty tuple, but with updated values
    return best_model_name, bestfamily_name, pars_err1, pars_err2, d_pars_err1,\
        d_pars_err2, which_statistic

def titlepage(df, best_model: pop.Individual, runname, params_error_1sig, params_error_2sig,
    the_pdf, maxgen, nind, spectra: list[Epoch], which_sigma,
    deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars: list[set[str]]):
    """
    Make a page with best fit parameters and errors
    """

    ncrash = len(df.copy()[df['chi2'] == 999999999])
    ntot = len(df)
    perccrash = round(100.0*ncrash/ntot,1)
    minrchi2 = best_model.fitting_params['rchi2']
    nlines = len(spectra[0].get_line_names())
    nEpochs = len(spectra)
    multiplicity = len(best_model.components)

    fig, ax = plt.subplots(multiplicity + 1,2,figsize=(12.5, 6.5 + 6*multiplicity),
        gridspec_kw={'height_ratios': [0.5] + [3]*multiplicity, 'width_ratios': [2, 8]})

    # Not catch all, but catch most solution
    path_to_ga = sys.argv[0].strip("GA_analysis.py")
    if os.path.isfile(path_to_ga + 'kiwi.jpg'):
        ax[0,0].imshow(mpimg.imread(path_to_ga + 'kiwi.jpg'))

    ax[0,0].axis('off')
    ax[0,1].axis('off')


    boldtext = {'ha':'left', 'va':'top', 'weight':'bold'}
    normtext = {'ha':'left', 'va':'top'}
    offs = 0.1
    yvalmax = 0.9
    ax[0,1].text(0.0, yvalmax, 'Run name', **boldtext)
    ax[0,1].text(0.25, yvalmax, runname, **normtext)
    ax[0,1].text(0.0, yvalmax-1*offs, 'Best rchi2', **boldtext)
    ax[0,1].text(0.25, yvalmax-1*offs, str(minrchi2), **normtext)
    ax[0,1].text(0.0, yvalmax-2*offs, 'Generations', **boldtext)
    ax[0,1].text(0.25, yvalmax-2*offs, str(maxgen), **normtext)
    ax[0,1].text(0.0, yvalmax-3*offs, 'Individuals per gen', **boldtext)
    ax[0,1].text(0.25, yvalmax-3*offs, str(nind), **normtext)
    ax[0,1].text(0.0, yvalmax-4*offs, 'Total # models', **boldtext)
    ax[0,1].text(0.25, yvalmax-4*offs, str(ntot), **normtext)
    ax[0,1].text(0.0, yvalmax-5*offs, 'Crashed models', **boldtext)
    ax[0,1].text(0.25, yvalmax-5*offs, str(perccrash) + '%', **normtext)
    ax[0,1].text(0.0, yvalmax-6*offs, 'Number of lines', **boldtext)
    ax[0,1].text(0.25, yvalmax-6*offs, str(nlines), **normtext)
    ax[0, 1].text(0.0, yvalmax - 7 * offs, 'Number of Epochs', **boldtext)
    ax[0, 1].text(0.25, yvalmax - 7 * offs, str(nEpochs), **normtext)
    ax[0, 1].text(0.0, yvalmax - 8 * offs, 'Multiplicity', **boldtext)
    ax[0, 1].text(0.25, yvalmax - 8 * offs, str(multiplicity), **normtext)

    if which_sigma == 2:
        psig = params_error_2sig
        deriv_psig = deriv_params_error_2sig
    else:
        psig = params_error_1sig
        deriv_psig = deriv_params_error_1sig



    for i in range(multiplicity):
        offs = 0.02
        yvalmax = 1.0
        secndcol = 0.15
        fig_i = i+1
        ax[fig_i,0].axis('off')
        ax[fig_i,1].axis('off')
        ax[fig_i,0].text(0.0, yvalmax, 'Component: ' + str(i), **boldtext)
        ax[fig_i,1].text(0.0, yvalmax, 'Parameter', weight='bold')
        ax[fig_i,1].text(secndcol, yvalmax, 'Best', weight='bold')
        ax[fig_i,1].text(secndcol*2, yvalmax, '-' + str(which_sigma)
            + r'$\mathbf{\sigma}$', weight='bold')
        ax[fig_i,1].text(secndcol*3, yvalmax, '+' + str(which_sigma)
            + r'$\mathbf{\sigma}$', weight='bold')
        ax[fig_i,1].text(secndcol*4, yvalmax,
            r'Min (' + str(which_sigma) + r'$\mathbf{\sigma}$)', weight='bold')
        ax[fig_i,1].text(secndcol*5, yvalmax,
            r'Max (' + str(which_sigma) + r'$\mathbf{\sigma}$)', weight='bold')
        for param in best_model.components[i].template.variables.keys():
            yvalmax = yvalmax - offs
            ax[fig_i,1].text(0.0, yvalmax, param)
            ax[fig_i,1].text(secndcol, yvalmax,
                round(psig[i][param][2],3))
            ax[fig_i,1].text(secndcol*2, yvalmax,
                round(psig[i][param][2]-psig[i][param][0],3))
            ax[fig_i,1].text(secndcol*3, yvalmax,
                round(psig[i][param][1]-psig[i][param][2],3))
            ax[fig_i,1].text(secndcol*4, yvalmax,
                round(psig[i][param][0],3))
            ax[fig_i,1].text(secndcol*5, yvalmax,
                round(psig[i][param][1],3))

        yvalmax = yvalmax - offs
        for paramname in deriv_pars[i]:
            yvalmax = yvalmax - offs
            ax[fig_i,1].text(0.0, yvalmax, paramname)
            ax[fig_i,1].text(secndcol, yvalmax,
                round(deriv_psig[i][paramname][2],3))
            ax[fig_i,1].text(secndcol*2, yvalmax,
                round(deriv_psig[i][paramname][2]-deriv_psig[i][paramname][0],3))
            ax[fig_i,1].text(secndcol*3, yvalmax,
                round(deriv_psig[i][paramname][1]-deriv_psig[i][paramname][2],3))
            ax[fig_i,1].text(secndcol*4, yvalmax,
                round(deriv_psig[i][paramname][0],3))
            ax[fig_i,1].text(secndcol*5, yvalmax,
                round(deriv_psig[i][paramname][1],3))

    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf


def titlepage_latex(df, runname, params_error_1sig, params_error_2sig,
    the_pdf, param_names, maxgen, nind, linedct, which_sigma,
    deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars):
    """
    Make a page with best fit parameters and errors
    """
    plt.rcParams['text.usetex'] = True

    ncrash = len(df.copy()[df['chi2'] == 999999999])
    ntot = len(df)
    perccrash = round(100.0*ncrash/ntot,1)
    minrchi2 = round(np.min(df['rchi2']),2)
    nlines = len(linedct['name'])

    fig, ax = plt.subplots(2,2,figsize=(12.5, 12.5),
        gridspec_kw={'height_ratios': [0.5, 3], 'width_ratios': [2, 8]})

    # Not catch all, but catch most solution
    path_to_ga = sys.argv[0].strip("GA_analysis.py")
    if os.path.isfile(path_to_ga + 'kiwi.jpg'):
        ax[0,0].imshow(mpimg.imread(path_to_ga + 'kiwi.jpg'))

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    boldtext = {'ha':'left', 'va':'top', 'weight':'bold'}
    normtext = {'ha':'left', 'va':'top'}
    offs = 0.12
    yvalmax = 0.9
    ax[0,1].text(0.0, yvalmax, r'{\bf Run name}', **boldtext)
    ax[0,1].text(0.25, yvalmax, runname, **normtext)
    ax[0,1].text(0.0, yvalmax-1*offs, r'{\bf Best rchi2}', **boldtext)
    ax[0,1].text(0.25, yvalmax-1*offs, str(minrchi2), **normtext)
    ax[0,1].text(0.0, yvalmax-2*offs, r'{\bf Generations}', **boldtext)
    ax[0,1].text(0.25, yvalmax-2*offs, str(maxgen), **normtext)
    ax[0,1].text(0.0, yvalmax-3*offs, r'{\bf Individuals per gen}', **boldtext)
    ax[0,1].text(0.25, yvalmax-3*offs, str(nind), **normtext)
    ax[0,1].text(0.0, yvalmax-4*offs,r'{\bf Total number of models}',**boldtext)
    ax[0,1].text(0.25, yvalmax-4*offs, str(ntot), **normtext)
    ax[0,1].text(0.0, yvalmax-5*offs, r'{\bf Crashed models}', **boldtext)
    ax[0,1].text(0.25, yvalmax-5*offs, str(perccrash) + '\%', **normtext)
    ax[0,1].text(0.0, yvalmax-6*offs, r'{\bf Number of lines}', **boldtext)
    ax[0,1].text(0.25, yvalmax-6*offs, str(nlines), **normtext)

    if which_sigma == 2:
        psig = params_error_2sig
        deriv_psig = deriv_params_error_2sig
    else:
        psig = params_error_1sig
        deriv_psig = deriv_params_error_1sig

    table_text = r"\begin{tabular}{l|r|rr} "
    table_text += r"{\bf Parameter} & {\bf Value} & {\bf min %i$\sigma$} &" \
        r" {\bf max %i$\sigma$} \rule{0pt}{2.6ex} \\ \hline " % (which_sigma,
                                                                 which_sigma)
    for paramname in param_names:
        table_text += r"%s & $%s_{-%s}^{+%s}$ & $%s$ & $%s$ \rule{0pt}{2.6ex}" \
                      r" \\ " % (
            paramname,
            np.format_float_positional(psig[paramname][2],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][2] - psig[paramname][0],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][1] - psig[paramname][2],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][0],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][1],
                                       trim="-", precision=2))
    table_text += r"\hline "
    for paramname in deriv_pars:
        table_text += r"%s & $%s_{-%s}^{+%s}$ & $%s$ & $%s$ \rule{0pt}{2.6ex}" \
                      r" \\ " % (
fix_latex(paramname),
np.format_float_positional(deriv_psig[paramname][2], trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][2] - deriv_psig[paramname][0],
                           trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][1] - deriv_psig[paramname][2],
                           trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][0], trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][1], trim="-", precision=2))

    table_text += r"\end{tabular}"
    ax[1,1].text(0, 0.5, table_text, ha="left", va="center")

    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    # Stop using latex rendering to not mess with any other plots
    plt.rcParams['text.usetex'] = False
    return the_pdf


def fix_latex(string):
    """
    removes underscores
    """
    string = string.replace("_", " ")
    return string

def remove_outliers_iqr(df, column, k=1.3):
    if column not in df.columns:
        return df
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (df[column] >= lower) & (df[column] <= upper)
    return df[mask]

def get_relevant_param_df(populations: list[pop.Population], best_ind: pop.Individual, comp):
    fit_keys = list(best_ind.fitting_params.keys())
    comp_keys = list(best_ind.components[comp].parameters.keys())

    columns = ['gen'] + fit_keys + comp_keys
    rows = []

    for generation in populations:
        gen_id = int(generation.name)

        for ind in generation.population:
            fit_params = ind.fitting_params

            if fit_params['rchi2'] < 99999999:
                comp_params = ind.components[comp].parameters

                row = [gen_id]
                row.extend(fit_params[k] for k in fit_keys)
                row.extend(comp_params[k] for k in comp_keys)

                rows.append(row)

    param_values = pd.DataFrame(rows, columns=columns)

    # Outlier filtering
    for par in best_ind.components[comp].vrads.keys():
        param_values = remove_outliers_iqr(param_values, par)

    return param_values


def fitnessplot(df, yval, params_error_1sig, params_error_2sig,
    the_pdf, maxgen, param_names: list[str], which_cmap=plt.cm.viridis, save_jpg=False):

    """
    Plot the fitness as a function of each free parameter. This function
    can be used for plotting the P-value, 1/rchi2 of all lines combined,
    or for the fitness of individual lines (1/rchi2)
    """

    # Prepare colorbar
    cmap = which_cmap
    bounds = np.linspace(0, maxgen+1, maxgen+2)
    norm = matplotlib.colors.BoundaryNorm(bounds, int(cmap.N*0.8))

    # Set up figure dimensions and subplots
    ncols = 5
    # nrows len(param_names)+1 to ensure space for the colorbar
    nrows =int(math.ceil(1.0*(len(param_names)+1)/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, figsizefact*nrows),
        sharey=True)

    # Loop through parameters
    for i in range(ncols*nrows):
        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= len(param_names):
            ax[crow,ccol].axis('off')
            continue
        param = param_names[i]
        # Make actual plots
        ax[crow,ccol].set_title(param)
        if param == 'Gamma_Edd':
            ax[crow,ccol].set_xlim(0,1.0)
        elif param == 'vinf_vesc':
            ax[crow,ccol].set_xlim(0,10.0)
        scat0 = ax[crow,ccol].scatter(df[param], df[yval], c=df['gen'], cmap=cmap, norm=norm, s=10)

        min1sig = params_error_1sig[param][0]
        max1sig = params_error_1sig[param][1]
        min2sig = params_error_2sig[param][0]
        max2sig = params_error_2sig[param][1]
        bestfit = params_error_2sig[param][2]
        # if not save_jpg:
        ax[crow,ccol].axvline(bestfit, color='orangered', lw=1.5)
        ax[crow,ccol].axvspan(min1sig, max1sig, color='gold',
            alpha=0.70, zorder=0)
        ax[crow,ccol].axvspan(min2sig, max2sig, color='gold',
            alpha=0.25, zorder=0)
        ax[crow, ccol].set_rasterized(True)

        # Set y-labels
        if ccol == 0:
            if yval == 'P-value':
                ax[crow,ccol].set_ylabel('P-value')
            elif yval == 'fitness':
                ax[crow,ccol].set_ylabel('Fitness')
            else:
                ax[crow,ccol].set_ylabel(r'1/$\chi^2_{\rm red}$')

        ax[crow,ccol].set_ylim(-0.05*max(df[yval]), max(df[yval])*1.10)

    # Colorbar
    cbar = plt.colorbar(scat0, orientation='horizontal', ax=ax[-1,-1])
    cbar.ax.set_title('Generation')

    # Set title
    if yval in ('invrchi2', 'P-value'):
        plt.suptitle('All lines (derived parameters)')
    else:
        plt.suptitle(yval)

    # Tight layout and save plot
    plt.tight_layout()
    if nrows == 2:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.85)
    else:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.90)
    if not save_jpg:
        if yval in ('invrchi2', 'P-value'):
            the_pdf.savefig(dpi=150)
        else:
            the_pdf.savefig(dpi=100)
        plt.close()

        return the_pdf
    else:
        return fig, ax


def fitnessplot_per_parameter(df, yval, params_error_1sig, params_error_2sig,
    the_pdf, maxgen, param_names: list[str], which_cmap=plt.cm.viridis, save_jpg=False):

    """
    Plot the fitness as a function for each line for a given free parameter.
    This function can be used for plotting the P-value, 1/rchi2 of all lines
    combined, or for the fitness of individual lines (1/rchi2)
    Also plots the fitness of the given parameter for all lines,
    and the sample density.
    """

    # Prepare colorbar
    cmap = which_cmap
    bounds = np.linspace(0, maxgen + 1, maxgen + 2)
    norm = matplotlib.colors.BoundaryNorm(bounds, int(cmap.N * 0.8))

    # Set up figure dimensions and subplots
    ncols = 5
    # nrows len(param_names)+1 to ensure space for the colorbar
    nrows = int(math.ceil(1.0 * (len(param_names) + 1) / ncols))
    nrows = max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(figsizefact * ncols, figsizefact * nrows),
                           sharey=True)

    # Loop through parameters
    for i in range(ncols * nrows):
        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= len(param_names):
            ax[crow, ccol].axis('off')
            continue
        param = param_names[i]
        # Make actual plots
        ax[crow, ccol].set_title(param)
        if param == 'Gamma_Edd':
            ax[crow, ccol].set_xlim(0, 1.0)
        elif param == 'vinf_vesc':
            ax[crow, ccol].set_xlim(0, 10.0)
        scat0 = ax[crow, ccol].scatter(df[yval], df[param], c=df['gen'], cmap=cmap, norm=norm, s=10)

        min1sig = params_error_1sig[yval][0]
        max1sig = params_error_1sig[yval][1]
        min2sig = params_error_2sig[yval][0]
        max2sig = params_error_2sig[yval][1]
        bestfit = params_error_2sig[yval][2]
        # if not save_jpg:
        ax[crow, ccol].axvline(bestfit, color='orangered', lw=1.5)
        ax[crow, ccol].axvspan(min1sig, max1sig, color='gold',
                               alpha=0.70, zorder=0)
        ax[crow, ccol].axvspan(min2sig, max2sig, color='gold',
                               alpha=0.25, zorder=0)
        ax[crow, ccol].set_rasterized(True)

        # Set y-labels
        if ccol == 0:
            if yval == 'P-value':
                ax[crow, ccol].set_ylabel('P-value')
            elif yval == 'fitness':
                ax[crow, ccol].set_ylabel('Fitness')
            else:
                ax[crow, ccol].set_ylabel(r'1/$\chi^2_{\rm red}$')

        ax[crow, ccol].set_ylim(-0.05 * max(df[param]), max(df[param]) * 1.10)

    # Colorbar
    cbar = plt.colorbar(scat0, orientation='horizontal', ax=ax[-1, -1])
    cbar.ax.set_title('Generation')

    # Set title
    if yval in ('invrchi2', 'P-value'):
        plt.suptitle('All lines (derived parameters)')
    else:
        plt.suptitle(yval)

    # Tight layout and save plot
    plt.tight_layout()
    if nrows == 2:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.85)
    else:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.90)
    if not save_jpg:
        if yval in ('invrchi2', 'P-value'):
            the_pdf.savefig(dpi=150)
        else:
            the_pdf.savefig(dpi=100)
        plt.close()

        return the_pdf
    else:
        return fig, ax



def unpack_tarfiles(savedmoddir, comp, mod):
    name = mod.name
    moddir = savedmoddir + name.split('_')[0] + '/' + name.split('_')[1] + '/'
    if not os.path.isdir(moddir + 'combined'):
        fw.mkdir(moddir + 'combined')
        fw.untar(moddir + 'combined.tar.gz', moddir + 'combined' + '/.')
    if comp:
        params = pd.read_csv(moddir + 'combined/params.csv')
        for index, row in params.iterrows():
            newloc = moddir + str(index) + '/'
            name = row['run_id']
            compparts = name.split('_')
            compname = str(index)
            if len(compparts) == 4:
                compname += '_' + compparts[-1]
            if not os.path.isdir(newloc):
                fw.mkdir(newloc)
                fw.untar(moddir + compname + '.tar.gz', newloc)


def prep_plot_data(spectrum, best_mod_location, plotmoddirlist, plotlineprofdir, active_line):
    # Read data per line
    wave_tmp, flux_tmp, err_tmp = spectrum.get_line_data(active_line)

    # Read profile of best fitting model
    best_prof_file = best_mod_location + '/combined/profiles/' + active_line + '.prof.comb'
    bestmodwave, bestmodflux = np.loadtxt(best_prof_file, unpack=True)

    # Read profiles of best fitting family of models
    lineflux_min = np.copy(bestmodflux)
    lineflux_max = np.copy(bestmodflux)
    if len(plotmoddirlist) > 0:
        for asig_mod in plotmoddirlist:
            fam_prof_file = os.path.join(asig_mod, active_line + '.prof.comb')
            smwave, smflux = np.loadtxt(fam_prof_file, unpack=True)

            np.minimum(lineflux_min, smflux, out=lineflux_min)
            np.maximum(lineflux_max, smflux, out=lineflux_max)
    lineprof_arr = np.array([bestmodwave, bestmodflux, lineflux_min,
                             lineflux_max]).T
    lineprof_arr_head = 'wave bestflux minflux maxflux'
    np.savetxt(plotlineprofdir + active_line + '.txt', lineprof_arr,
               header=lineprof_arr_head)

    return wave_tmp, flux_tmp, err_tmp, bestmodwave, bestmodflux, lineflux_min, lineflux_max


def lineprofiles(pool, spectrum: Epoch, savedmoddir,
    best_mod_location, bestfamily, the_pdf, plotlineprofdir,
    extra_fwmod, extra_mod, save_jpg=False):
    """
    Create plot with line profiles of best fitting models.
    In the background, plot the data.
    """
    active_lines = spectrum.get_active_lines()
    nlines = len(active_lines)

    # Untar best fitting models.
    plotmoddirlist = []
    for mod in bestfamily:
        name = mod.name
        part1, part2 = name.split('_')[:2]
        moddir = os.path.join(savedmoddir, part1, part2, 'combined')
        plotmoddirlist.append(os.path.join(moddir, 'profiles'))

    # Set up figure dimensions and subplots
    ncols = 5
    nrows =int(math.ceil(1.0*nlines/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, 0.7*figsizefact*nrows))

    prep_data_func = functools.partial(prep_plot_data, spectrum, best_mod_location, plotmoddirlist, plotlineprofdir)
    prep_data = pool.map(prep_data_func, active_lines)

    # Loop through parameters
    for i in range(ncols*nrows):
        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= nlines:
            ax[crow,ccol].axis('off')
            continue

        active_line = active_lines[i]

        wave_tmp, flux_tmp, err_tmp, bestmodwave, bestmodflux, lineflux_min, lineflux_max = prep_data[i]

        # Make actual plots
        ax[crow,ccol].set_title(active_line)
        ax[crow,ccol].axhline(1.0, color='black', lw=0.8)
        ax[crow,ccol].errorbar(wave_tmp, flux_tmp, yerr=err_tmp,
            fmt='o', color='black', ms=0)
        ax[crow,ccol].fill_between(bestmodwave, lineflux_min, lineflux_max,
            color='#8cd98c', alpha=0.7)
        ax[crow,ccol].plot(bestmodwave, bestmodflux, color='#1ca641', lw=2.4,
            alpha=1.0)
        #ax[crow,ccol].set_xlim(linedct['left'][i], linedct['right'][i])
        # ax[crow,ccol].set_ylim(*ax[crow,ccol].get_ylim())
        ax[crow,ccol].set_ylim(np.min([np.min(flux_tmp) * 0.95, np.min(lineflux_min) * 0.95]),
                               np.max([np.max(flux_tmp) * 1.05, np.max(lineflux_max) * 1.05]))

        # plot an extra fastwind model
        if not extra_fwmod == '/':
            extramfile = extra_fwmod + active_lines[i] + '.prof'
            if os.path.isfile(extramfile):
                em_wave, em_flux = np.loadtxt(extramfile, unpack=True)
                ax[crow,ccol].plot(em_wave, em_flux, color='dodgerblue', lw=2.4)
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='red',
                    lw=2.4,alpha=1.0)
            else:
                print(extramfile, 'not found')
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='orangered',
                    lw=2.4, alpha=1.0)

        # plot another
        if not extra_mod == '':
            if os.path.isfile(extra_mod):
                em_wave, em_flux = np.loadtxt(extra_mod, unpack=True)
                ax[crow,ccol].plot(em_wave, em_flux, color='blue', lw=2.4)
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='orangered',
                    lw=2.4, alpha=1.0)
            else:
                print(extra_mod, 'not found')

    # Tight layout and save plot
    plt.tight_layout()

    if not save_jpg:
        the_pdf.savefig(dpi=150)
        plt.close()
        return the_pdf
    else:
        return fig, ax

def prep_composite_data(spectrum, best_mod_location, plotmoddirlist, compmoddirlist, multiplicity, best_individual, bestfamily, compsdir, active_line):
    c = 299792.458  # km/s
    spectrum_name = spectrum.name.split('/')[-1]
    # Read data per line
    wave_tmp, flux_tmp, err_tmp = spectrum.get_line_data(active_line)

    # Read profile of best fitting model
    best_prof_file = best_mod_location + '/combined/profiles/' + active_line + '.prof.comb'
    bestmodwave, bestmodflux = np.loadtxt(best_prof_file, unpack=True)

    # Read profiles of best fitting family of models
    lineflux_min = np.copy(bestmodflux)
    lineflux_max = np.copy(bestmodflux)
    if len(plotmoddirlist) > 0:
        for asig_mod in plotmoddirlist:
            fam_prof_file = asig_mod + active_line + '.prof.comb'
            smwave, smflux = np.loadtxt(fam_prof_file, unpack=True)

            np.minimum(lineflux_min, smflux, out=lineflux_min)
            np.maximum(lineflux_max, smflux, out=lineflux_max)

    # Read profiles of individual components of the best model
    bestmodflux_comps, lineflux_min_comps, lineflux_max_comps = [], [], []
    best_fluxes = []
    best_conts = []

    for comp in range(multiplicity):
        vrad = best_individual.components[comp].parameters[spectrum_name]
        prof_file = best_mod_location + '/' + str(comp) + '/profiles/' + active_line + '.prof.fin'

        wave, flux, cont = np.loadtxt(prof_file, unpack=True)

        wave_shift = wave * (1.0 + vrad / c)

        flux_i = np.interp(bestmodwave, wave_shift, flux)
        cont_i = np.interp(bestmodwave, wave_shift, cont)

        best_fluxes.append(flux_i)
        best_conts.append(cont_i)

    # Compute continuum sum
    cont_sum = np.sum(best_conts, axis=0)

    for comp in range(multiplicity):
        vrad = best_individual.components[comp].parameters[spectrum_name]
        flux = best_fluxes[comp]
        cont = best_conts[comp]

        scale = cont / cont_sum
        flux = scale * flux + (1 - scale)

        flux_min = flux.copy()
        flux_max = flux.copy()

        # Process model family
        if compmoddirlist:
            for asig_mod, mod in zip(compmoddirlist, bestfamily):
                fam_file = asig_mod[comp] + active_line + '.prof.fin'

                smwave, smflux, smcont = np.loadtxt(fam_file, unpack=True)

                smwave = smwave * (1.0 + vrad / c)

                smflux = np.interp(bestmodwave, smwave, smflux)
                smcont = np.interp(bestmodwave, smwave, smcont)

                scale = smcont / cont_sum
                smflux = scale * smflux + (1 - scale)

                # Update without temporary arrays
                np.minimum(flux_min, smflux, out=flux_min)
                np.maximum(flux_max, smflux, out=flux_max)

        bestmodflux_comps.append(flux)
        lineflux_min_comps.append(flux_min)
        lineflux_max_comps.append(flux_max)

        lineprof_arr = np.array([bestmodwave, flux, flux_min,
                                 flux_max]).T
        lineprof_arr_head = 'wave bestflux minflux maxflux'
        np.savetxt(compsdir + str(comp) + '/lineprofs/' + active_line + '.txt', lineprof_arr,
                   header=lineprof_arr_head)

    return wave_tmp, flux_tmp, err_tmp, bestmodwave, bestmodflux, bestmodflux_comps, lineflux_min, lineflux_max, lineflux_min_comps, lineflux_max_comps


def composite_lineprofiles(pool, best_individual: pop.Individual, spectrum: Epoch, savedmoddir,
    best_mod_location, bestfamily: list[pop.Individual], the_pdf,
    extra_fwmod, extra_mod, multiplicity, compsdir, save_jpg=False):
    """
    Create plot with line profiles of best fitting models.
    In the background, plot the data.
    """
    active_lines = spectrum.get_active_lines()
    nlines = len(active_lines)

    # Untar best fitting models and their components
    plotmoddirlist = []
    compmoddirlist = []
    for mod in bestfamily:
        name = mod.name
        moddir = savedmoddir + name.split('_')[0] + '/' + name.split('_')[1] + '/'
        plotmoddirlist.append(moddir + 'combined/profiles/')
        complist = []
        for index in range(multiplicity):
            newloc = moddir + str(index) + '/'
            complist.append(newloc + 'profiles/')
        compmoddirlist.append(complist)

    # Set up figure dimensions and subplots
    ncols = 5
    nrows =int(math.ceil(1.0*nlines/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, 0.7*figsizefact*nrows))

    prep_data_func = functools.partial(prep_composite_data, spectrum, best_mod_location, plotmoddirlist, compmoddirlist, multiplicity, best_individual, bestfamily, compsdir)

    prep_data = pool.map(prep_data_func, active_lines)

    # Loop through parameters
    for i in range(ncols*nrows):

        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= nlines:
            ax[crow,ccol].axis('off')
            continue

        active_line = active_lines[i]
        wave_tmp, flux_tmp, err_tmp, bestmodwave, bestmodflux, bestmodflux_comps, lineflux_min, lineflux_max, lineflux_min_comps, lineflux_max_comps = prep_data[i]

        # Make actual plots
        ax[crow,ccol].set_title(active_line)
        ax[crow,ccol].axhline(1.0, color='black', lw=0.8)
        ax[crow,ccol].errorbar(wave_tmp, flux_tmp, yerr=err_tmp,
            fmt='o', color='black', ms=0)
        ax[crow,ccol].fill_between(bestmodwave, lineflux_min, lineflux_max,
            color='#8cd98c', alpha=0.7)
        ax[crow,ccol].plot(bestmodwave, bestmodflux, color='#1ca641', lw=2.4,
            alpha=1.0)
        ax[crow,ccol].set_ylim(np.min([np.min(flux_tmp) * 0.95, np.min(lineflux_min) * 0.95]),
                           np.max([np.max(flux_tmp) * 1.05, np.max(lineflux_max) * 1.05]))

        colors = ['r', 'b', 'y', 'g']
        for comp in range(multiplicity):
            ax[crow, ccol].fill_between(bestmodwave, lineflux_min_comps[comp], lineflux_max_comps[comp],
                                        color=colors[comp], alpha=0.5)
            ax[crow, ccol].plot(bestmodwave, bestmodflux_comps[comp], color=colors[comp], lw=2.4,
                                alpha=1.0)

        # plot an extra fastwind model
        if not extra_fwmod == '/':
            extramfile = extra_fwmod + active_lines[i] + '.prof'
            if os.path.isfile(extramfile):
                em_wave, em_flux = np.loadtxt(extramfile, unpack=True)
                ax[crow,ccol].plot(em_wave, em_flux, color='dodgerblue', lw=2.4)
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='red',
                    lw=2.4,alpha=1.0)
            else:
                print(extramfile, 'not found')
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='orangered',
                    lw=2.4, alpha=1.0)

        # plot another
        if not extra_mod == '':
            if os.path.isfile(extra_mod):
                em_wave, em_flux = np.loadtxt(extra_mod, unpack=True)
                ax[crow,ccol].plot(em_wave, em_flux, color='blue', lw=2.4)
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='orangered',
                    lw=2.4, alpha=1.0)
            else:
                print(extra_mod, 'not found')

    # Tight layout and save plot
    plt.tight_layout()

    if not save_jpg:
        the_pdf.savefig(dpi=150)
        plt.close()
        return the_pdf
    else:
        return fig, ax

def correlationplot(the_pdf, dfs, corrpars, best_individual):
    """
    Create a correlation plot of the parameters in the list corrpars.
    """
    mult = len(dfs)
    temp_dfs = []
    for df in dfs:
        df_sort = df.sort_values(by=['invrchi2'])
        temp_dfs.append(df_sort)
    dfs = temp_dfs

    # Set up figure dimensions and subplots
    ncols = len(corrpars)*mult
    nrows = ncols
    hratios = 30*np.ones(ncols)
    wratios = 30*np.ones(ncols)
    hratios[0] = 1.0
    wratios[-1] = 1.0
    figsizefact = 2.0
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, figsizefact*nrows),
            sharex='col', sharey='row',
            gridspec_kw={'height_ratios': hratios, 'width_ratios': wratios})


    if ncols == 1:
        ax = np.array([[ax]])
    elif ncols == 0:
        plt.close()
        return the_pdf

    # Loop through parameters to create correlation plot
    pairlist = []
    for ccol in range(int(ncols/mult)):
        for crow in range(int(nrows/mult)):
            for i in range(mult):
                for j in range(mult):
                    pc1 = corrpars[ccol]
                    pc2 = corrpars[crow]
                    col = ccol*mult + i
                    row = crow*mult + j
                    if col >= row:
                        ax[row,col].axis('off')
                    else:
                        ax[row,col].scatter(dfs[i][pc1], dfs[j][pc2],
                            c=dfs[i]['invrchi2'],s=10, rasterized=True)

                    ax[row,col].set_xlim(np.min(dfs[i][pc1]), np.max(dfs[i][pc1]))
                    ax[row,col].set_ylim(np.min(dfs[j][pc2]), np.max(dfs[j][pc2]))
    # Label axes
    for i in range(0, int(ncols/mult)-1):
        for comp in range(mult):
            ax[-1,i*mult + comp].set_xlabel(corrpars[i]+"_"+str(comp))
            ax[(i+1)*mult + comp, 0].set_ylabel(corrpars[i+1]+"_"+str(comp))

    # Tight layout and save plot
    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf

def get_fwmaxtime(controlfile):
    dct = fw.read_control_pars(controlfile)
    timeout = dct['fw_timeout']
    return timeout

def binary_fw_performance(the_pdf, df, controlfile):
    """
        Show maximum interations, convergence and run time of FW models.
        """

    # Pick up fastwind timeout to assign a number to the runs that ran to max
    fw_timeout = get_fwmaxtime(controlfile)
    fw_timeout_min = 1.0 * fw_timeout / 60.0
    df.loc[(df['cputime_0'] == 99999.9), 'cputime_0'] = fw_timeout
    df.loc[(df['cputime_1'] == 99999.9), 'cputime_1'] = fw_timeout - df['cputime_0']

    # Only consider models that can generate line profiles
    df = df[df['chi2'] < 999999999]
    df = df[df['maxcorr_0'] > 0.0]
    df = df[df['maxcorr_1'] > 0.0]
    df['cputime_min_0'] = 1.0 * df['cputime_0'].values / 60.0
    df['cputime_min_1'] = 1.0 * df['cputime_1'].values / 60.0

    nb = 101
    bins_maxit = np.linspace(0, 100, nb)
    bins_maxco = np.linspace(-3, 1.5, nb)
    bins_ticpu = np.linspace(0, fw_timeout_min, nb)

    for comp in range(2):
        fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
        ax[0, 0].hist(df['maxit'+'_'+str(comp)], bins_maxit,
                      color='#2b0066', alpha=0.7)
        ax[0, 1].hist(np.log10(df['maxcorr'+'_'+str(comp)]), bins_maxco,
                      color='#009c60', alpha=0.7)
        ax[0, 2].hist(df['cputime_min'+'_'+str(comp)], bins_ticpu,
                      color='#b5f700', alpha=0.7)

        ax[0, 0].set_xlabel('Maximum iteration component '+str(comp))
        ax[0, 1].set_xlabel('log(Maximum correction) component '+str(comp))
        ax[0, 2].set_xlabel('CPU-time (minutes) component '+str(comp))
        ax[0, 0].set_ylabel('Count')
        ax[0, 1].set_ylabel('Count')
        ax[0, 2].set_ylabel('Count')

        sct1 = ax[1, 0].scatter(np.log10(df['maxcorr'+'_'+str(comp)]), df['maxit'+'_'+str(comp)],
                                s=6, c=df['cputime'+'_'+str(comp)] / 60.0, rasterized=True)
        ax[1, 0].set_xlabel('log(Maximum correction) component '+str(comp))
        ax[1, 0].set_ylabel('Maximum iteration component '+str(comp))
        cbar1 = plt.colorbar(sct1, ax=ax[1, 0])
        cbar1.ax.set_title(r'CPU-time (min)', fontsize=9)

        sct2 = ax[1, 1].scatter(np.log10(df['maxcorr'+'_'+str(comp)]), df['cputime'+'_'+str(comp)] / 60.0,
                                s=6, c=df['maxit'+'_'+str(comp)], rasterized=True)
        ax[1, 1].set_xlabel('log(Maximum correction) component '+str(comp))
        ax[1, 1].set_ylabel('CPU-time (minutes) component '+str(comp))
        cbar2 = plt.colorbar(sct2, ax=ax[1, 1])
        cbar2.ax.set_title(r'Max. iteration', fontsize=9)

        sct3 = ax[1, 2].scatter(df['cputime'+'_'+str(comp)] / 60.0, df['maxit'+'_'+str(comp)],
                                s=4, c=np.log10(df['maxcorr'+'_'+str(comp)]), rasterized=True)
        ax[1, 2].set_xlabel('CPU-time (minutes) component '+str(comp))
        ax[1, 2].set_ylabel('Maximum iteration component '+str(comp))
        cbar3 = plt.colorbar(sct3, ax=ax[1, 2])
        cbar3.ax.set_title(r'log(Max. corr.)', fontsize=9)

        # Tight layout and save plot
        plt.tight_layout()
        the_pdf.savefig(dpi=150)
        plt.close()

    return the_pdf

def fw_performance(the_pdf, df, controlfile):
    """
    Show maximum interations, convergence and run time of FW models.
    """

    # Pick up fastwind timeout to assign a number to the runs that ran to max
    fw_timeout = get_fwmaxtime(controlfile)
    fw_timeout_min = 1.0*fw_timeout/60.0
    df.loc[(df['cputime'] == 99999.9), 'cputime'] = fw_timeout

    # Only consider models that can generate line profiles
    df = df[df['chi2'] < 999999999]
    df = df[df['maxcorr'] > 0.0]
    df['cputime_min'] = 1.0 * df['cputime'].values / 60.0

    nb = 101
    bins_maxit = np.linspace(0, 100, nb)
    bins_maxco = np.linspace(-3, 1.5, nb)
    bins_ticpu = np.linspace(0, fw_timeout_min, nb)

    fig, ax = plt.subplots(2,3, figsize=(12,6.5))
    ax[0,0].hist(df['maxit'], bins_maxit,
        color='#2b0066', alpha=0.7)
    ax[0,1].hist(np.log10(df['maxcorr']), bins_maxco,
        color='#009c60', alpha=0.7)
    ax[0,2].hist(df['cputime_min'], bins_ticpu,
        color='#b5f700', alpha=0.7)

    ax[0,0].set_xlabel('Maximum iteration')
    ax[0,1].set_xlabel('log(Maximum correction)')
    ax[0,2].set_xlabel('CPU-time (minutes)')
    ax[0,0].set_ylabel('Count')
    ax[0,1].set_ylabel('Count')
    ax[0,2].set_ylabel('Count')

    sct1 = ax[1,0].scatter(np.log10(df['maxcorr']), df['maxit'],
        s=6, c=df['cputime']/60.0, rasterized=True)
    ax[1,0].set_xlabel('log(Maximum correction)')
    ax[1,0].set_ylabel('Maximum iteration')
    cbar1 = plt.colorbar(sct1, ax=ax[1,0])
    cbar1.ax.set_title(r'CPU-time (min)', fontsize=9)

    sct2 = ax[1,1].scatter(np.log10(df['maxcorr']), df['cputime']/60.0,
        s=6, c=df['maxit'], rasterized=True)
    ax[1,1].set_xlabel('log(Maximum correction)')
    ax[1,1].set_ylabel('CPU-time (minutes)')
    cbar2 = plt.colorbar(sct2, ax=ax[1,1])
    cbar2.ax.set_title(r'Max. iteration', fontsize=9)

    sct3 = ax[1,2].scatter(df['cputime']/60.0, df['maxit'],
        s=4, c=np.log10(df['maxcorr']), rasterized=True)
    ax[1,2].set_xlabel('CPU-time (minutes)')
    ax[1,2].set_ylabel('Maximum iteration')
    cbar3 = plt.colorbar(sct3, ax=ax[1,2])
    cbar3.ax.set_title(r'log(Max. corr.)', fontsize=9)

    # Tight layout and save plot
    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf

def convergence(the_pdf, populations: list[pop.Population], npspec,
    paramspaces: list[pop.Template], deriv_pars, comp=0):
    mult = len(paramspaces)

    evol_list_best = dict()
    evol_list_1sig_up = dict()
    evol_list_1sig_down = dict()
    evol_list_2sig_up = dict()
    evol_list_2sig_down = dict()
    for apar in paramspaces[comp].variables.keys():
        evol_list_best[apar] = []
        evol_list_1sig_up[apar] = []
        evol_list_1sig_down[apar] = []
        evol_list_2sig_up[apar] = []
        evol_list_2sig_down[apar] = []
    best_so_far = populations[0].population[0]
    for popul in populations:
        # Compute uncertainties
        best_uncertainty, n1sig, n2sig = get_local_uncertainties(popul, npspec,
            paramspaces, deriv_pars, best_so_far, incl_deriv=False)

        # Unpack all computed values
        best_so_far, bestfamily, params_error_1sig, params_error_2sig, which_statistic = best_uncertainty
        for par in paramspaces[comp].variables.keys():
            evol_list_best[par].append(params_error_1sig[comp][par][2])
            evol_list_1sig_up[par].append(params_error_1sig[comp][par][1])
            evol_list_1sig_down[par].append(params_error_1sig[comp][par][0])
            evol_list_2sig_up[par].append(params_error_2sig[comp][par][1])
            evol_list_2sig_down[par].append(params_error_2sig[comp][par][0])

    gens = range(int(populations[-1].name)-len(populations), int(populations[-1].name))
    # fig, ax = plt.subplots(1, len(param_names))
    # Set up figure dimensions and subplots
    ncols = 3
    nrows =int(math.ceil(1.0*(len(paramspaces[comp].variables.keys())/ncols)))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 4.0
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, 0.3*figsizefact*nrows), sharex=True)

    # Loop through parameters
    for i in range(ncols*nrows):

        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1
        if crow == nrows -1:
            ax[crow,ccol].set_xlabel('Generation')

        if i >= len(paramspaces[comp].variables.keys()):
            ax[crow,ccol].axis('off')
            continue

        param = list(paramspaces[comp].variables.keys())[i]

        ax[crow,ccol].plot(gens, evol_list_best[param], color='red')
        ax[crow,ccol].fill_between(gens, evol_list_1sig_down[param],
            evol_list_1sig_up[param], color='gold', alpha=0.70)
        ax[crow,ccol].fill_between(gens, evol_list_2sig_down[param],
            evol_list_2sig_up[param], color='gold', alpha=0.25)
        ax[crow,ccol].set_ylim(float(paramspaces[comp].variables[param][0]), float(paramspaces[comp].variables[param][1]))
        ax[crow,ccol].set_ylabel(param)

    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf

def save_bestvals(best_model_comp: pop.Component, deriv_pars, params_error_1sig, params_error_2sig,
    deriv_params_error_1sig, deriv_params_error_2sig, savebest_txt):
    """
    Save best fit parameters and errors to text file
    """

    if os.path.isfile(savebest_txt):
        os.remove(savebest_txt)

    write_lines = []
    rv = 4
    lj = 10
    lj0 = 15
    for apar in list(best_model_comp.template.variables.keys()):
        bestfit = params_error_2sig[apar][2]
        low1sig = str(round(bestfit - params_error_1sig[apar][0], rv))
        up1sig = str(round(params_error_1sig[apar][1] - bestfit, rv))
        low2sig = str(round(bestfit - params_error_2sig[apar][0], rv))
        up2sig = str(round(params_error_2sig[apar][1] - bestfit, rv))
        bestfit = str(round(bestfit,rv))
        savestr = ((apar).ljust(lj0) + ' ' + bestfit.ljust(lj) + ' '
            + low1sig.ljust(lj) + ' ' + up1sig.ljust(lj) + ' '
            + low2sig.ljust(lj) + ' ' + up2sig.ljust(lj))
        write_lines.append(savestr)

    for apar in deriv_pars:
        bestfit = deriv_params_error_2sig[apar][2]
        low1sig = str(round(bestfit - deriv_params_error_1sig[apar][0], rv))
        up1sig = str(round(deriv_params_error_1sig[apar][1] - bestfit, rv))
        low2sig = str(round(bestfit - deriv_params_error_2sig[apar][0], rv))
        up2sig = str(round(deriv_params_error_2sig[apar][1] - bestfit, rv))
        bestfit = str(round(bestfit,rv))
        savestr = ((apar).ljust(lj0) + ' ' + bestfit.ljust(lj) + ' '
            + low1sig.ljust(lj) + ' ' + up1sig.ljust(lj) + ' '
            + low2sig.ljust(lj) + ' ' + up2sig.ljust(lj))
        write_lines.append(savestr)

    with open(savebest_txt, 'a') as myfile:
        myfile.write('# Parameter'.ljust(lj0) + ' best'.ljust(lj+1) +
            ' low1sig'.ljust(lj+1) + ' up1sig'.ljust(lj+1) +
            ' low2sig'.ljust(lj+1) + ' up2sig' + '\n')
        for aline in write_lines:
            myfile.write(aline)
            myfile.write('\n')

def save_parameters(param_df, component_directory):
    for i in range(len(param_df)):
        param_df[i].to_csv(component_directory + str(i) + '/parameters.csv', index=False)
