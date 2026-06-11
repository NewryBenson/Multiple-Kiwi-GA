import datetime
import itertools
import os
import sys
import tarfile
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import math
import glob

from epoch import Epoch

from numpy import ndarray

from scipy import interpolate
import broaden as br
import population as pop
from scipy.optimize import least_squares

if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def check_indir(indir: str) -> bool:
    """Check if the input directory exits"""
    if not os.path.isdir(indir):
        print("No input directory found!")
        print("I was searching here: " + indir)
        print("Exiting")
        return False
    return True


def make_file_dict(indir: str, outdir: str):
    """Make a dictionary of meta files."""

    #subdirectories
    componentdir = indir + 'components/'
    epochdir = indir + 'epochs/'

    # File names of input files
    linelistfile = 'line_list.txt'
    paramspacecomponentfiles = os.listdir(componentdir)
    paramspacecomponentfiles.sort()
    radinfofile = 'radius_info.txt'
    defvalfile = 'defaults_fastwind.txt'
    epochfiles = os.listdir(epochdir)
    epochfiles.sort()
    ctrlfile = 'control.txt'

    # File names of output files
    chi2file = 'chi2.csv'
    precomp = 'check_precomputed.csv'
    mutationfile = 'mutation_by_gen.txt'
    charblimfile = 'charbonneau_limits.txt'
    bestchi2file = 'best_chi2.txt'
    paramspacefile_out = 'parameter_space.txt'
    genvarfile_out = 'genetic_variety.txt'

    # File names of files for run continuation
    # These are copies that contain only fully completed generations
    chi2_contfile = 'chi2_cont.csv'
    precomp_contfile = 'precomp_cont.csv'
    generation_contfile = 'savegen_cont.pkl'
    fitnesses_contfile = 'savefitness_cont.txt'
    redchi2_contfile = 'redchi2s_cont.txt'

    dct = {}

    dct["linelist_in"] = indir + linelistfile
    dct["paramspaces_in"] = [componentdir + component for component in paramspacecomponentfiles]
    dct["radinfo_in"] = indir + radinfofile
    dct["defvals_in"] = indir + defvalfile
    dct["epochs_in"] = [epochdir + epoch for epoch in epochfiles]
    dct["control_in"] = indir + ctrlfile

    dct["chi2_out"] = outdir + chi2file
    dct["precomp_out"] = outdir + precomp
    dct["mutation_out"] = outdir + mutationfile
    dct["charblim_out"] = outdir + charblimfile
    dct["bestchi2_out"] = outdir + bestchi2file
    dct["paramspace_out"] = outdir + paramspacefile_out
    dct["genvar_out"] = outdir + genvarfile_out

    dct["chi2_cont"] = outdir + chi2_contfile
    dct["precomp_cont"] = outdir + precomp_contfile
    dct["gen_cont"] = outdir + generation_contfile
    dct["fit_cont"] = outdir + fitnesses_contfile
    dct["redchi_cont"] = outdir + redchi2_contfile

    return dct


def mkdir(path: str) -> None:
    """Create a directory"""
    os.makedirs(path, exist_ok=True)


def tar(src_dir, tar_path):
    """Tar a directory from src_dir to tar_path"""
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(src_dir, '.')


def untar(tar_path, dst_dir):
    """Untar a directory from tar_path to dst_dir"""
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(path=dst_dir)


def init_setup(outdir: str) -> tuple[str, str, str, str]:
    """ Setup the structure of the output directory
    in the subdirectory that is used for the run, and
    return the paths to the created directories.
    """
    rundir = outdir + 'run/'
    savedir = outdir + 'saved/'
    indir = outdir + 'input_copy/'
    for adir in (outdir, rundir, savedir, indir):
        mkdir(adir)
    return outdir, rundir, savedir, indir


def copy_input(adict, indir: str) -> None:
    """Copy the input directory to the output directory.
    This is not strictly necessary, but when looking at the
    output it is nice to have the input also at hand.
    """
    for key in adict:
        if "_in" in key:
            if isinstance(adict[key], str):
                shutil.copy(adict[key], indir)
    mkdir(indir + "components/")
    mkdir(indir + "epochs/")
    for epoch in adict["epochs_in"]:
        shutil.copy(epoch, indir + "epochs/")
    for component in adict["paramspaces_in"]:
        shutil.copy(component, indir + "components/")


def read_control_pars(control_source: str):
    """ Read the control parameters from text file """

    # Read values from file and put into dictionary
    keys, vals = np.genfromtxt(control_source, dtype=str, comments='#').T
    ctrldct = {str(k): str(v) for k, v in zip(keys, vals)}

    # Convert the numeric files to integers or floats
    ctrldct["nind"] = int(ctrldct["nind"])
    ctrldct["ngen"] = int(ctrldct["ngen"])
    ctrldct["f_gen1"] = float(ctrldct["f_gen1"])
    ctrldct["ratio_po"] = float(ctrldct["ratio_po"])
    ctrldct["f_parent"] = float(ctrldct["f_parent"])
    ctrldct["p_value"] = float(ctrldct["p_value"])
    ctrldct["clone_fraction"] = float(ctrldct["clone_fraction"])
    ctrldct["w_gauss_br"] = float(ctrldct["w_gauss_br"])
    ctrldct["w_gauss_na"] = float(ctrldct["w_gauss_na"])
    ctrldct["b_gauss_br"] = float(ctrldct["b_gauss_br"])
    ctrldct["b_gauss_na"] = float(ctrldct["b_gauss_na"])
    ctrldct["mut_rate_na"] = float(ctrldct["mut_rate_na"])
    ctrldct["mut_rate_init"] = float(ctrldct["mut_rate_init"])
    ctrldct["mut_rate_min"] = float(ctrldct["mut_rate_min"])
    ctrldct["mut_rate_max"] = float(ctrldct["mut_rate_max"])
    ctrldct["mut_rate_factor"] = float(ctrldct["mut_rate_factor"])
    ctrldct["pure_reinsert_min"] = float(ctrldct["pure_reinsert_min"])
    ctrldct["fit_cutoff_min_charb"] = float(ctrldct["fit_cutoff_min_charb"])
    ctrldct["fit_cutoff_max_charb"] = float(ctrldct["fit_cutoff_max_charb"])
    ctrldct["cutoff_increase_genv"] = float(ctrldct["cutoff_increase_genv"])
    ctrldct["cutoff_decrease_genv"] = float(ctrldct["cutoff_decrease_genv"])

    ctrldct["fw_timeout"] = int(''.join(filter(str.isdigit, ctrldct["fw_timeout"]))) * 60
    ctrldct["program_time_limit"] = int(''.join(filter(str.isdigit, ctrldct["program_time_limit"]))) * 60

    n_parent = ctrldct["nind"] * ctrldct["ratio_po"]
    ctrldct["n_keep_parent"] = math.ceil(n_parent * ctrldct["f_parent"])
    f_keep_offspring = ctrldct["f_parent"] * ctrldct["ratio_po"]
    ctrldct["n_keep_offspring"] = n_parent - ctrldct["n_keep_parent"]

    return ctrldct


def prepare_output_files(adict, cont_tf: bool) -> None:
    """Move the _cont files to chi2 and duplicate files if the run is
    continuing another run, otherwise remove the old output files,
    if any are present.
    """

    if cont_tf:
        shutil.copy(adict["chi2_cont"], adict["chi2_out"])
        shutil.copy(adict["precomp_cont"], adict["precomp_out"])
    else:
        for key in adict:
            if "_out" in key and os.path.isfile(adict[key]):
                os.remove(adict[key])


def read_linelist(theflinelist: str)  -> tuple[ndarray, ndarray]:
    """Read the line_list file and output numpy arrays"""
    llist_names = np.genfromtxt(theflinelist, dtype='str').T[0]
    llist_params = np.genfromtxt(theflinelist, dtype='float').T[1:]
    return llist_names, llist_params


def read_data(linelist: str, epochs: list[str]) -> tuple[ndarray, ndarray, list[Epoch], ndarray]:
    """Based on the line_list info, select the right data
    from the spectrum, and output data, line names and resolution.
    Each line in renormalised with the values found in the linelist

    Input:
    - linelist file (line names should be as in FORMAL_INPUT)
    - file with normalised spectrum

    Output:
    - array with line names
    - array with resolution of each line
    - array with per line a an array of arrays containing per line
        - wavelength
        - (renomalised) flux
        - errors on the flux
    """

    names, llp = read_linelist(linelist)
    res, lbound, rbound, rv, normlx, normly, normrx, normry, lw, ang = llp

    epoch_classes = []

    for epoch in epochs:
        epoch_classes.append(Epoch(epoch, (names, lbound, rbound)))

    for i in range(len(names)):
        if normly[i] != 0.0 or normry[i] != 0.0:
            #TODO what to do with renormalizing?
            print("Renormalizing lines is not implemented yet", flush=True)
            sys.exit()
        if rv[i] != 0.0:
            #TODO what to do what RV shifting?
            print("RV shifting lines before binary calculation is not implemented yet", flush=True)
            sys.exit()

    return names, res, epoch_classes, lw


def create_FORMAL_INPUT(modinicalc, line_subset, lfile, create=True):
    """Create a FORMAL_INPUT file that contains only the lines that
    will be fitted. Based on the linelist we go through the
    FORMAL_INPUT "master" file and only copy those that we need to
    the FORMAL_INPUT that we will use.
    Furthermore, the function checks whether all lines that are in
    the line_subset (the diagnositc lines) are present in the
    FORMAL_INPUT_master file, if not, it exits the run.
    If the parameter 'create' is false, only a check is done,
    and the FORMAL input file is not really created.
    """

    # Read all line-info, this is needed for the UV_ v11 lines
    thelnames, llp = read_linelist(lfile)
    res, lbound, rbound, rv, normlx, normly, normrx, normry, lw, ang = llp

    # Navigate to the main inicalc directory, the one that will
    # be copied all the time.
    os.chdir(modinicalc)

    # Read all lines of the 'master' FORMAL_INPUT file
    if not os.path.isfile('FORMAL_INPUT_master'):
        shutil.copy('FORMAL_INPUT', 'FORMAL_INPUT_master')
    with open('FORMAL_INPUT_master') as f:
        lines = f.readlines()

    # Loop through all lines that will be needed and copy them
    # to a new FORMAL_INPUT file.
    # The not so pretty if-statements below are there to take into
    # account that the information about a line with multiple
    # transitions can be spread out over several lines.
    formal_new = [':T VSINI\n', '0.\n']
    continueline = False
    list_formal_lines = []
    ninputs = 0
    lenforminput = 0
    for line in lines:
        splitline = line.strip().split()
        if len(splitline) > 2:
            if splitline[0] in line_subset or continueline:
                if splitline[0] in line_subset:
                    list_formal_lines.append(splitline[0])
                    nsubslines = int(splitline[1])
                    lenforminput = nsubslines * 4 + 2
                    ninputs = len(splitline)
                if continueline:
                    ninputs = ninputs + len(splitline)
                if ninputs < lenforminput:
                    continueline = True
                else:
                    continueline = False
                formal_new.append(line)

    for ic in range(len(thelnames)):
        theUVline = thelnames[ic]
        if theUVline.startswith('UV_'):
            thelb = str(int(theUVline.split('_')[1]))
            therb = str(int(theUVline.split('_')[2]))
            theang = str(ang[ic])
            newformalin = 'UV ' + thelb + ' ' + therb + ' ' + theang + '\n'
            formal_new.append(newformalin)

    if create:
        # Write the collected lines to the new FORMAL_INPUT file.
        # This is the file that will be used during the run.
        with open('FORMAL_INPUT', 'w') as f:
            for aline in formal_new:
                f.write("%s" % aline)
            f.write("\n")

    # Navigate back to the main directory
    os.chdir('..')

    missing_lines = []
    for aline in line_subset:
        if not aline.startswith('UV_'):
            if aline not in list_formal_lines:
                missing_lines.append(aline)

    if missing_lines != []:
        print('ERROR! Some diagnostic lines are not ' +
              'found in FORMAL_INPUT_master!')
        for missing in missing_lines:
            print(missing + ' not found')
        if not create:
            return False
        else:
            print('Exiting Kiwi-GA... :-(')
            sys.exit()
    if not create:
        print('All lines are present in FORMAL_INPUT_master')
        return True


def gen_modnames(gen: int, size: int, multiplicity: int, zfillen=4):
    """Generate model names of the format xxxx_xxxx_x, e.g.
    for generation 23 and individual 147 the 2nd components this is 0023_0147_2.
    """
    genname = str(gen).zfill(zfillen)
    indnames = np.arange(size)

    modnames = []
    for ind in indnames:
        indiv = []
        for i in range(multiplicity):
            indiv.append(genname + '_' + str(ind).zfill(zfillen) + '_' + str(i))
        modnames.append(indiv)

    return np.array(modnames)

def init_comp_dir(inidir, therundir, name):
    moddir = therundir + name + '/'
    mkdir(moddir)
    modinicalc = moddir + 'inicalc/'
    shutil.copytree(inidir, modinicalc)
    mkdir(modinicalc + name)
    return moddir

def get_vinf(component: pop.Component, loc):
    if component.parameters['vinf'] == -1:
        #handle vinf according to Hawcroft C. 2023
        #change these values in initComponent as well
        if loc == 'SMC':
            a, b = 0.089, 1560  # SMC
        elif loc == 'LMC':
            a, b = 0.088, 1200  # LMC
        else:
            a, b = 0.102, 1300  # GAL
        teff = component.parameters['teff']
        vinf = a * teff - b
        component.parameters['vinf'] = round(vinf, 0)

        # Scaling factor changes with temperature, numbers from
        # Lamers & Cassinelli 1999, fig 2.20 (page 49)
        # logg = component.parameters['logg']
        # g_cgs = 10 ** logg
        #
        # Rsun = 6.96e10  # cm
        # radius_rsun = component.radius
        # radius_cgs = radius_rsun * Rsun
        #
        # cms_to_kms = 1.0e-5
        # vesc_cgs = np.sqrt(2 * g_cgs * radius_cgs)
        # vesc_kms = vesc_cgs * cms_to_kms
        #
        # teff = component.parameters['teff']
        # if teff > 21000.0:
        #     scale_factor = 2.6
        # elif teff <= 21000.0 and teff > 10000.0:
        #     scale_factor = 1.3
        # else:
        #     scale_factor = 0.7
        # vinf_approx = vesc_kms * scale_factor
        # component.parameters['vinf'] = round(vinf_approx, 0)

def get_mdot(component: pop.Component, significant_digits = 6):
    logmdot = component.parameters['mdot']
    realmdot = 10**logmdot

    # Round. Use ceil instead of floor because the log10(mdot)
    # values are always negative.
    sdcor = math.ceil(logmdot)
    roundmdot = round(realmdot, int(-sdcor + significant_digits))

    component.parameters['mdot'] = roundmdot

def get_vclmax(component: pop.Component):
    vclstart = component.parameters['vclstart']
    vclmax = component.parameters['vclmax']
    if vclmax == -1.0:
        component.parameters['vclmax'] = min(round(vclstart*2, 4), 1.0)
    elif vclmax == -2.0:
        component.parameters['vclmax'] = vclstart + 0.10
    elif vclmax >= 0.0 and vclmax <= 1.0:
        if vclmax < vclstart:
            component.parameters['vclmax'] = min(round(vclstart + 0.05, 4), 1.0)


def clumping_type(ficval: float) -> str:
    if ficval >= 999:
        return 'thin'
    return 'thick'

def add2indat(inl, genes: pop.Component, values, element=''):
    if element=='':
        line = ''
        for value in values:
            if value == 'radius':
                line = line + str(genes.radius) + ' '
            else:
                line = line + str(genes.parameters[value]) + ' '

        line = line[:-1] + '\n'
    else:
        if genes.parameters[values[0]] == -1:
            return
        line = element + ' ' + str(genes.parameters[values[0]]) + '\n'
    inl.append(line)

def fcl_rep_hillier(genes: pop.Component):
    """ Get the maximum clumping parameter of the Hillier exponential
        clumping law, assuming an outer radius of 120.0 Rsun.
        Lower that slightly to be safe; this is the represetative clumping
        parameter.
    """
    fcl_out = genes.parameters['fclump']
    beta = genes.parameters['beta']
    vcl = genes.parameters['vcl']
    vinf = genes.parameters['vinf']

    r_in = 1.004
    radius = np.linspace(r_in, 120.0, 1000)
    velocity_r = vinf * (1-r_in/radius)**beta
    fclump_rad = 1.*fcl_out + (1.-1.*fcl_out)*np.exp(-velocity_r/vcl)
    max_fcl = np.max(fclump_rad)
    fcl_rep = round(max_fcl*0.90,5)
    if fcl_rep < 1.0:
        fcl_rep = 1.0
    return fcl_rep


def get_fx_obs(genes: pop.Component):
    mdot = genes.parameters['mdot']
    vinf = genes.parameters['vinf']

    mdot = mdot / 10 ** (-6)
    logmdotvinf = np.log10(mdot / vinf)

    # Relation from Kudritzki, Palsa, Feldmeier et al. (1996)
    logfx = -5.45 - 1.05 * logmdotvinf
    fx = round(10 ** (logfx), 8)

    genes.parameters['fx'] = fx


def get_fx_theory(genes: pop.Component):
    msun = 1.989e33
    rsun = 6.955e10
    year = 365 * 24 * 60 * 60
    mdot = genes.parameters['mdot'] * msun / year  # to g/s
    vinf = genes.parameters['vinf'] * 1e5  # to cm
    radius = genes.parameters['radius'] * rsun  # to cm

    # Compute log10 of wind density in cgs units
    logWD = np.log10(mdot / (4 * np.pi * radius ** 2 * vinf))

    # Relation to get fx that gives approx Lx = 10**-7 Lstar by Brands
    logfx = -0.5541 + 1.2442 * logWD + 0.0851 * logWD ** 2
    fx = round(10 ** logfx, 8)

    genes.parameters['fx'] = fx


def create_indat(genes: pop.Component, name: str, compdir: str, loc, indat_file='INDAT.DAT', formal_in='formal.in', broad_in='broad.in'):
    get_vinf(genes, loc)
    get_mdot(genes)
    get_vclmax(genes)
    clumptype = clumping_type(genes.parameters['fic'])

    inl = ["'" + name + "'\n"]

    if genes.parameters['logfclump'] <= np.log10(1000.0):
        genes.parameters['fclump'] = round(10**genes.parameters['logfclump'], 7)

    # Stuff needed in every FW model.
    add2indat(inl, genes, ['optne_update', 'he_one', 'it_start', 'itmore'])
    add2indat(inl, genes, ['optmixed'])
    add2indat(inl, genes, ['teff', 'logg', 'radius'])
    add2indat(inl, genes, ['rmax', 'tmin'])
    add2indat(inl, genes, ['mdot', 'vmin_start', 'vinf', 'beta', 'vdiv'])
    add2indat(inl, genes, ['yhe', 'ihe_start'])
    add2indat(inl, genes, ['optmod', 'opttlucy', 'megas', 'accel', 'optcmf'])
    add2indat(inl, genes, ['micro', 'metallicity', 'lines', 'lines_in_model'])
    add2indat(inl, genes, ['enat_cor', 'expansion', 'set_first', 'set_step'])

    genes.parameters['fclump_rep'] = fcl_rep_hillier(genes)

    # Wind clumping and porosity etc, optically thin or thick
    if clumptype == 'thin':

        # If vcl > 0, use Hilliers exponential clumping law
        if genes.parameters['vcl'] > 0:
            add2indat(inl, genes, ['fclump_rep', 'fclump', 'vcl', 'vcldummy'])
        # Else, use the linear step function law
        else:
            add2indat(inl, genes, ['fclump', 'vclstart', 'vclmax'])
    else:
        # Assume fic was given in log scale if fic < 0
        if genes.parameters['fic'] < 0.0:
            genes.parameters['fic'] = round(10 ** genes.parameters['fic'], 6)

        inl.append('THICK\n')

        # If vcl > 0, use Hilliers exponential clumping law
        if genes.parameters['vcl'] > 0:
            add2indat(inl, genes, ['fclump_rep', 'fclump', 'vcl', 'vcldummy'])
            add2indat(inl, genes, ['fic', 'fic', 'vcl', 'vcldummy'])
            add2indat(inl, genes, ['fvel', 'fvel', 'vcl', 'vcldummy'])
            add2indat(inl, genes, ['hclump', 'hclump', 'vcl', 'vcldummy'])
        # Else, use the linear step function law
        else:
            add2indat(inl, genes, ['fclump', 'vclstart', 'vclmax'])
            add2indat(inl, genes, ['fic', 'vclstart', 'vclmax'])
            add2indat(inl, genes, ['fvel', 'vclstart', 'vclmax'])
            add2indat(inl, genes, ['hclump', 'vclstart', 'vclmax'])

    # Abundances (will only be added if not set to -1)
    # Mind the allcaps spelling that has to be written
    # to the INDAT file for the multi-letter abbreviations!
    add2indat(inl, genes, ['C'], 'C')
    add2indat(inl, genes, ['N'], 'N')
    add2indat(inl, genes, ['O'], 'O')
    add2indat(inl, genes, ['Mg'], 'MG')
    add2indat(inl, genes, ['Si'], 'SI')
    add2indat(inl, genes, ['P'], 'P')
    add2indat(inl, genes, ['S'], 'S')
    add2indat(inl, genes, ['Fe'], 'FE')
    add2indat(inl, genes, ['Na'], 'NA')
    add2indat(inl, genes, ['Ca'], 'CA')

    # XRAYS - the parameter 'xpow' determines which X-ray prescription is used
    #  -  if xpow <= -1000, the one of  Carneiro+16 is used. In this case 'fx'
    #     is the X-ray volume filling fraction and 'xpow' has no meaning.
    #  -  if xpow > -1000 (but in practice > 0) the prescription of Puls+20
    #     is used. In this case 'fx' is n0, a normalisation of the power law,
    #     (see n_so in Puls+20 paper for details), and xpow the PL exponent.

    # Include X-rays if the volume filling fraction fx > 0.0
    # (fx = 0 means no volume filled with X-rays, so exclude X-rays)
    # When fx > 1000, estimate it based on mdot and vinf:
    if genes.parameters['xpow'] <= -1000:
        # Use the Carneiro+16 prescription
        if genes.parameters['fx'] > 1000:
            get_fx_obs(genes)  # # Kudritzki relation to get 10**-7
        elif genes.parameters['fx'] < -1000:
            get_fx_theory(genes)  # Theoretical relation to get 10**-7
        # Use logscale fx value if that is in a valid range
        #  (only when it has set to such value in defaults, or in para-
        #  meter space, it will)
        if genes.parameters['logfx'] <= np.log10(16.0):
            genes.parameters['fx'] = round(10 ** genes.parameters['logfx'], 7)
        # Add X-rays if the volume filling fraction > 0 *and* if teff is high
        # enough X-rays are in FW not allowed for Teff<25000 (model will crash)
        if genes.parameters['fx'] > 0.0 and genes.parameters['teff'] >= 25000.0:
            inl.append('XRAYS ' + str(genes.parameters['fx']) + '\n')
            add2indat(inl, genes, ['gamx', 'mx', 'Rinx', 'uinfx', 'xpow'])
    else:
        # Use the Puls+20 prescription

        if genes.parameters['fx'] > 0.0 and genes.parameters['teff'] >= 25000.0:
            inl.append('XRAYS ' + str(genes.parameters['fx']) + '\n')
            add2indat(inl, genes, ['gamx', 'mx', 'Rinx', 'uinfx', 'xpow'])

    # Write indat file
    with open(compdir + 'inicalc/' + indat_file, 'w') as f:
        for indatline in inl:
            f.write(indatline)

    # Write input file for pformal
    with open(compdir + 'inicalc/' + formal_in, 'w') as f:
        f.write(name + '\n')
        windt = genes.parameters['windturb']
        if windt > 0.0 and windt < 1.0:
            turbstring = str(genes.parameters['micro']) + ' ' + str(windt) + '\n'
        else:
            turbstring = str(genes.parameters['micro']) + '\n'
        f.write(turbstring)
        f.write(str(genes.parameters['do_iescat']) + '\n')

    # Write input file for broaden.py
    with open(compdir + broad_in, 'w') as f:
        f.write(str(genes.parameters['vrot']) + '\n')
        f.write(str(genes.parameters['macro']) + '\n')


def execute_fastwind(atom, fw_timeout, inicalcdir, verbose):
    os.chdir(inicalcdir)

    exe = f"./pnlte_{atom}.eo"
    logfile = "pnlte.log"

    if verbose == 'Y' or verbose == 'Yes':
        print('Start pnlte ' + inicalcdir + ' at ' + str(datetime.datetime.now()), flush=True)

    try:
        with open(logfile, "w") as fout:
            subprocess.run(
                [exe],
                stdout=fout,  # large output goes to file
                stderr=subprocess.PIPE,  # capture only stderr
                text=True,
                timeout=fw_timeout,
                check=True
            )

    except subprocess.CalledProcessError as e:
        print(f"{exe} of {inicalcdir} failed with code {e.returncode}")

    except subprocess.TimeoutExpired as e:
        print(f"{exe} of {inicalcdir} timed out")

    if verbose == 'Y' or verbose == 'Yes':
        print('Start formalsol ' + inicalcdir + ' at ' + str(datetime.datetime.now()), flush=True)

    exe = f"./pformalsol_{atom}.eo"
    try:
        with open("formal.in", "r") as fin, open("pformal.log", "w") as fout:
            subprocess.run(
                [exe],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=15 * 60,  # 15 minutes in seconds
                check=True
            )

    except subprocess.CalledProcessError as e:
        print(f"{exe}  of {inicalcdir} failed with code {e.returncode}")

    except subprocess.TimeoutExpired as e:
        print(f"{exe}  of {inicalcdir} timed out")

    os.chdir('../../../../')


def read_fwline(OUT_file):
    '''Get wavelength and normflux from OUT.-file
       Treat CMF parts of the spectrum different from v10-
       like lines'''
    if not OUT_file.split('OUT.')[-1].startswith('UV_'):
        tmp_matrix = np.loadtxt(OUT_file, max_rows=161, unpack=True)
        if tmp_matrix.size == 0:
            wave, flux, continuum = [0], [0], [0]
        else:
            wave, flux, continuum = tmp_matrix[2], tmp_matrix[4], tmp_matrix[3]
    else:
        tmp_matrix = np.loadtxt(OUT_file, unpack=True)
        if tmp_matrix.size == 0:
            wave, flux, continuum = [0], [0], [0]
        else:
            wave, flux, continuum = tmp_matrix[1], tmp_matrix[3], tmp_matrix[2]
    return wave, flux, continuum


def prep_broad(linename, line_file, profiles, rmax, radius):
    """Read in fastwind output of the 'OUT.' format and
    convert this to a file that can be read by broaden.py.
    """
    out_clean = profiles + linename + '.prof'
    wave, flux, continuum = read_fwline(line_file)
    continuum = (rmax*radius)**2 * continuum
    if len(wave) == 1:
        out_clean = 'skip'
    else:
        np.savetxt(out_clean, np.array([wave, flux]).T)
    return out_clean, wave, flux, continuum

def apply_broadening(mname, moddir, linenames, lineres, rmax, radius):
    """Broaden the fastwind output with the instrumental profile,
    rotational broadening and macro broadening. The values for
    the instrumental broadening can differ per line and are
    given to the function, the values for the rotational and
    macroturbulence are read from a file generated in the
    function create_indat.
    """

    modinicalc = moddir + 'inicalc/'
    fw_run_output = modinicalc + mname + '/'
    profiles = moddir + 'profiles/'
    mkdir(profiles)

    # Look up all FW line output and exit if there is none.
    linefiles = glob.glob(fw_run_output + 'OUT.*')
    if len(linefiles) == 0:
        return 0

    # These likely have a different order than the filenames
    # that are read in from the linefile.
    linenames_fromfile = []
    for line in linefiles:
        the_line_name = line.rpartition('_')[0][4:]
        the_line_name = the_line_name.rpartition('OUT.')[-1]
        if the_line_name.startswith('UV_'):
            lsplit = the_line_name.split('_')
            the_line_name = 'UV_' + lsplit[1] + '_' + lsplit[2]
        else:
            the_line_name = the_line_name.rpartition('OUT.')[-1]
        linenames_fromfile.append(the_line_name)

    # Read in the broadening properties for the model.
    vrot, vmacro = np.genfromtxt(moddir + 'broad.in')

    # Create a dictionary for lookup of resolving power per line
    resdct = dict(zip(linenames, lineres))

    # Loop through the OUT. files and apply broadening
    for linename, linefle in zip(linenames_fromfile, linefiles):
        # Convert to readable format
        finput, wave, flux, continuum = prep_broad(linename, linefle, profiles, rmax, radius)
        if finput == 'skip':
            return 0
        # Lookup resolving power
        res = resdct[linename]
        # Apply broadening
        new_wave, new_flux = br.broaden_fwline(wave, flux, vrot, res, vmacro)
        new_continuum = np.interp(new_wave, wave, continuum)
        np.savetxt(finput + ".fin", np.array([new_wave, new_flux, new_continuum]).T)

    return 1


def run_fw(modelatom, compdir, name, fw_timeout, lineinfo, rmax, radius, verbose):

    execute_fastwind(modelatom, fw_timeout, compdir + 'inicalc/', verbose)

    # Apply instrumental, rotational and macroturbulent
    # broadening to the fastwind OUT. files.
    linenames, lineres = lineinfo[:2]

    try:
        return apply_broadening(name, compdir, linenames, lineres, rmax, radius)
    except Exception as error:
        print(f"Application of broadening failed due to {error}", flush=True)
        return 0


def grep_pnlte(modinicalc, search, outputfile, loc):
    """Function to search the pnlte-log file"""
    pnltelog = modinicalc + 'pnlte.log'
    tmp = modinicalc + outputfile + '.tmp'
    txt = modinicalc + outputfile + '.txt'
    grep = 'grep "' + search + '" ' + pnltelog + ' >> ' + tmp + ' ; '
    tail = 'tail -1 ' + tmp + ' > ' + txt + ' ; '
    rm = 'rm ' + tmp
    os.system(grep + tail + rm)
    if os.path.getsize(txt) > 0:
        value = np.genfromtxt(txt)[loc]
    else:
        value = 0
    return float(value)

def get_runinfo(modinicalc):
    """Function that looks up the number of NLTE-iterations that
    fastwind has done, the maximum correction of the last iteration,
    and, if the model has finished, the total CPU time.
    This is done by using grep on the pnlte.log file.
    (this file will later be removed)
    """
    if os.path.exists(modinicalc + 'pnlte.log'):
        try:
            maxcor = grep_pnlte(modinicalc, "CORR. MAX:", 'corr_max', -1)
        except:
            maxcor = 0.0
        try:
            maxit = grep_pnlte(modinicalc, "+  ITERATION NO", 'it_max', -2)
        except:
            maxit = 0
        try:
            cputime = grep_pnlte(modinicalc, "CPU time", 'cpu', -1)
            if cputime == 0.0:
                cputime = 99999.9
        except:
            cputime = 99999.9
    else:
        maxcor = 0.0
        maxit = 0
        cputime = 99999.9

    return [maxcor, maxit, cputime]

def get_xlum_out(fw_run_out):
    """ Get the X-ray luminosity from the FW output
        Input: path to model directory (string)
        Output: Lx/L (string)
    """

    xlumitfile = fw_run_out + '/XLUM_ITERATION'

    if os.path.isfile(xlumitfile):
        with open(xlumitfile) as f:
            content = f.readlines()
            if len(content) > 0:
                xlumline = content[-1].strip().split()
                xlum = xlumline[2]
            else:
                xlum = -1
    else:
        xlum = -1

    return xlum

def parallelcrop(list1, list2, list3, start_list1, stop_list1):
    """ Based on values in list1, crop the same arguments of
    list2 and 3. Used for cropping spectra based on wavelength
    boundaries. If list3 equals [], only 2 lists are cropped"""

    newlist1 = list1[(list1 > start_list1) & (list1 < stop_list1)]
    newlist2 = list2[(list1 > start_list1) & (list1 < stop_list1)]
    if len(list3) == 0:
        return newlist1, newlist2
    else:
        newlist3 = list3[(list1 > start_list1) & (list1 < stop_list1)]
        return newlist1, newlist2, newlist3

def ionizing_fluxes(lam, fnu, radius):
    c = 2.99792458e10
    h = 6.6260755e-27
    rsun = 6.957e10

    nu = c/(lam * 1e-8)# Hz [per second]
    photon_energy_nu = nu*h # Hz * ergs s^-1 = ergs [units of energy]

    integrand = fnu/photon_energy_nu # ph s^-1 cm^-2 Hz^-1
    sorting = nu.argsort()
    nu = nu[sorting]
    integrand = integrand[sorting]
    nulow_HI = c/(912.0e-8)
    nulow_HeI = c/(504.0e-8)
    nulow_HeII = c/(228.0e-8)

    nip = 1000000
    the_ip = interpolate.interp1d(nu, integrand)
    nu = np.linspace(min(nu), max(nu), nip)
    integrand = the_ip(nu)

    nuHI, integrandHI = parallelcrop(nu, integrand, [], nulow_HI, 1e100)
    nuHeI, integrandHeI = parallelcrop(nu, integrand, [], nulow_HeI, 1e100)
    nuHeII, integrandHeII = parallelcrop(nu, integrand, [], nulow_HeII, 1e100)

    # Integrate the integrand [ph s^-1 cm^-2 Hz^-1] over frequency: Hz
    # ph s^-1 cm^-2  [number of photons per surface area per second]
    q0 = trapezoid(integrandHI, nuHI)
    Q0 = q0 * 4*np.pi * (radius*rsun)**2 # ph s^-1 [integrate over surface]
    logq0 = round(np.log10(q0),3)
    logQ0 = round(np.log10(Q0),3)
    q1 = trapezoid(integrandHeI, nuHeI)
    Q1 = q1 * 4*np.pi * (radius*rsun)**2 # ph s^-1 [integrate over surface]
    logq1 = round(np.log10(q1),3)
    logQ1 = round(np.log10(Q1),3)
    q2 = trapezoid(integrandHeII, nuHeII)
    Q2 = q2 * 4*np.pi * (radius*rsun)**2 # ph s^-1 [integrate over surface]
    logq2 = round(np.log10(q2),3)
    logQ2 = round(np.log10(Q2),3)

    return logq0, logQ0, logq1, logQ1, logq2, logQ2

def read_fluxcont(fw_run_out, rstar, rmax_fw):
    """Check if FLUXCONT is there and if so, read it to get out
       the ionising fluxes.
       Input: path to model directory and model name,
           maximum radius of FW model, stellar radius (both in rsun).
       Output: q0, Q0, q1, Q1, q2, Q2
    """

    fluxcont = fw_run_out + '/FLUXCONT'

    if os.path.isfile(fluxcont):

        # Look up the number of useful lines in the FLUXCONT
        lcount = -2
        for aline in open(fluxcont, 'r').readlines():
            lcount = lcount+1
            if len(aline.split()) == 1:
                break

        rsun = 6.96e10 # cm
        rstar = float(rstar)
        rmax_fw = float(rmax_fw)
        stellar_surface = 4*np.pi*(rsun*rstar)**2

        # Only read non empty files, FLUXCONT has typically about
        # 1700-1800 lines containing flux information
        if lcount > 500:
            # Get FASTWIND spectrum
            lam, logFnu = np.genfromtxt(fluxcont, max_rows=lcount,
                skip_header=1, delimiter='').T[1:3]
            fnu = 10**logFnu # ergs/s/cm^2/Hz / RMAX^2
            fnu = fnu * rmax_fw**2 # ergs/s/A

            q0, Q0, q1, Q1, q2, Q2 = ionizing_fluxes(lam, fnu, rstar)

            return [q0, Q0, q1, Q1, q2, Q2]

    return [-888, -888, -888, -888, -888, -888]


def run_diagnostic_saves(comp_outdir, compinicalc, name, genes: pop.Component):
    fw_run_out = compinicalc + name + '/'
    runinfo = get_runinfo(compinicalc)
    xlum = get_xlum_out(fw_run_out)
    ionfluxinfo = read_fluxcont(fw_run_out, genes.radius, genes.parameters['rmax'])

    header = ['run_id', 'maxcorr', 'maxit', 'cputime', 'xlum', 'logq0', 'logQ0', 'logq1', 'logQ1', 'logq2', 'logQ2', 'radius']

    data = [name]

    for item in runinfo:
        data.append(item)

    data.append(xlum)

    for item in ionfluxinfo:
        data.append(item)

    data.append(genes.radius)

    #save new params in component also
    for diagnostic in range(1, len(header)-1):
        genes.parameters[header[diagnostic]] = data[diagnostic]

    parameters = genes.parameters.copy().keys()

    for param in parameters:
        header.append(param)
        data.append(genes.parameters[param])

    np.savetxt(comp_outdir + 'params.csv', np.array([header, data]), fmt='%s', delimiter=',')


def save_failed_output(compdir, savedir, name, genes):
    name_parts = name.split('_')
    gendir = savedir + name_parts[0] + '/'
    inddir = gendir + name_parts[1] + '/'
    if len(name_parts) == 3:
        comp_outdir = inddir + name_parts[2] + '/'
    else:
        comp_outdir = inddir + name_parts[2]
        for additional in name_parts[3:]:
            comp_outdir += '_' + additional
        comp_outdir += '/'
    mkdir(gendir)
    mkdir(inddir)
    mkdir(comp_outdir)
    compinicalc = compdir + 'inicalc/'

    # Copy the files that describe the model to the savedir
    shutil.copy(compinicalc + 'INDAT.DAT', comp_outdir + 'INDAT.DAT')
    shutil.copy(compdir + 'broad.in', comp_outdir + 'broad.in')
    shutil.copy(compinicalc + 'formal.in', comp_outdir + 'formal.in')

    run_diagnostic_saves(comp_outdir, compinicalc, name, genes)

    # Compress saved dir to tar.gz file and remove the directory
    if len(name_parts) == 3:
        tarfilename = inddir + name_parts[2] + '.tar.gz'
    else:
        tarfilename = inddir + name_parts[2]
        for additional in name_parts[3:]:
            tarfilename += '_' + additional
        tarfilename += '.tar.gz'

    tar(comp_outdir, tarfilename)
    shutil.rmtree(comp_outdir)


    # Remove all the files from 'run'
    shutil.rmtree(compdir)


def save_fastwind_output(compdir, savedir, name, genes):
    name_parts = name.split('_')
    gendir = savedir + name_parts[0] + '/'
    inddir = gendir + name_parts[1] + '/'
    if len(name_parts) == 3:
        comp_outdir = inddir + name_parts[2] + '/'
    else:
        comp_outdir = inddir + name_parts[2]
        for additional in name_parts[3:]:
            comp_outdir += '_' + additional
        comp_outdir += '/'

    profile_dir = comp_outdir + 'profiles/'
    mkdir(gendir)
    mkdir(inddir)
    mkdir(comp_outdir)
    mkdir(profile_dir)
    compinicalc = compdir + 'inicalc/'

    profiles = compdir + 'profiles/'
    os.system('cp ' + profiles + '*.prof.fin ' + comp_outdir + 'profiles/')

    # Copy the files that describe the model to the savedir
    shutil.copy(compinicalc + 'INDAT.DAT', comp_outdir + 'INDAT.DAT')
    shutil.copy(compdir + 'broad.in', comp_outdir + 'broad.in')
    shutil.copy(compinicalc + 'formal.in', comp_outdir + 'formal.in')

    run_diagnostic_saves(comp_outdir, compinicalc, name, genes)

    # Remove all the files from 'run'
    shutil.rmtree(compdir)

def get_fastwind_output(inicalcdir, rundir, savedir, modelatom, fw_timeout, lineinfo, fail_counter: int, precomp, verbose, loc, genes: pop.Component, timer=0.0):

    start_time = time.time()
    name = genes.name

    compdir = init_comp_dir(inicalcdir, rundir, name)
    create_indat(genes, name, compdir, loc)
    out = run_fw(modelatom, compdir, name, fw_timeout, lineinfo, genes.parameters['rmax'], genes.radius, verbose)
    if out == 0:
        total_time = timer + time.time() - start_time
        if 0 <= fail_counter < 5 and total_time < 100:
            fail_counter += 1
            save_failed_output(compdir, savedir, name, genes)
            new_try = pop.Retry(genes, fail_counter, loc, verbose)
            new_try.append_precompute(precomp)
            return get_fastwind_output(inicalcdir, rundir, savedir, modelatom, fw_timeout, lineinfo, fail_counter, precomp, verbose, loc, new_try, timer=total_time)
        else:
            save_failed_output(compdir, savedir, name, genes)
            return None
    save_fastwind_output(compdir, savedir, name, genes)
    return genes

def failed_model(epochs: list[Epoch], multiplicity: int):
    """Returns the fitness values of a crashed model"""
    chi2_tot = 999999999
    rchi2_tot = 999999999
    dof_tot = -1
    fitness = 0.0
    fitnesses_lines = np.zeros(len(epochs[0].get_line_names()))
    fitm = 999999999
    vrads_per_epoch = []
    for epoch in epochs:
        line = [epoch.name]
        for _ in range(multiplicity):
            line.append(0.0)
        vrads_per_epoch.append(line)
    return fitm, fitness, chi2_tot, rchi2_tot, dof_tot, epochs[0].get_line_names(), fitnesses_lines, vrads_per_epoch

def store_model(chi2file, name, gen, fitinfo):
    fitmeasure, fitness, chi2_tot, rchi2_tot, dof_tot, linenames, linefitns, vrads = fitinfo

    write_lines = []

    if not os.path.isfile(chi2file):
        hstring = '#run_id,gen,chi2,rchi2,dof,fitness'
        for linename in linenames:
            hstring = hstring + ',' + linename
        hstring = hstring + '\n'
        write_lines.append(hstring)

    istr = name + ',' + str(gen) + ',' + str(chi2_tot) + ',' + str(rchi2_tot) + ',' + str(dof_tot) + ',' + str(fitness)

    for lfit in linefitns:
        istr = istr + ',' + str(lfit)
    istr = istr + '\n'
    write_lines.append(istr)

    with open(chi2file, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)


def shrink_savefile(comp_outdir, lineinfo):
    linenames, lineres, epochs, lineweight = lineinfo
    for line in linenames:
        linefile = comp_outdir + 'profiles/' + line + '.prof.fin'
        wave, flux, cont = np.loadtxt(linefile, unpack=True)
        lower, upper = np.inf,0
        for epoch in epochs:
            wave_data, _, _ = epoch.get_line_data(line)
            if len(wave_data) > 0:
                lower = min(lower, min(wave_data))
                upper = max(upper, max(wave_data))
        #500 km/s buffer around the area
        c = 299792.458  # km/s
        lower = lower * (1.0 - 500 / c)
        upper = upper * (1.0 + 500 / c)
        save_wave = np.linspace(lower, upper, 200)
        save_flux = np.interp(save_wave, wave, flux)
        save_cont = np.interp(save_wave, wave, cont)
        np.savetxt(linefile, np.array([save_wave, save_flux, save_cont]).T,fmt='%10.5f')


def cleanup_files(components, inddir, vrads, lineinfo):
    # Compress saved dir to tar.gz file and remove the directory. Also create vrads and params.csv file in the combined dir and zip that
    combined = inddir + 'combined/'
    tarnamecombined = inddir + 'combined.tar.gz'
    mkdir(combined)

    write_lines = []

    istr = ''
    for vrad in vrads:
        for i in vrad:
            istr = istr + str(i) + ','
        istr = istr + '\n'
    write_lines.append(istr)

    with open(combined+'vrads.txt', 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

    params = pd.DataFrame()

    for component in components:
        name_parts = component.split('_')
        if len(name_parts) == 3:
            tarfilename = inddir + name_parts[2] + '.tar.gz'
            comp_outdir = inddir + component.split('_')[2] + '/'
        else:
            tarfilename = inddir + name_parts[2]
            comp_outdir = inddir + name_parts[2]
            for additional in name_parts[3:]:
                tarfilename += '_' + additional
                comp_outdir += '_' + additional
            tarfilename += '.tar.gz'
            comp_outdir += '/'
        shrink_savefile(comp_outdir, lineinfo)
        comp_params = pd.read_csv(comp_outdir + 'params.csv')
        params = pd.concat([params, comp_params])
        tar(comp_outdir, tarfilename)
        shutil.rmtree(comp_outdir)

    params.to_csv(combined+'params.csv', index=False)
    tar(combined, tarnamecombined)
    shutil.rmtree(combined)


def permuted_initial_guesses(x):
    perms = set(itertools.permutations(x))
    return [np.array(p) for p in perms]

def fit_vrads(epoch: Epoch, line_data, guesses):
    res0 = least_squares(epoch.get_residuals, guesses, args=[line_data], method='trf', bounds=(min(guesses)-600, max(guesses)+600))

    best_result = res0
    best_cost = res0.cost

    for x0 in permuted_initial_guesses(res0.x):
        res = least_squares(
            epoch.get_residuals,
            x0=x0,
            args=(line_data,),
            method="trf",
            bounds=(min(guesses) - 600, max(guesses) + 600)
        )

        if res.cost < best_cost:
            best_cost = res.cost
            best_result = res

    return best_result.x

def calc_chi2_line(epoch: Epoch , linename, line_data, vrads, dof, inddir, maxlen=150):
    """Calculate the chi2 value of a line, and, in case
    the model spectrum is saved in high resolution, create and
    save a degraded version of the model spectrum to prevent massive
    output files.
    (Note: a file with 150 lines is 7.3K, for a run of 20 lines,
    180 generations, and 240 individuals, the total output is then
    about 6GB).
    """
    c = 299792.458  # km/s
    #epoch data:
    wave_data, flux_data, error_data = epoch.get_line_data(linename)

    if len(wave_data) < 1:
        return 0, 0, 0

    total_flux = np.zeros_like(flux_data)
    total_cont = np.zeros_like(flux_data)

    for k, vrad in enumerate(vrads):
        model = line_data[k].T
        wave_model, flux_model, cont_model = model

        shifted_wave = wave_model * (1.0 + vrad / c)

        flux_interp = np.interp(wave_data, shifted_wave, flux_model)
        cont_interp = np.interp(wave_data, shifted_wave, cont_model)

        total_flux += flux_interp * cont_interp
        total_cont += cont_interp

    combined_flux = total_flux / total_cont

    chi2_line = np.sum(((flux_data - combined_flux) / error_data)**2)
    np_line = len(flux_data)
    dof_line = np_line-dof
    rchi2_line = chi2_line / dof_line

    # Save a low res combined spectrum
    combined = inddir + 'combined/'
    combinedprofiles = combined + 'profiles/'
    mkdir(combined)
    mkdir(combinedprofiles)
    save_wave = np.linspace(min(wave_data), max(wave_data), maxlen)
    save_flux = np.interp(save_wave, wave_data, combined_flux)

    np.savetxt(combinedprofiles + linename + '.prof.comb', np.array([save_wave, save_flux]).T,fmt='%10.5f')

    return chi2_line, rchi2_line, np_line

def calc_fitness(rchi2s, weights):
    """ Calculate the fitness, given reduced chi2 values and
    weights for each spectral line.
    """
    weights = np.array(weights)
    rchi2s = np.array(rchi2s)
    fitness = 1./(np.sum(weights * rchi2s) / np.sum(weights))
    return fitness

def assess_fitness(components: list[str], dof: int, lineinfo: tuple[list[str], list[int], list[Epoch], list[int]], savedir: str, inddir: str, fitmeasure, vrad_guesses: list[float]):
    """
    Given a list of names of components holding the broadened data, their individual vrad will be fitted per epoch,
    their spectra will be added together and the result will be compared to the data.
    The resulting fitness measures will be writen to an output file.
    :param components: list of the names that hold the broadened data per component
    :param dof: number of free parameters over all components
    :param lineinfo: the names, resolution, different epochs and their weights of the lines to be fit
    :param inddir: the directory containing the components of the current individual
    :return:
    """
    linenames, lineres, epochs, lineweight = lineinfo
    linefiles = []
    for lname in linenames:
        linefiles.append('profiles/' + lname + '.prof.fin')

    try:
        chi2_tot = 0
        dof_tot = 0
        chi2_lines = np.zeros(len(linenames))
        rchi2_lines = np.zeros(len(linenames))
        #weight_lines = []

        #preload the data
        line_data = []
        for line in linefiles:
            ind_line_files = []
            for comp in components:
                parts = comp.split('_')
                gen = parts[0]
                ind = parts[1]
                if len(parts) == 3:
                    name = parts[2]
                else:
                    name = parts[2] + '_' + parts[3]
                path = savedir + gen + '/' + ind + '/' + name
                if os.path.exists(path):
                    ind_line_files.append(np.loadtxt(path + '/' + line))
                else:
                    mkdir(path)
                    untar(path + '.tar.gz', path + '/')
                    os.remove(path + '.tar.gz')
                    ind_line_files.append(np.loadtxt(path + '/' + line))
            line_data.append(ind_line_files)

        vrad_per_epoch = []
        for epoch_idx in range(len(epochs)):
            # first: find the best fitting vrad
            vrads = fit_vrads(epochs[epoch_idx], line_data, vrad_guesses)
            line = [epochs[epoch_idx].name]
            for vrad in vrads:
                line.append(vrad)
            vrad_per_epoch.append(line)
            #then, calculate the fitness of this epoch
            for i in range(len(linefiles)):
                chi2info = calc_chi2_line(epochs[epoch_idx], linenames[i], line_data[i], vrads, dof, inddir)
                chi2_line, rchi2_line, np_line = chi2info

                chi2_lines[i] += chi2_line
                rchi2_lines[i] += rchi2_line
                #weight_lines.append(lineweight[i])

                chi2_tot += chi2_line
                dof_tot += np_line

        dof_tot = dof_tot - dof
        rchi2_tot = chi2_tot / dof_tot
        fitness = calc_fitness(rchi2_lines, lineweight)

        fitnesses_lines = 1./np.array(rchi2_lines)

    except Exception as error:
        print(f"Model fitness assessment failed due to {error}", flush=True)
        return failed_model(epochs, len(components))

    ####################### FITNESS MEASURE #######################
    # The reproduction code assumes higher value for the fitness
    # measure = fitter model therefore we inverse the fitness here.
    # For chi2 as a measure this is already the case. Because only
    # the _order_ order the fitness of the models is relevant for
    # reproduction, and not the absolute fitness, the way of
    # 'changing the scale' is not important.
    if fitmeasure == 'chi2':
        fitm = chi2_tot
    else:
        if fitness != 0.0:
            fitm = 1./fitness
        else:
            fitm = 999999999

    return fitm, fitness, chi2_tot, rchi2_tot, dof_tot, linenames, fitnesses_lines, vrad_per_epoch


def evaluate_fitnesses(population: pop.Population, lineinfo, savedir, precompfile, dof, fitmeasure, chi2file):
    epochs = lineinfo[2]

    gendir = savedir + population.name + '/'

    precompdata = pd.read_csv(precompfile)

    for individual in population.population:
        inddir = gendir + individual.name.split('_')[1] + '/'
        failure = False
        components = []
        vrad_guesses = []
        for comp in individual.components:
            name = comp.name
            status = precompdata.loc[precompdata['name'] == name, 'status'].iloc[0]
            if status == 'fail':
                failure = True
            else:
                vrad_guesses.append(comp.parameters['vrad_guess'])
                components.append(precompdata.loc[precompdata['name'] == name, 'result_owner'].iloc[0])
        if failure:
            fitinfo = failed_model(epochs, population.multiplicity)
        else:
            fitinfo = assess_fitness(components, dof, lineinfo, savedir, inddir, fitmeasure, vrad_guesses)
        store_model(chi2file, individual.name, population.name, fitinfo)
        cleanup_files(components, inddir, fitinfo[-1], lineinfo)
        individual.set_fitting_params(fitinfo)


def read_mut_gen(mut_gen_file):
    """ Function for restarting the run. Reads mutation rate and
    generation number of last generation
    """
    mutgenlines = np.genfromtxt(mut_gen_file)
    lenmut  = len(np.array(mutgenlines.shape))

    # If multiple generations have been computed already, take
    # the last line of the file only.
    if lenmut == 2:
        mutgenlines = mutgenlines[-1]

    gen = int(mutgenlines[0])
    mutrate = mutgenlines[1]

    return gen, mutrate