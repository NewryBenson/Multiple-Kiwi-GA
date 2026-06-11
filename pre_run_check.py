# Sarah Brands s.a.brands@uva.nl 25-02-2020
# Script with some basic checks on Kiwi-GA input.
#
# Updated for use on Snellius by Frank Backs 18-10-2021
#
# Sander Van den Eynden 11-3-2026 edits: Multiple-Kiwi-GA
#
# Should be executed before starting a run:
#  - Checks several aspects of the run setup
#      * Prints report where errors are pointed out
#      * Plots spectrum, line boundaries and parameter space
#        (saved to pdf in input directory of the run)
#  - Creates a job file
#
# Usage: 
#   python pre_run_check.py <runname>
# Or if you want to both print and save the output of the script:
#   python pre_run_check.py <runname> | tee -a "input/<runname>/prerun.log"
# Saves log into run input directory (also the pdf report goes here).

import os
import sys 
import math
import numpy as np
from matplotlib import pyplot as plt
# from PyPDF2 import PdfFileMerger
from matplotlib.backends.backend_pdf import PdfPages

import fastwind_wrapper as fw
import population as pop
import cluster_inputs as ci
from epoch import Epoch

fw.mkdir('runs/')
jobscriptfile = 'runs/run_kiwiGA.job' # name of job script file
minutes_str = ci.max_wall_time # maximum wall time -- but see below!
walltime_flex = False # adjust maximum wall time depending on ngen
mnts_gen = ci.time_per_gen # minutes per generation if walltime_flex
n_cpu_core = ci.cores_per_node  # number of CPUs per node
username = ci.username
codedir = ci.codedir  # 'Kiwi-GA'


# For spectral regions wider than this, an extra wide plot will be made.
wide_specrange_lim = 100.0 # Angstrom

# When checking the UV/CMF range, a computed spectral range is considered
# "too large" when it is larger than x Angstrom from the edge of the 
# range that is fitted.
larger_wavemax = 5.0 # Angstrom

run_name = sys.argv[1]
if run_name.endswith('/'):
    run_name = run_name[:-1]
if len(sys.argv) > 2:
    if sys.argv[2] == 'restart' or sys.argv[2] == '-restart':
        do_restart='yes'
    else:
        do_restart='no'
else:
    do_restart='no'


inputmain = 'input/'
inputdir = inputmain + run_name + '/'
componentdir = inputdir + 'components/'
epochdir = inputdir + 'epochs/'
controlname = inputdir + 'control.txt'
paramspacecomponentfiles = os.listdir(componentdir)
paramspacecomponentfiles.sort()
radinfoname = inputdir + 'radius_info.txt'
linesetname = inputdir + 'line_list.txt'
defaultsname = inputdir + 'defaults_fastwind.txt'
epochfiles = os.listdir(epochdir)
epochfiles.sort()

paramspace_in = [componentdir + component for component in paramspacecomponentfiles]
epoch_in = [epochdir + epoch for epoch in epochfiles]

multiplicity = len(paramspace_in)

param_fig = inputdir + 'parameter_input.pdf'
spectrum_fig = inputdir + 'spectrum_input.pdf'
spectrum_fig2 = inputdir + 'spectrum_input2.pdf'

pdfs = [param_fig, spectrum_fig]
merged_report = inputdir + "pre_run_report.pdf"

reportPDF = PdfPages(merged_report)

report = inputdir + "pre_run_report.txt"

if not os.path.isdir(inputdir):
    print("Could not find direcotry " + inputdir)
    print("    Typos happen to the best")
    sys.exit()

def print_write(the_string, the_file):
    with open(the_file, "a") as text_file:
        text_file.write(the_string + '\n')

def printsection(title):
    print('\n____' + title + '____')

def printsumline(entry, comment, nd):
    printstring = '##     >  ' + entry + ': ' + comment
    spaces = (nd - len(printstring)-2)*' ' + '##'
    printstring = printstring + spaces
    print(printstring)

###################################
#        Check parameters         #
###################################

# Start of script, Set up dictionary
nd = 79
intro = '  Starting pre-run check for: ' + run_name + '  '
halfpt = int((nd - len(intro)-2)/2)
otherhalfpt = int((nd - len(intro) - halfpt))
print('\n' + halfpt*'#'  + intro + otherhalfpt*'#')
checkdict = {}

# Checking the format of line_list.txt before attempting to read 
ll_check = np.genfromtxt(linesetname).T
ncol_linedata = len(ll_check)
ncol_line_needed = 11     
if ncol_linedata != ncol_line_needed:
    print("    ERROR in line_list.txt: found " + str(ncol_linedata) + " columns "
       "(" + str(ncol_line_needed) +  " expected). Exiting.")
    sys.exit()

# Read in files
ctrldct = fw.read_control_pars(controlname)
lineinfo = fw.read_data(linesetname, epoch_in)

# Read input files and data
paramspaces: list[pop.Template] = []
for component in paramspace_in:
    paramspaces.append(pop.Template(component, defaultsname))

# Test if inicalc as specified in control.txt is present
test_inidir = ctrldct['inicalcdir']
if test_inidir[-1] != '/':
    print("ERRROR! inicalcdir name as specified in control.txt" + 
        " does not end with '/'. Aborting pre_run_check.")
    sys.exit()
test_inidir = test_inidir[:-1]
if test_inidir not in next(os.walk('.'))[1]:
    print("ERROR! fastwind directory '" + ctrldct['inicalcdir'] +
        "' not found. Aborting pre_run_check.")
    sys.exit()

nfree = sum([len(comp.variables.keys()) for comp in paramspaces])

# Check n_ind
if (ctrldct["nind"]*multiplicity+1)/n_cpu_core % 1.0  > 0.0:
    print("WARNING!! not using all " + str(n_cpu_core) + " cpu's per" 
        " core")
    input('Press enter to ignore...')

print("\nnind = " + str(ctrldct["nind"]))
print("ngen = " + str(ctrldct["ngen"]))

if walltime_flex:
    minutes_str = str(int(math.ceil(float(ctrldct["ngen"])*mnts_gen)))

printsection("FW version")
# Check version 10 vs 11
checkdict["FW version"] = True 
if ctrldct["inicalcdir"].startswith('v10'):
    inic0 = ctrldct["inicalcdir"]
    modelat0 = ctrldct["modelatom"]
    if not os.path.isfile(inic0 + "pnlte_" + modelat0 + ".eo"):
        print("ERROR! " + inic0 + "pnlte_" + modelat0 + ".eo not found")
        checkdict["FW version"] = False
    if not os.path.isfile(inic0 + "pformalsol_" + modelat0 + ".eo"):
        print("ERROR! " + inic0 + "pformalsol_" + modelat0 + ".eo not found")
        checkdict["FW version"] = False
    if modelat0 == 'A10HHe':
        print("WARNING: do you really want to use model atom A10HHe" +
            " in combination with v10?")
        checkdict["FW version"] = False
    for aline in lineinfo[0]:
        if aline.startswith('UV_'):
            print("ERROR: CMF line included in linelist: " + aline)
            print("   This is not possible with v10. Aborting pre-run check")
            sys.exit()
elif ctrldct["inicalcdir"].startswith('v11'):
    if not ctrldct["modelatom"] == 'A10HHe':
        print("ERROR! Use modelatom 'A10HHe' for the CMF version 11 of FW")
        checkdict["FW version"] = False
    for aline in lineinfo[0]:
        if not aline.startswith('UV_'):
            print("WARNING: non CMF line included in linelist: " + aline)
            print("   This is not yet tested for v11.")
            continue
        strt0, stop0 = aline.split('_')[1:]
        if len(strt0) == 5 and len(stop0) == 5:
            if not float(stop0) > float(strt0):
                print("ERROR: format of UV line not correct: " + aline)
                print("Stop wavelength should exceed start wavelength")
                print("Exiting")
                sys.exit()
        else:
            print("ERROR: format of UV line not correct: " + aline)
            print("  Format should be: UV_xxxxx_yyyyy with x and y start" +
            " start stop wavelengths \n   of the lines in Angstrom, padded" +
            " with leading zeros to 5 digit numbers. \n   e.g. " +
            " UV_01211_01625 for the range 1211-1625 Ansgtrom. \n   Exiting.")
            sys.exit()
else:
    print("WARNING: Inicalc dir does not start with v10 or v11: " +
        "no proper checks can be carried out on input files")
    checkdict["FW version"] = False

if checkdict["FW version"]:
    print("FW version ok")
else:
    print("Potential problems with FW version/model atom")
    print("   Check modelatom and inicalcdir combination")
    
# Check mutation rate parameters
printsection('Mutation rate')
checkdict["Mutation"] = True
mut_adjust_type = ctrldct["mut_adjust_type"]
if not mut_adjust_type in ('constant', 'charbonneau', 'autocharb'):
    print('ERROR: mut_adjust_type unknown: ' + mut_adjust_type)
    checkdict["Mutation"] = False

fit_cutoff_min_charb = ctrldct["fit_cutoff_min_charb"]
fit_cutoff_max_charb = ctrldct["fit_cutoff_max_charb"]
w_gauss_na = ctrldct['w_gauss_na']
w_gauss_br = ctrldct['w_gauss_br']
b_gauss_na = ctrldct['b_gauss_na']
b_gauss_br = ctrldct['b_gauss_br']
type_br = ctrldct["broad_type"]
type_na = ctrldct["narrow_type"]

print("Narrow type: " + type_na)
print("Broad type : " + type_br)
if w_gauss_br < w_gauss_na and type_br == type_na:
    print('ERROR: narrow gaussian is broader than wide gaussian')
    checkdict["Mutation"] = False
    print('   - Narrow gauss width: ' + str(w_gauss_na))
    print('   - Broad gauss width : ' + str(w_gauss_br))
if w_gauss_br > 0.50 and type_br == 'frac':
    print('WARNING: high value for w_gauss_br')
    checkdict["Mutation"] = False
    print('   - Broad gauss width : ' + str(w_gauss_br))
if w_gauss_na > 2.50 and type_na == 'step':
    print('WARNING: high value for w_gauss_na')
    checkdict["Mutation"] = False
    print('   - Narrow gauss width: ' + str(w_gauss_na))
if w_gauss_na > 0.05 and type_na == 'frac':
    print('WARNING: high value for w_gauss_na')
    checkdict["Mutation"] = False
    print('   - Narrow gauss width: ' + str(w_gauss_na))
if b_gauss_br > 0.15:
    print('WARNING: high value for b_gauss_br')
    checkdict["Mutation"] = False
    print('   - Braod gauss base  : ' + str(b_gauss_br))
if b_gauss_na != 0.0:
    print('WARNING: b_gauss_na is nonzero.')
    print('  This means it will cause mutations far away from current value.')
    print('   - Narrow gauss base : ' + str(b_gauss_br))
    checkdict["Mutation"] = False

if ctrldct["use_string"] in ('yes', 'y', 'True', True):
    checkdict["Mutation"] = False
    print('WARNING: using Charbonneau scheme with strings. '
       'This will make the run slower')
    if int(ctrldct["sigs_string"]) != 2:
        print('WARNING: significant digits different than in pyGA')
    if (float(ctrldct["fracdouble_string"]) < 0.0 or 
            float(ctrldct["fracdouble_string"]) > 1.0):
        print('WARNING: fracdouble_string should have a value 0 <= value <= 1.0, ' 
            '\n current value is ' + str(ctrldct["fracdouble_string"]))


if not abs(float(ctrldct["mut_rate_max"]) - 2./nfree) < 0.001:
    print("2/nfree =! mut_rate_max")
    print("   - 2/nfree      = "+ str(round(2./nfree,3)))
    print("   - mut_rate_max = " + str(ctrldct["mut_rate_max"]))
    checkdict["Mutation"] = False

if checkdict["Mutation"] == True:
    print('No suspicious things found in mutation parameters.')

mut_adj_type = ctrldct["mut_adjust_type"]
if mut_adj_type == 'autocharb':
    printsection('Auto adaption charbonneau limits')
    ac_fit_a = ctrldct["ac_fit_a"]
    ac_fit_b = ctrldct["ac_fit_b"]
    ac_max_factor = ctrldct["ac_max_factor"]
    ac_maxgen = ctrldct["ac_maxgen"]
    ac_maxgen_min = ctrldct["ac_maxgen_min"]
    ac_maxgen_max = ctrldct["ac_maxgen_max"]
    ac_lowerlim = ctrldct["ac_lowerlim"]
    ac_upperlim = ctrldct["ac_upperlim"]
    print('ac_fit_a        ' + ac_fit_a)
    print('ac_fit_b        ' + ac_fit_b)
    print('ac_max_factor   ' + ac_max_factor)
    print('ac_maxgen       ' + ac_maxgen)
    print('ac_maxgen_min   ' + ac_maxgen_min)
    print('ac_maxgen_max   ' + ac_maxgen_max)
    print('ac_lowerlim     ' + ac_lowerlim)
    print('ac_upperlim     ' + ac_upperlim)

printsection('Line list')
# Checks on line_list.txt
checkdict["Line list"] = True
for ii in range(len(lineinfo[0])):
    lname0 = lineinfo[0][ii]
    lineweight = ll_check[9][ii]
    uvres0 = ll_check[10][ii]
    uvlb0 = ll_check[2][ii]
    uvrb0 = ll_check[3][ii]
    norml0 = ll_check[6][ii]
    normr0 = ll_check[8][ii]
    rvcorr0 = ll_check[4][ii]
    if norml0 != 0.0 or normr0 != 0.0:
        print("WARNING: normalisation correction used for line " + lname0 +
              ".\n    This should work in principle, but is not extensively " +
              "tested.\n    Please carefully check the output of this run.")
        checkdict["Line list"] = False
    if rvcorr0 != 0.0:
        print("WARNING: radial velocity correction used for line " + lname0 +
              ".\n    This should work in principle, but is not extensively " +
              "tested.\n    Please carefully check the output of this run.")
        checkdict["Line list"] = False
    if lname0.startswith('UV_'):
        if not uvres0 > 0:
            print('ERROR! Line ' + lname0 + ' needs a stepsize > 0 Angstrom')
            print('    Please edit column 11 of line_list.txt. Exiting.')
            sys.exit()
        elif uvres0 < 0.05:
            print('WARNING: Line ' + lname0 + ' will be computed with a step ' +
                  'size of ' + str(uvres0) + '\n   Computation may be slow. ')
            checkdict["Line list"] = False
        elif uvres0 > 1.0:
            print('WARNING: Line ' + lname0 + ' will be computed with a step ' +
                  'size of ' + str(uvres0) + '\n   Is this small enough? ')
            checkdict["Line list"] = False
        nameL = float(lname0.split('_')[1])
        nameR = float(lname0.split('_')[2])
        if (nameL - uvlb0 < -larger_wavemax or uvrb0 - nameR < -larger_wavemax):
            print("WARNING: wavelength range specified do not correspond " +
                  "to name of CMF line: \n    " + lname0 +
                  " (lbound=" + str(uvlb0) + ", rbound=" + str(uvrb0) + ")" +
                  "\n    Model region much larger than fitting region!")
            checkdict["Line list"] = False
        if (uvlb0 - nameL < 0.0 or nameR - uvrb0 < 0.0):
            print("ERROR: wavelength range specified do not correspond " +
                  "to name of CMF line: \n    " + lname0 +
                  " (lbound=" + str(uvlb0) + ", rbound=" + str(uvrb0) + ")" +
                  "\n    Model region does not cover fitting region!")
            checkdict["Line list"] = False
    elif not uvres0 == 0:
        print('ERROR for line_list.txt, line: ' + lname0)
        print('Column 11 of line_list.txt should equal 0.0 for ' +
              ' a non CMF line.')
        checkdict["Line list"] = False
    if lineweight != 1.0 and ctrldct["fitmeasure"] == 'chi2':
        print("WARNING! Line weight of " + lname0 + " != 1.0, but " +
              "fitmeasure is chi2. \n     Lineweight has no effect: " +
              " \n      choose fitness as fitmeasure or set all weights to 1.0")
        checkdict["Line list"] = False

if checkdict["Line list"]:
    print("Line list seems ok, but please check also the plot")

# Checks on parameter space
printsection('Parameter spaces')
print('Number of free parameters: ' + str(nfree))
checkdict["Parameter space"] = True

if multiplicity > 2:
    print("WARNING: multiplicity is " + str(multiplicity) +
          ".\n    This should work in principle, but is not extensively " +
          "tested.\n    Please carefully check the output of this run.")
    checkdict["Parameter space"] = False


for paramspace in paramspaces:
    printsection('Next parameter space')
    for pn in paramspace.variables.keys():
        print('   - ' + pn)

    all_names = np.concatenate((list(paramspace.variables.keys()), list(paramspace.fixed.keys())))
    free_names = paramspace.variables.keys()
    nonfree_names = paramspace.fixed.keys()
    nonfree_vals = paramspace.fixed.values()
    free_dict = paramspace.variables
    fixed_dict = paramspace.fixed

    # Check if all elements are in defaults
    # If an addional parameter is needed in an update of the GA, then
    # add it to this list to ensure a run does not crash if you copy
    # an old defaults_fastwind.txt file
    fastwind_def_complete = ['C',
                             'N',
                             'O',
                             'Si',
                             'Mg',
                             'P',
                             'Fe',
                             'S',
                             'Na',
                             'Ca',
                             'optne_update',
                             'he_one',
                             'it_start',
                             'itmore',
                             'optmixed',
                             'rmax',
                             'tmin',
                             'vmin_start',
                             'beta',
                             'vdiv',
                             'ihe_start',
                             'optmod',
                             'opttlucy',
                             'megas',
                             'accel',
                             'optcmf',
                             'micro',
                             'metallicity',
                             'lines',
                             'lines_in_model',
                             'enat_cor',
                             'expansion',
                             'set_first',
                             'set_step',
                             'fclump',
                             'logfclump',
                             'vclstart',
                             'vclmax',
                             'vcl',
                             'vcldummy',
                             'opthopf',
                             'do_iescat',
                             'vclmax',
                             'opthopf',
                             'do_iescat',
                             'opthopf',
                             'do_iescat',
                             'do_iescat',
                             'windturb',
                             'fic',
                             'hclump',
                             'fx',
                             'uinfx',
                             'mx',
                             'gamx',
                             'Rinx',
                             'logfx',
                             'xpow',
                             'ratio']

    for a_def_par in fastwind_def_complete:
        #if not ((a_def_par in defnames) or (a_def_par in param_names)):
        if not a_def_par in all_names:
            print("\n\n   ERROR: '" + a_def_par + "' neither in defaults_" +
                  "fastwind.txt nor in parameter_space.txt\n\n")
            sys.exit()

    print('\nFixed values read in from parameter space file: ')
    for key, value in paramspace.fixed.items():
        print('   - ' + key + ': ' + str(value))

    if 'vinf' in free_names:
        if float(free_dict['vinf'][0]) == 0.0:
            checkdict["Parameter space"] = False
            print("ERROR: vinf cannot be 0.0!")
        if float(free_dict['vinf'][1]) == 0.0:
            checkdict["Parameter space"] = False
            print("ERROR: vinf cannot be 0.0!")

    if 'vturb' in all_names:
        checkdict["Parameter space"] = False
        print("ERROR: 'vturb' is not a parameter, use instead: \n  'micro' (for " +
            "micro turbulence), 'macro' (for macro turbulence) or \n  'windturb'" +
            " (for wind turbulence)")

    print('')
    if 'fclump' in free_names and 'logfclump' in free_names:
        checkdict["Parameter space"] = False
        print("ERROR: both fclump and logfclump are free parameters!")
    elif 'fclump' in free_names:
        if float(fixed_dict['logfclump']) < np.log10(1000.0):
            checkdict["Parameter space"] = False
            print("ERROR: fclump free, but logfclump set to < 3.0")
        else:
            print("Using 'fclump' as free clumping parameter")
    elif 'logfclump' in free_names:
        print("Using 'logfclump' as free clumping parameter")
        if float(free_dict['logfclump'][1]) > 2.0:
            checkdict["Parameter space"] = False
            print("logfclump (upper) = " + str(free_dict['logfclump'][1]) + "!")
            print("ERROR: fclump upper limit is very high!")
        if float(free_dict['logfclump'][0]) < 0.0:
            checkdict["Parameter space"] = False
            print("logfclump (lower) = " + str(free_dict['logfclump'][0]) + "!")
            print("ERROR: fclump should never be lower than 1.0!")
    else:
        print('No clumping fitted')
        if 'logfclump' not in all_names:
            print('ERROR: add "logfclump" to defaults_fastwind.txt !!')
            checkdict["Parameter space"] = False

    printsection("X-rays")

    if not ('fx' in all_names and 'logfx' in all_names):
        checkdict["Parameter space"] = False
        print("ERROR! X-rays not specified.")
        if not 'fx' in all_names:
            print("  --> fx missing")
        else:
            print("  --> logfx missing")
        print("Add to params or defaults file")
    else:
        need_xraydetails = False
        if 'fx' in free_names:
            need_xraydetails = True
            print("X-rays included")
            print("   - fx is a free parameter")
            if 'logfx' not in free_names and float(fixed_dict['logfx']) < np.log10(16.0):
                checkdict["Parameter space"] = False
                print("ERROR --> logfx set to < 1.2 but not in use")
        if 'logfx' in free_names:
            need_xraydetails = True
            print("X-rays included")
            print("   - logfx is a free parameter")
            if 'fx' not in free_names and float(fixed_dict['fx']) > 0:
                checkdict["Parameter space"] = False
                print("ERROR --> fx set to > 0 but not in use")
        elif (float(fixed_dict['fx']) > 0.0) or (float(fixed_dict['fx']) < -1000):
            need_xraydetails = True
            print("X-rays included")
            if float(fixed_dict['fx']) > 1000:
                print("   - Estimating fx with Kudritzki 1996 law")
                if float(fixed_dict['xpow']) > -1000:
                    print("   - Kudritzki estimate cannot be used with the Puls+20")
                    print("     prescription of X-rays!")
                    checkdict["Parameter space"] = False
            if float(fixed_dict['fx']) < -1000:
                print("   - Estimating fx with theoretical law")
                if float(fixed_dict['xpow']) > -1000:
                    print("   - Kudritzki estimate cannot be used with the Puls+20")
                    print("     prescription of X-rays!")
                    checkdict["Parameter space"] = False
            else:
                print("   - fx is fixed at " + fixed_dict['fx'])
        if need_xraydetails:
            if ctrldct["inicalcdir"].startswith('v11'):
                print("ERROR: X-rays not yet included in GA for CMF version 11.")
                print("Please set fx to -1 and logfx to > 1.3 or change to v10.")
                print("Exiting.")
                sys.exit()
            if 'teff' not in free_names:
                if float(fixed_dict['teff']) < 25000.0:
                    print("ERROR: X-rays included but Teff fixed to < 25000")
                    checkdict["Parameter space"] = False
            else:
                if float(free_dict['teff'][1]) < 25000.0:
                    print("ERROR: X-rays included but full parameter space \n"+
                          "will be treated without X-rays: all Teff < 25000")
                    checkdict["Parameter space"] = False
                elif float(free_dict['teff'][0]) < 25000.0:
                    print("\n\nWARNING: X-rays included but part of parameter "+
                          "space will be treated\nwithout X-rays: Teff range "+
                          "covers Teff < 25000\n\n")
            if not ('gamx' in all_names and 'Rinx' in all_names and 'mx' in all_names
                and 'uinfx' in all_names):
                checkdict["Parameter space"] = False
                print("ERROR: One or more X-ray parameters are missing")
            else:
                print("Other parameters:")
                for apar in ('uinfx', 'mx', 'gamx', 'Rinx'):
                    if apar in free_names:
                        print("   - " + apar + " is a free parameter")
                    else:
                        print("   - " + apar + " = " + fixed_dict[apar])
        else:
            print("X-rays not included")
            for apar in ('uinfx', 'mx', 'gamx', 'Rinx'):
                if apar in free_names:
                    checkdict["Parameter space"] = False
                    print("ERROR: " + apar + " is a free parameteter but " +
                        "fx is 0.0!")
        if 'fx' in free_names and 'logfx' in free_names:
            checkdict["Parameter space"] = False
            print("ERROR: both fx and logfx are free parameters")
        elif 'fx' in free_names and fixed_dict['logfx'] <= np.log(16.0):
            checkdict["Parameter space"] = False
            print("ERROR: fx is a free parameter, but logfx is in use")
            print("   value of logfx = " + str(fixed_dict['logfx']))
        elif 'logfx' in free_names and float(fixed_dict['fx']) > 1000.0:
            checkdict["Parameter space"] = False
            print("ERROR: logfx is a free parameter, but fx is set to be chosen ")
            print("       according to the Kudritzki law (fx > 1000).")
        elif 'logfx' in free_names:
            if float(free_dict['logfx'][1]) >= np.log10(16.0):
                checkdict["Parameter space"] = False
                print("ERROR: please set logfx upperbound no larger than 1.2")


    if 'fic' not in free_names:
        if 'fic' in nonfree_names:
            if fixed_dict['fic'] == -1:
                tparams = ''
                tlen = 0
                print('WARNING! fic in defaults set to -1')
                print('This means a fixed value of 10^-1 = 0.1 is used')
                print('If you intent to use optically thin clumping,')
                print('Set this value to 999 and rerun this script')
                input('If you intent to fix fic to 0.01, press enter')
            else:
                tparams = ''
                tlen = -10
        else:
            tparams = ''
            tlen = 0
        for aparam in free_names:
            #if aparam in ('vclstart', 'vclmax', 'fvel', 'h'):
            if aparam in ('fvel', 'h'):
                tparams = tparams + '"' + aparam + '" '
                tlen = tlen + 1
        if tlen == 1:
            tparams = tparams + 'is'
        else:
            tparams = tparams + 'are'
        if tlen > 0:
            print('ERROR: no thick clumping, but ' + tparams + ' varied!')
            checkdict["Parameter space"] = False
    if 'vcl' in free_names and ('vclstart' in  free_names or 'vclmax' in free_names):
        print('ERROR: vclstart and vcl cannot be fitted simultaneously')
        print('       vcl > 0: use linear step function clumping law')
        print('                vclstart not used')
        print('       vcl > -1: use linear step function clumping law')
        print('                vcl not used')
        checkdict["Parameter space"] = False

    if ctrldct["inicalcdir"].startswith("v11"):
        if 'fic' in free_names:
            print("ERROR: optically thick clumping not in CMF version 11!")
            print("Exiting")
            sys.exit()
        elif 'fic' in free_names:
             if fixed_dict['fic'] != -1:
                 print("ERROR: optically thick clumping not in CMF" +
                      " version 11! fic != -1. Exiting.")
                 sys.exit()

    if checkdict["Parameter space"]:
        print('\nParameter space ok.')

    # Check stepsizes in parameter space
    warning_na = 3.0
    printsection('Step sizes')
    checkdict['Step size'] = True
    for par in free_names:
        stepsize = float(free_dict[par][2])
        if stepsize > 0.0:
            pwidth = np.abs(float(free_dict[par][1]) - float(free_dict[par][0]))
            if np.abs(pwidth/stepsize - np.round(pwidth/stepsize,0)) > 0.0001:
                print('   - ' + par + ': ERROR: step size cannot divide'
                    ' parameter range into equal parts')
                print('     start: ' + str(free_dict[par][0]) + ', stop: ' +
                     str(free_dict[par][1]) + ', step: ',  str(free_dict[par][2]) )
                print(np.abs(pwidth/stepsize))
                print(int(pwidth/stepsize))
                checkdict['Step size'] = False
            stepfrac = 1.0*stepsize/pwidth
            if stepsize < 0:
                checkdict['Step size'] = False
                print('  - ' + par + ': ERROR: negative stepsize')
            if stepsize == 0.0:
                checkdict['Step size'] = False
                print('  - ' + par + ': ERROR: stepsize = 0.0')
            if stepfrac*warning_na > w_gauss_na and type_na == 'frac':
                print('  - ' + par + ': WARNING: stepsize is large ' +
                    'compared to narrow mutation width')
                print('    ' + 'ratio is ' + str(round(w_gauss_na/stepfrac,2)) +
                     ' while it is preferrably ~>' + str(warning_na) +
                     '\n    Current step size: ' + str(stepsize) +
                     '\n    --> consider to decrease step size or increase w_gauss_na')
                checkdict['Step size'] = False
        else:
            print("ERROR: stepsize must exceed 0.0")
            checkdict['Step size'] = False

    if checkdict['Step size'] == True:
        print('All stepsizes ok.')

# Check wheter all lines in the line list are present in the 
# FORMAL_INPUT master file 
printsection('Formal Input')
print(ctrldct["inicalcdir"])
tf_formal = fw.create_FORMAL_INPUT(ctrldct["inicalcdir"], lineinfo[0], 
    linesetname,create=False)
checkdict["FORMAL_INPUT"] = tf_formal

# Check properties reinsertion scheme
printsection('Reinsertion scheme')
ratio_po = ctrldct['ratio_po']
f_parent = ctrldct['f_parent']
if ratio_po == 1.0 and f_parent == 0.0:
    print('You will be using pure reinsertion')
else:
    print('You will be using elitist/fitness based reinsertion scheme')
if f_parent >= 1.0 - (1.0 / ratio_po):
    print('Ratios of parents/offspring are ok.')
    checkdict['Reinsertion'] = True
else:   
    print('ERROR in reinsertion ratios!')
    print('  ratio_po = ' + str(ratio_po))
    print('  f_parent = ' + str(f_parent))
    print('Must satisfy: f_parent >= 1-1/ratio_po')
    checkdict['Reinsertion'] = False

# Print summary of the results
sumstr = '  Summary of pre-run checks:'
sumstr = sumstr + (nd - len(sumstr) - 5)*' ' + '##'
print('\n' + nd*'#') 
print("##" + (nd-4)*' ' + '##')
print('## ' + sumstr)
print("##" + (nd-4)*' ' + '##')
for i in checkdict:
    if checkdict[i]:
        printsumline(i, 'all ok.', nd)
    else:
        printsumline(i, 'irregularities found --> CHECK INPUT!', nd)
print("##" + (nd-4)*' ' + '##')
print(nd*'#')

if not checkdict['Step size']:
    print("\nIrregularities found in stepsize: cannot make plots")
    sys.exit()

###################################
#        Plot param space         #
###################################
print('\nMaking plots...')
component = 0
for paramspace in paramspaces:
    pnames = list(paramspace.variables.keys())
    ncols = 3
    nrows =int(math.ceil(1.0*len(pnames)/ncols))
    nrows =max(nrows, 2)

    w_gauss_na = ctrldct["w_gauss_na"]
    w_gauss_br = ctrldct["w_gauss_br"]
    b_gauss_na = ctrldct["b_gauss_na"]
    b_gauss_br = ctrldct["b_gauss_br"]
    do_doublebroad = ctrldct["doublebroad"]
    nbins=[]
    ccol = ncols - 1
    crow = -1
    props = dict(facecolor='white', alpha=1.0)
    fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    for i in range(ncols*nrows):
        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= len(pnames):
            ax[crow,ccol].axis('off')
            continue

        par = pnames[i]

        start = float(paramspace.variables[par][0])
        stop = float(paramspace.variables[par][1])
        step = float(paramspace.variables[par][2])
        width = stop - start
        nbin = int((width/step) + 1)
        nbins.append(nbin)
        vlines = np.linspace(start, stop, nbin)

        if ctrldct["narrow_type"] == 'frac':
            sig_na = w_gauss_na * width
        elif ctrldct["narrow_type"] == 'step':
            sig_na = w_gauss_na * step
        sig_br = w_gauss_br * width
        b_na = b_gauss_na
        b_br = b_gauss_br
        xgauss = np.linspace(start, stop, 1000)
        na_gauss = pop.Component.gauss(xgauss, b_na, 1.0, start+int(nbin/2)*step, sig_na)
        if do_doublebroad == 'yes':
            br_gauss = pop.Component.double_gauss(xgauss, b_br, 1.0, start+int(nbin/2)*step,
                sig_br)
        else:
            br_gauss = pop.Component.gauss(xgauss, b_br, 1.0, start+int(nbin/2)*step, sig_br)
        br_gauss = br_gauss / max(br_gauss)
        na_gauss = na_gauss / max(na_gauss)

        for vline in vlines:
            ax[crow,ccol].axvline(vline, lw=1.0, alpha=0.5)
        if ctrldct['use_string'] not in ('y', 'yes', 'True', True):
            ax[crow,ccol].plot(xgauss, na_gauss, 'red', lw=2)
            ax[crow,ccol].plot(xgauss, br_gauss, 'orange', lw=2)
        ax[crow,ccol].set_xlim(start-3*step, stop+3*step)
        ax[crow,ccol].set_title(par)
        ax[crow,ccol].set_yticks([])
        boxtext = 'Step size: ' + str(step) + ',\n nsteps: ' + str(len(vlines))
        ax[crow,ccol].text(0.05, 0.88, boxtext,
            transform=ax[crow,ccol].transAxes, fontsize=9, bbox=props)

    fig.suptitle('Parameter space ' + str(component) + ' check: ' + run_name , fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(param_fig)
    reportPDF.savefig()
    plt.close()

    print('\n       >  Plotted parameter space and mutation distributions of component ' + str(component))
    component+=1

###################################
#          Plot spectrum          #
###################################

lnames, line_params = fw.read_linelist(linesetname)
names, res, epochs, lweight = lineinfo

left_bounds = line_params[1]
right_bounds = line_params[2]
radial_vels = line_params[3]

epochCount = 0
for epoch in epochs:
    activeNames = epoch.get_active_lines()
    idx = np.where(np.isin(names, activeNames))[0]
    activeRes = res[idx]
    activeWeight = lweight[idx]
    activeLbounds = left_bounds[idx]
    activeRbounds = right_bounds[idx]
    ncols = 4
    nrows =int(math.ceil(1.0*len(activeNames)/ncols))
    nrows =max(nrows, 2)

    ccol = ncols - 1
    crow = -1
    props = dict(facecolor='white', alpha=1.0)
    props2 = dict(facecolor='darkblue', alpha=0.8)
    errprops = dict(facecolor='red', alpha=1.0)
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    extra_AA = 2.0
    specwave, specflux, specerr = epoch.wave, epoch.norm, epoch.error
    if np.count_nonzero(np.isnan(specflux)) > 0:
        print("ERROR! Spectrum contains one or more NaN values")
        print("   At wavelengths:")
        for nanwave in specwave[np.isnan(specflux)]:
            print("    * " + str(round(nanwave,3)))
        print("\n> Could not make plot for " + run_name + ", exiting...")
        sys.exit()
    specmin, specmax = min(specflux), max(specflux)
    for i in range(ncols*nrows):
        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= len(activeNames):
            ax[crow,ccol].axis('off')
            continue

        linename = activeNames[i]


        # not implemented
        # for j in range(len(lnames)):
        #     if lnames[j] == names[i]:
        #         lbound = fw.rvshift(left_bounds[j], radial_vels[j])
        #         rbound = fw.rvshift(right_bounds[j], radial_vels[j])
        #         the_rv = radial_vels[j]

        wave, flux, err = epoch.get_line_data(linename)
        resolution = activeRes[i]
        weight = activeWeight[i]
        lbound = activeLbounds[i]
        rbound = activeRbounds[i]

        # not implemented
        # specwave_rv = fw.rvshift(specwave, the_rv)
        # swave, sflux, serr = fw.parallelcrop(specwave_rv, specflux, specerr,
        #     lbound-extra_AA, rbound+extra_AA)

        npoint_error = False
        if not len(wave) > nfree + 1:
            print('\nERROR! Too few data points for this line compared to ' 
                'the amount of free parameters.')
            print('Line: ' + linename)
            print('Data points: ' + str(len(wave)))
            print('Free params: ' + str(nfree)+'\n')
            npoint_error = True

        if npoint_error:
            ax[crow,ccol].axvspan(lbound,rbound, color='red', alpha=0.6)
        elif weight != 1.0:
            ax[crow,ccol].axvspan(lbound,rbound, color='green', alpha=0.3)
        else:
            ax[crow,ccol].axvspan(lbound,rbound, color='gold', alpha=0.3)
        # if the_rv != 0.0:
        #     ax[crow, ccol].text(0.50, 0.9, "rv = " + str(the_rv) + ' km/s',
        #         transform=ax[crow,ccol].transAxes, fontsize=8, bbox=props2,
        #         ha='center', color='white')
        # ax[crow,ccol].errorbar(swave, sflux, yerr=serr, ls='', fmt='o',
        #     color='C0', alpha=0.5, markersize=3.0)
        ax[crow,ccol].errorbar(wave, flux, yerr=err, ls='', fmt='o',
            color='darkblue', markersize=3.0)
        ax[crow,ccol].set_ylim(specmin, specmax)
        ax[crow,ccol].set_title(linename)
        ax[crow,ccol].axhline(1.0, lw=1.0, color='grey')
        ax[crow,ccol].axvline(lbound, lw=1.0, color='red')
        ax[crow,ccol].axvline(rbound, lw=1.0, color='red')
        boxtext = 'R = ' + str(resolution) + ',\n weight = ' + str(weight)
        ax[crow,ccol].text(0.05, 0.05, boxtext,
            transform=ax[crow,ccol].transAxes, fontsize=8, bbox=props)
        if npoint_error:
            errortext = 'Not enough data points\n compared to #free parameters'
            ax[crow,ccol].text(0.5, 0.5, errortext, ha='center',
                transform=ax[crow,ccol].transAxes, fontsize=8, bbox=errprops,
                color='white')

    fig.suptitle('Spectrum check epoch ' + str(epochCount) + ': ' + run_name, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(spectrum_fig)
    reportPDF.savefig()
    plt.close()

    bc = 0
    for ii in range(len(ll_check[1])):
        if ll_check[3][ii] - ll_check[2][ii] > wide_specrange_lim:
            bc = bc + 1

    if bc > 0:
        plc = -1
        fig, ax = plt.subplots(bc, 1, figsize=(3*ncols, 2.0*bc))
        for ii in range(len(activeNames)):
            lname0 = activeNames[ii]
            lbound = activeLbounds[ii]
            rbound = activeRbounds[ii]
            lwidth = rbound - lbound
            if lwidth > wide_specrange_lim:
                plc = plc + 1

                # for j in range(len(lnames)):
                #     if lnames[j] == lname0:
                #         lbound = fw.rvshift(left_bounds[j], radial_vels[j])
                #         rbound = fw.rvshift(right_bounds[j], radial_vels[j])
                #         the_rv = radial_vels[j]
                wave, flux, err = epoch.get_line_data(lname0)
                resolution = res[ii]
                weight = lweight[ii]

                # specwave_rv = fw.rvshift(specwave, the_rv)
                # swave, sflux, serr = fw.parallelcrop(specwave_rv, specflux,
                #     specerr, lbound-extra_AA, rbound+extra_AA)

                if bc > 1:
                    # ax[plc].errorbar(swave, sflux, yerr=serr, ls='', fmt='o',
                    #     color='C0', alpha=0.5, markersize=3.0)
                    ax[plc].errorbar(wave, flux, yerr=err, ls='', fmt='o',
                        color='darkblue', markersize=3.0)
                    ax[plc].axhline(1.0, color='black', lw=1.0, zorder=0)
                    ax[plc].set_title(lname0)
                else:
                    # ax.errorbar(swave, sflux, yerr=serr, ls='', fmt='o',
                    #     color='C0', alpha=0.5, markersize=1.0)
                    ax.errorbar(wave, flux, yerr=err, ls='', fmt='o',
                        color='darkblue', markersize=1.0)
                    ax.axhline(1.0, color='black', lw=1.0, zorder=0)
                    ax.set_title(lname0)

        plt.suptitle('Spectrum: larger plot for wide spectral regions')
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(spectrum_fig2)
        pdfs.append(spectrum_fig2)
        reportPDF.savefig()

    print('       >  Plotted spectrum and line selection for epoch ' + str(epochCount))
    epochCount += 1

reportPDF.close()

for pdf in pdfs:
    os.remove(pdf)

n_node = int((ctrldct["nind"]+1)/n_cpu_core)
n_cpu = ctrldct["nind"]+1
if n_node > 1:
    ucx_string = "UCX_Settings='-x UCX_NET_DEVICES=mlx5_0:1'"
    run_string = '    mpiexec -n $ncpu python3 kiwiGA.py ${runname}'
else:
    ucx_string = ''
    run_string = '    srun --mpi=pmi2 -n $ncpu python3 kiwiGA.py ${runname}'


jobscript = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --time={minutes_str}
#SBATCH --ntasks={ci.cores_per_node}
#SBATCH --cpus-per-task=1
{ci.extra_sbatch % (run_name, run_name)}

runname={run_name}
do_restart={do_restart}
ncpu={str(ctrldct["nind"]*2+1)}
inidir={test_inidir}

echo Run ${{runname}}
echo Using $ncpu CPUs
echo Do restart? $do_restart

# Load modules
{ci.modules}

# Define paths
scratch=/{ci.scratch_loc}/{username}/${{runname}}/
homedir=/{ci.home_loc}/{username}/{codedir}/
results=/{ci.home_loc}/{username}/{codedir}/runs/${{runname}}/

echo Deleting and rebuilding scratch

# Delete scratch and copying back the needed files
rm -rf $scratch
mkdir -p $scratch
mkdir -p ${{scratch}}output/
cp -r ${{homedir}}*.py $scratch
cp -r ${{homedir}}filter_transmissions $scratch
mkdir -p ${{scratch}}input/
mkdir -p ${{scratch}}input/${{runname}}/
cp -r ${{homedir}}input/${{runname}}/* ${{scratch}}input/${{runname}}/.
cp -r ${{homedir}}${{inidir}} $scratch


echo Puting results back in case of a restart

if [ "$do_restart" == "yes" ]
then
    cp -r $results/saved ${{scratch}}output/
    cp $results/mutation_by_gen.txt ${{scratch}}output/
    cp $results/redchi2s_cont.txt ${{scratch}}output/
    cp $results/savefitness_cont.txt ${{scratch}}output/
    cp $results/chi2_cont.csv ${{scratch}}output/
    cp $results/precomp_cont.csv ${{scratch}}output/
    cp $results/savegen_cont.pkl ${{scratch}}output/
else
    rm -rf $results
    mkdir $results
fi

# Navigate to computation directory
cd $scratch

echo Starting run!
date

# Start run
if [ "$do_restart" == "yes" ]
then
    echo ...restarting run
    {run_string} -c
else
    echo ... starting run
    {run_string}
fi

date
echo ... Run ENDED!

echo Copying relevant results:
rm -rf output/run/
cp -r output/. $results

echo Deleting scratch
cd $homedir
rm -rf $scratch
"""

f = open(jobscriptfile, "w")
f.write(jobscript)
f.close()

if do_restart == 'no':
    print('\nCreated ' + jobscriptfile + ' --- NO restart')
if do_restart == 'yes':
    print('\nCreated ' + jobscriptfile + ' --- WITH RESTART!')
print('\nEnd of pre-run check of ' + run_name)




