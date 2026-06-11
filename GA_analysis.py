# Script for visualising the run output of Kiwi-GA
# Created by Sarah Brands @ 29 July 2022
import shutil
import time

from func_GA_analysis import load_snapshot

init_start = time.time()
from schwimmbad import MPIPool
import sys
import os
import numpy as np
import pandas as pd
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import func_GA_analysis as fga
import fastwind_wrapper as fw
import paths as ppp
import population as pop
from epoch import Epoch
import functools

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
###############################################################################
#  Parse arguments (e.g. runname)
###############################################################################
print("Imported packages in " + str(round(time.time()-init_start, 2)) + " seconds.")
start = time.time()
print("Initializing GA_analysis... " , end = '', flush = True)


parser = argparse.ArgumentParser()
parser.add_argument('runname', help='Specify a run name')
parser.add_argument('-full', help='Create all plots possible',
    action='store_true', default=False)
parser.add_argument('-prof', help='Plot line profiles',
    action='store_true', default=False)
parser.add_argument('-fast', help='Only make fitness plot and title page',
    action='store_true', default=False)
parser.add_argument('-open', help='After making the report, open it.',
    action='store_true', default=False)
parser.add_argument('-param', help='Makes fitness per line plots grouped by' +
    'parameter (not part of -full)',
    action='store_true', default=False)
parser.add_argument('-fitness', help='Add fitness plot with fitness ' +
    '(in addition to 1/chi2)',
    action='store_true', default=False)
parser.add_argument('-latex', help='Makes a latex table on the title page',
    action='store_true', default=False)
parser.add_argument('-radius', help='Correct radius (compute FW model)',
    action='store_true', default=False)
parser.add_argument('-maxgen', type=int, help="If 1 number is specified, " +
    "the last generation to consider. If 2 numbers are supplied the first " +
    "and last generation to consider",
    default=[], nargs="*")
parser.add_argument('-em', help='Add an extra FW model to the line profile'  +
    ' plots. Supply here the path to the spectrum, which should be an asci ' +
    'file containing two columns (wavelength (angstrom) and normalised flux).',
    default='')
parser.add_argument('-efm', help='Add an extra FW model to the line  ' +
    'profile plots. Supply here the path to the directory. The directory ' +
    'should contain (broadened) line profiles with names as in the GA run ' +
    '+ ending on .prof, e.g. "HALPHA.prof". The files schould be asci ' +
    'format and have two columns: wavelength (angstrom) and normalised flux. ',
    default='')
parser.add_argument('-line', help='Plot the fitness plots per line',
    action='store_true', default=False)
parser.add_argument('-corr', help='Plot the correlation plots of the given parameters',
    action='store_true', default=False)
parser.add_argument('-conv', help='Plot the convergence plots of the variable parameters.',
    action='store_true', default=False)
parser.add_argument('-perf', help='Plot the fastwind performance plots.',
    action='store_true', default=False)
parser.add_argument('-comp', help='Plot the line profiles including component brakedown.',
    action='store_true', default=False)

args = parser.parse_args()

runname = args.runname

###############################################################################
#  Definition of paths & files
###############################################################################


datapath = ppp.datapath_analysis
outpath = ppp.outpath_analysis + runname + '/output/'
fastwind_local = ppp.fastwind_local

pdfname = outpath + runname + '.pdf'
if args.full:
    pdfname = outpath + runname + '_full.pdf'
if args.fast:
    pdfname = outpath + runname + '_fast.pdf'
if args.param:
    pdfname = outpath + runname + '_param.pdf'

plotlineprofdir = outpath + 'lineprofs/'
fw.mkdir(outpath)
fw.mkdir(plotlineprofdir)

inputcopydir = datapath + 'input_copy/'
savedmoddir = datapath + 'saved/'

thechi2file = datapath + 'chi2.csv'
precomputefile = datapath + 'check_precomputed.csv'
thebestchi2file = datapath + 'best_chi2.txt'
themutgenfile = datapath + 'mutation_by_gen.txt'
thecontrolfile = inputcopydir + 'control.txt'
thelinefile = inputcopydir + 'line_list.txt'
theparamfile = inputcopydir + 'parameter_space.txt'
paramspacecomponentfiles = os.listdir(inputcopydir + 'components/')
epochfiles = os.listdir(inputcopydir + 'epochs/')
paramspace_in = [inputcopydir + 'components/' + component for component in paramspacecomponentfiles]
epoch_in = [inputcopydir + 'epochs/' + epoch for epoch in epochfiles]
multiplicity = len(paramspace_in)
theradiusfile = inputcopydir + 'radius_info.txt'
thefwdefaultfile = inputcopydir + 'defaults_fastwind.txt'
savebestfilename = runname + '_bestvals.txt'
savebestindatfileloc = outpath

compsavedir = outpath + 'components/'
fw.mkdir(compsavedir)

for i in range(multiplicity):
    comp = compsavedir + str(i) + '/'
    fw.mkdir(comp)
    fw.mkdir(comp + 'lineprofs/')

extra_fwmod = args.efm
if not extra_fwmod.endswith('/'):
    extra_fwmod = extra_fwmod + '/'
extra_mod = args.em

###############################################################################
#  Read GA output files
###############################################################################

print("Complete in " + str(round(time.time()-start, 2)) + " seconds.")

print("Generating report for << " + runname + " >>")

start = time.time()
print("Reading output files... " , end = '', flush = True)



# Read chi2.csv into pandas dataframe
df = pd.read_csv(thechi2file)

precompute = pd.read_csv(precomputefile)

# Finish dataframe with proper units for run_id's and generation numbers
mingen = 0
df['gen'] = df['gen'].astype(int)
if len(args.maxgen) == 0:
    maxgen = np.max(df['gen']) + 1
elif len(args.maxgen) == 1:
    maxgen = args.maxgen[0]
    df = df[df["gen"] <= args.maxgen[0]]
    pdfname = pdfname[:-4] + "_gen_%i.pdf" % args.maxgen[0]
elif len(args.maxgen) == 2:
    df = df[(df["gen"] >= args.maxgen[0]) * (df["gen"] <= args.maxgen[1])]
    pdfname = pdfname[:-4] + "_gen_%i-%i.pdf" % (args.maxgen[0], args.maxgen[1])
    mingen = args.maxgen[0]
    maxgen = args.maxgen[1]
else:
    print("Too many arguments for -maxgen!")
    exit()

df_orig = df.copy()

load_snapshot = functools.partial(fga.load_snapshot, savedmoddir)

snapshot_populations: list[pop.Population] = list(pool.map(load_snapshot, list(range(mingen, maxgen))))

# Read spectrum
spectra: list[Epoch] = []
names, llp = fw.read_linelist(thelinefile)
res, lbound, rbound, rv, normlx, normly, normrx, normry, lw, ang = llp
for epoch in epoch_in:
    spectra.append(Epoch(epoch, (names, lbound, rbound)))

# Read parameter_space
example_individual = snapshot_populations[-1].population[0]
paramspaces: list[pop.Template] = [x.template for x in example_individual.components]

nfree = sum([len(list(x.variables.keys())) for x in paramspaces])

npspec = 0
for epoch in spectra:
    for line in epoch.get_line_names():
        npspec += len(epoch.get_line_data(line)[0])

dof_tot = npspec - nfree

#sanity check:
if not dof_tot == int(df['dof'].iloc[0]):
    print("DOF DOES NOT MATCH, " + str(dof_tot) + " vs " + str(int(df['dof'].iloc[0])))

# Read number of individuals
nind = len(snapshot_populations[-1].population)


# Do radius correction. This is always done if a FW model is present
df, best_indat, best_model_name, best_mod_location = fga.radius_correction(df, fastwind_local, runname,
    thecontrolfile, theradiusfile, datapath, outpath, comp_fw=args.radius)
print("Complete in " + str(round(time.time()-start, 2)) + " seconds.")
start = time.time()
print("Doing general setup... " , end = '', flush = True)

best_gen_name = best_model_name.split('_')[0]
for generation in snapshot_populations:
    if generation.name == best_gen_name:
        best_model =generation.get_individual_with_name(best_model_name)
        break

deriv_pars = fga.more_parameters(snapshot_populations)


###############################################################################
#  Calculate P-value and best fit parameters
###############################################################################

# Compute uncertainties
best_uncertainty, n1sig, n2sig = fga.get_uncertainties(best_model, snapshot_populations,
    npspec, paramspaces, deriv_pars, incl_deriv=True)
np.savetxt(outpath + 'n_1sig_2sig.txt', np.array([n1sig, n2sig]))

# Add additional radius uncertainties
#best_uncertainty = fga.add_anchor_magnitude_uncertainty(df, runname,
#                                                        best_uncertainty,
#                                                       fastwind_local,
#                                                        theradiusfile)

# Unpack all computed values
best_model, bestfamily, params_error_1sig, \
    params_error_2sig, deriv_params_error_1sig, deriv_params_error_2sig, \
    which_statistic = best_uncertainty

# reformat the param values
get_param_dfs = functools.partial(fga.get_relevant_param_df, snapshot_populations, best_model)

param_dfs = list(pool.map(get_param_dfs, list(range(multiplicity))))

###############################################################################
#  Create plots
###############################################################################

print("Complete in " + str(round(time.time()-start, 2)) + " seconds.")

if args.full or args.comp or args.prof:
    start = time.time()
    print("Starting untarring process... ", end = '', flush = True)

    untar_func = functools.partial(fga.unpack_tarfiles, savedmoddir, args.full or args.comp)
    pool.map(untar_func, bestfamily)

    print("Complete in " + str(round(time.time() - start, 2)) + " seconds.")

plot_start = time.time()
print("Starting plotting process... ")

with PdfPages(pdfname) as the_pdf:
    #  Create a title page with best fit parameters
    # if args.latex:
    #     the_pdf = fga.titlepage_latex(df, runname, params_error_1sig,
    #         params_error_2sig, the_pdf, param_names, maxgen, nind, linedct, 1,
    #         deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars)
    # else:
    start = time.time()
    print("     Creating titlepage... " , end = '', flush = True)
    the_pdf = fga.titlepage(df, best_model, runname, params_error_1sig,
        params_error_2sig, the_pdf, maxgen, nind, spectra, 1,
        deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars)
    print("Done in " + str(round(time.time()-start, 2)) + " seconds.")

    # Create overview fitness plot (1/rchi2)
    for i in range(multiplicity):
        start = time.time()
        print("     Creating parameter fitnessplot for component " + str(i) + "... " , end = '', flush = True)
        the_pdf = fga.fitnessplot(param_dfs[i], 'invrchi2', params_error_1sig[i],
                                  params_error_2sig[i], the_pdf, maxgen, list(paramspaces[i].variables.keys()))
        print("Done in " + str(round(time.time()-start, 2)) + " seconds.")
        start = time.time()
        print("     Creating derived parameter fitnessplot for component " + str(i) + "... " , end = '', flush = True)
        the_pdf = fga.fitnessplot(param_dfs[i], 'invrchi2', deriv_params_error_1sig[i],
            deriv_params_error_2sig[i], the_pdf,maxgen, deriv_pars[i])
        print("Done in " + str(round(time.time()-start, 2)) + " seconds.")
    if args.fitness or args.full:
        pass
        # if binary:
        #     the_pdf = fga.fitnessplot(df, 'fitness', params_error_1sig,
        #                               params_error_2sig, the_pdf, param_names[0], param_space[0], maxgen, appendix='_0')
        #     the_pdf = fga.fitnessplot(df, 'fitness', params_error_1sig,
        #                               params_error_2sig, the_pdf, param_names[1], param_space[1], maxgen, appendix='_1')
        # else:
        #     the_pdf = fga.fitnessplot(df, 'fitness', params_error_1sig,
        #         params_error_2sig, the_pdf, param_names, param_space,maxgen)

    if args.prof or args.full:
        #  Create line profile plots
        start = time.time()
        print("     Creating lineprofiles... " , end = '', flush = True)
        for spectrum in spectra:
            the_pdf = fga.lineprofiles(pool, spectrum, savedmoddir,
                best_mod_location, bestfamily, the_pdf,
                plotlineprofdir, extra_fwmod, extra_mod)
        print("Done in " + str(round(time.time()-start, 2)) + " seconds.")

    if multiplicity > 1 and (args.comp or args.full):
        # Line profile plots with components breakdown
        start = time.time()
        print("     Creating lineprofiles with component breakdown... " , end = '', flush = True)
        for spectrum in spectra:
            the_pdf = fga.composite_lineprofiles(pool, best_model, spectrum, savedmoddir,
                best_mod_location, bestfamily, the_pdf, extra_fwmod, extra_mod, multiplicity, compsavedir)
        print("Done in " + str(round(time.time()-start, 2)) + " seconds.")

    # Create convergance plot
    if args.conv or args.full:
        start = time.time()
        print("     Creating convergence plots... ", end='', flush=True)
        for i in range(multiplicity):
            the_pdf = fga.convergence(the_pdf, snapshot_populations, npspec, paramspaces, deriv_pars, i)
        print("Done in " + str(round(time.time() - start, 2)) + " seconds.")

    # Fastwind performance plot
    if args.perf or args.full:
        start = time.time()
        print("     Creating performance plots... ", end='', flush=True)
        for i in range(multiplicity):
            the_pdf = fga.fw_performance(the_pdf, param_dfs[i], thecontrolfile)
        print("Done in " + str(round(time.time() - start, 2)) + " seconds.")

    if not args.fast:
        #  Create correlation plots
        corr_vars = ['teff', 'logg', 'yhe', 'vrot', 'micro']
        if len(corr_vars) > 0 and (args.corr or args.full):
            start = time.time()
            print("     Creating correlation plots... " , end = '', flush = True)
            the_pdf = fga.correlationplot(the_pdf, param_dfs, corr_vars, best_model)
            print("Done in " + str(round(time.time()-start, 2)) + " seconds.")



        # # P-value plot
        # if which_statistic in ('Pval_ncchi2', 'Pval_chi2'):
        #     #  Create overview fitness plot (P-value)
        #     if binary:
        #         the_pdf = fga.fitnessplot(df, 'P-value', params_error_1sig,
        #                                   params_error_2sig, the_pdf, param_names[0], param_space[0], maxgen,
        #                                   appendix='_0')
        #         the_pdf = fga.fitnessplot(df, 'P-value', params_error_1sig,
        #                                   params_error_2sig, the_pdf, param_names[1], param_space[1], maxgen,
        #                                   appendix='_1')
        #     else:
        #         the_pdf = fga.fitnessplot(df, 'P-value', params_error_1sig,
        #                                 params_error_2sig, the_pdf, param_names, param_space,maxgen)

        if args.line or args.full:
            # Create fitness plots per line
            start = time.time()
            print("     Creating fitness plots per line... " , end = '', flush = True)
            for spectrum in spectra:
                 for yval in spectrum.get_active_lines():
                    for i in range(multiplicity):
                         the_pdf = fga.fitnessplot(param_dfs[i], yval, params_error_1sig[i],
                                                   params_error_2sig[i], the_pdf, maxgen, list(paramspaces[i].variables.keys()))
            print("Done in " + str(round(time.time()-start, 2)) + " seconds.")

        if args.param or args.full:
            # Create fitness plots per parameter
            start = time.time()
            print("     Creating fitness plots per parameter... " , end = '', flush = True)
            for i in range(multiplicity):
                for yval in paramspaces[i].variables.keys():
                    for spectrum in spectra:
                         the_pdf = fga.fitnessplot_per_parameter(param_dfs[i], yval, params_error_1sig[i],
                                                       params_error_2sig[i], the_pdf, maxgen, list(spectrum.get_active_lines()))
            print("Done in " + str(round(time.time()-start, 2)) + " seconds.")

print("All plots were created! This took " + str(round(time.time()-plot_start, 2)) + " seconds.")
start = time.time()
print("Saving... " , end = '', flush = True)
fga.save_parameters(param_dfs, compsavedir)
shutil.copy(thelinefile , outpath+'line_list.txt')
for i in range(multiplicity):
    fga.save_bestvals(best_model.components[i], deriv_pars[i], params_error_1sig[i], params_error_2sig[i],
        deriv_params_error_1sig[i], deriv_params_error_2sig[i], compsavedir + str(i) + '/' + savebestfilename)

    shutil.copy(best_indat[i],compsavedir + str(i) + '/' + '/INDAT.DAT_' + str(i))
print("Done in " + str(round(time.time()-start, 2)) + " seconds.")

print('Report saved to ' + pdfname)

print("GA_analysis ran in " + str(round(time.time()-init_start, 2)) + " seconds.")

if args.open:
    os.system('open ' + pdfname)
