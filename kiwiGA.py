import os
import shutil
import sys
import time

import numpy as np
import argparse
import functools
from schwimmbad import MPIPool

import paths as paths
import population as pop
import fastwind_wrapper as fw

''' INITIALIZE / SET UP '''
program_start_time = time.time()
#Start MPIPool to control the distrubution of models over CPUs
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

# Read command line arguments and exit if no input is found.
parser = argparse.ArgumentParser(description='Run Kiwi GA')
parser.add_argument('runname', help='Specify run name')
parser.add_argument('-c', action='store_true', help='Continue run')
args = parser.parse_args()
inputdir = paths.inputdir + args.runname + '/'
if not fw.check_indir(inputdir):
    pool.close()
    sys.exit()

# Initial setup of directories and file paths
# Note that if you want to make a subdirectory, you have to take
# this into account when going back to the main directory after
# fastwind has run (in the function execute_fastwind)
outputdir = paths.outputdir
fd = fw.make_file_dict(inputdir, outputdir)
fw.mkdir(outputdir)
outdir, rundir, savedir, indir = fw.init_setup(outputdir)
fw.copy_input(fd,indir)
# The control file is from now on read from the input_copy dir
# So if the user wants to change control files, this has to
# be done there. Changes in the original input dir have no effect.
fd["control_in"] = indir + fd["control_in"].split('/')[-1]

multiplicity = len(fd['paramspaces_in'])

# Read control parameters
cdict = fw.read_control_pars(fd["control_in"])

# Remove (new run) or replace (continued run) old output files.
fw.prepare_output_files(fd, args.c)

# Read input files and data
paramspaces: list[pop.Template] = []
for component in fd["paramspaces_in"]:
    paramspaces.append(pop.Template(component, fd["defvals_in"]))

radinfo = np.genfromtxt(fd["radinfo_in"], comments='#', dtype='str')

dof = sum([len(list(x.variables.keys())) for x in paramspaces])

lineinfo = fw.read_data(fd["linelist_in"], fd["epochs_in"])

''' PREPARE FASTWIND '''

# Create a FORMAL_INPUT file containing the relevant lines.
fw.create_FORMAL_INPUT(cdict['inicalcdir'], lineinfo[0], fd["linelist_in"])

# Initialise the fitness function with parameters that are
# the same for every model.

max_gen_duration = cdict["fw_timeout"]

''' THE GENETIC ALGORITHM STARTS HERE '''
# When starting from scratch, the first generation is calculated
if not args.c:
    if max_gen_duration*2.5 > cdict["program_time_limit"]:
        print("Not enough time to run first generation, exiting")
        pool.close()
        sys.exit()
    gencount = 0

    fail_counter = 0

    run_fastwind = functools.partial(fw.get_fastwind_output, cdict["inicalcdir"], rundir, savedir, cdict["modelatom"],
                                     cdict["fw_timeout"], lineinfo, fail_counter, fd['precomp_out'], cdict['be_verbose'], cdict['location'])

    nind = cdict["nind"]

    init_population = pop.InitPopulation(nind, gencount, paramspaces, radinfo, cdict['be_verbose'], cdict['location'])
    init_population.gen_modnames(retry=fail_counter)
    init_population.init_precomp(fd["precomp_out"])

    flat_gen = init_population.construct_flat_gen()

    # Reorder input for eval_fitness function and assess fitness.
    final_genes = list(pool.map(run_fastwind, flat_gen))

    init_population.update_precompute(final_genes, fd["precomp_out"])

    generation = init_population.get_normal_population(final_genes, radinfo)

    fw.evaluate_fitnesses(generation, lineinfo, savedir, fd["precomp_out"], dof, cdict["fitmeasure"], fd["chi2_out"])

    fitmeasures, red_chi2s = generation.get_fitm_chi2r()
    # The fittest individual is selected
    genbest, best_fitness, best_rchi2 = generation.get_fittest(fitmeasures, red_chi2s)

    generation.store_lowestchi2(fd["bestchi2_out"], red_chi2s)

    mutation_rate = cdict["mut_rate_init"]
    generation.store_mutation(fd["mutation_out"],  mutation_rate)
    generation.store_charbonneaulimits(fd["charblim_out"], cdict)
    shutil.copy(fd["chi2_out"], fd["chi2_cont"])
    shutil.copy(fd["precomp_out"], fd["precomp_cont"])
    generation.save(fd["gen_cont"])
    generation.save(savedir + generation.name + '/' + generation.name + '_snapshot.pkl')
    np.savetxt(fd["fit_cont"], fitmeasures)
    np.savetxt(fd["redchi_cont"], red_chi2s)

    generation.print_report(fitmeasures, best_fitness, cdict["be_verbose"])

# When continuing an old run, simply pick up the gencount, mutation
# rate and the fitmeasures and parameters of the last generation.
else:
    gencount, mutation_rate = fw.read_mut_gen(fd["mutation_out"])
    generation = pop.Population.from_file(fd["gen_cont"])
    fitmeasures = np.genfromtxt(fd["fit_cont"])
    red_chi2s = np.genfromtxt(fd["redchi_cont"])
    genbest, best_fitness, best_rchi2 = generation.get_fittest(fitmeasures, red_chi2s)


while gencount <= cdict["ngen"]:
    gen_start_time = time.time()
    time_elapsed = gen_start_time - program_start_time
    time_left = cdict["program_time_limit"] - time_elapsed
    if time_left < 1.05 * max_gen_duration:
        print("Not confident in remaining time (" + str(time_left/60) + " minutes), stopping Kiwi GA")
        break

    gencount = gencount + 1
    fail_counter = -1 #no retries

    # Read control parameters: the user can change these during the run.
    # !!! The control file is read from *input_copy* directory,
    #     so changing values in the input directory has no effect !
    cdict = fw.read_control_pars(fd["control_in"])

    #Read template files again for changes
    paramspaces: list[pop.Template] = []
    for component in fd["paramspaces_in"]:
        paramspaces.append(pop.Template(component, fd["defvals_in"]))
    generation.refresh_templates(paramspaces)

    if gencount > cdict["ngen"]:
        print("Desired gencount reached, stopping Kiwi GA")
        break

    # Re-initialise the fitness function with parameters that are
    # the same for every model. The control parameters, especially
    # the fw_timeout, might be changed by the user during the run.
    run_fastwind = functools.partial(fw.get_fastwind_output, cdict["inicalcdir"], rundir, savedir, cdict["modelatom"],
                                     cdict["fw_timeout"], lineinfo, fail_counter, fd['precomp_out'], cdict['be_verbose'], cdict['location'])

    # Reproduce and asses fitness
    generation_o = generation.reproduce(mutation_rate,
        cdict["clone_fraction"], fd["precomp_out"],
        cdict["w_gauss_na"], cdict["w_gauss_br"], cdict["b_gauss_na"],
        cdict["b_gauss_br"], cdict["mut_rate_na"], cdict["nind"],
        cdict["narrow_type"], cdict["broad_type"], cdict["doublebroad"], cdict['location'], gencount)

    generation_o.gen_modnames(retry=fail_counter)

    generation_o.extend_precomp(fd["precomp_out"])

    flat_gen = generation_o.construct_flat_gen()

    # Reorder input for eval_fitness function and assess fitness.
    final_genes = list(pool.map(run_fastwind, flat_gen))

    i = 0
    for ind in generation_o.population:
        for j in range(len(ind.components)):
            if final_genes[i] is not None:
                ind.components[j] = final_genes[i]
            i += 1


    generation_o.update_precompute(final_genes, fd["precomp_out"])


    fw.evaluate_fitnesses(generation_o, lineinfo, savedir, fd["precomp_out"], dof, cdict["fitmeasure"], fd["chi2_out"])

    fitmeasures_o, red_chi2s_o = generation_o.get_fitm_chi2r()
    # The parent population (generation, fitmeasures), is created
    # based on the offpsring pop. (generation_o, fitmeasures_o)
    if cdict["ratio_po"] == 1.0 and cdict["f_parent"] == 0.0:
        # Case of pure reinsertion: offspring pop = parent pop.,
        # but the fittest individual of the run always survives
        # (This only has to be done explictly if the pure reinsertion
        # scheme is used, otherwise this is the case automatically.)
        fitmeasures, red_chi2s = generation_o.reincarnate(fitmeasures_o, red_chi2s_o, genbest, best_fitness, best_rchi2)
        generation = generation_o
    else:
        # In the other cases, i.e. when the reinsertion schemes of
        # elitist and fitness-based are combined, the best inidividuals
        # of the parent population and the offspring are combined.
        top_x_offspring_ind, fitmeasures_o, red_chi2s_o = generation_o.get_top_x_fittest(fitmeasures_o, red_chi2s_o, cdict["n_keep_offspring"])
        top_x_parent_ind, fitmeasures, red_chi2s = generation.get_top_x_fittest(fitmeasures, red_chi2s, cdict["n_keep_parent"])
        top_x_ind = np.concatenate((top_x_parent_ind, top_x_offspring_ind))
        fitmeasures  = np.concatenate((fitmeasures, fitmeasures_o))
        red_chi2s = np.concatenate((red_chi2s, red_chi2s_o))
        generation = pop.Population(top_x_ind.tolist(), gencount)

    genbest, best_fitness, best_rchi2 = generation.get_fittest(fitmeasures, red_chi2s)

    generation.store_lowestchi2(fd["bestchi2_out"], red_chi2s)

    # Before adjusting the mutation rate, set the charbonneau limits,
    # if 'autocharb' is chosen. This is done every generation so that you
    # can change the mutation type during the run, if wanted.
    if cdict['mut_adjust_type'] == 'autocharb':
        cdict = generation.autoadjust_charbonneau(cdict, fd, gencount)

    # Depending on the scheme chosen, adjust the mutation rate.
    # If the chosen scheme is 'constant', no adaption is made.
    if cdict["mut_adjust_type"] in ('charbonneau', 'autocharb'):
        mutation_rate = generation.adjust_mutation_rate_charbonneau(mutation_rate,
            fitmeasures, cdict["mut_rate_factor"], cdict["mut_rate_min"],
            cdict["mut_rate_max"], cdict["fit_cutoff_min_charb"],
            cdict["fit_cutoff_min_charb"])

    # Store mutation rate and files for run continuation
    # Copies of the chi2 file and dupl file are certain to only
    # contain the output of a fully completed generation.
    generation.store_mutation(fd["mutation_out"],  mutation_rate)
    generation.store_charbonneaulimits(fd["charblim_out"], cdict)
    shutil.copy(fd["chi2_out"], fd["chi2_cont"])
    shutil.copy(fd["precomp_out"], fd["precomp_cont"])
    generation.save(fd["gen_cont"])
    generation_o.save(savedir + generation.name + '/' + generation.name + '_snapshot.pkl')
    np.savetxt(fd["fit_cont"], fitmeasures)
    np.savetxt(fd["redchi_cont"], red_chi2s)

    generation.print_report(fitmeasures, best_fitness, cdict["be_verbose"])
    gen_end_time = time.time()
    gen_duration = gen_end_time - gen_start_time
    if gen_duration > max_gen_duration:
        max_gen_duration = gen_duration

pool.close()
sys.exit()
