import copy
import fcntl
import os
import random
from statistics import median

import numpy as np
import pandas as pd
import magnitude_to_radius as m2r

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Template:
    def __init__(self, template_file: str, fastwind_defaults: str):
        '''
        :param template_file: parameter space file for this template
        :param fastwind_defaults: default parameters for fastwind
        '''
        pspace = np.genfromtxt(template_file, dtype=str)
        defvals = np.genfromtxt(fastwind_defaults, dtype=str)

        self.variables = dict()
        self.fixed = dict()

        for param in pspace:
            if param[1] == param[2]:
                self.fixed[param[0]] = param[1]
            else:
                step = param[3]
                if '.' in step:
                    rounding = len(step.split('.')[-1])
                else:
                    rounding = 0
                self.variables[param[0]] = [param[1], param[2], step, rounding]

        for param in defvals:
            if param[0] not in self.fixed and param[0] not in self.variables:
                self.fixed[param[0]] = param[1]

    def draw_rand_param(self, param: str):
        '''
        Get a rondom value for a parameter from its range
        :param param: The parameter for which to draw the value
        :return: The randomly drawn value for the parameter
        '''
        themin, themax, thestep, rounding = self.variables[param]
        paramarray = np.arange(float(themin), float(themax), float(thestep))
        randparam = random.choice(paramarray)
        return round(randparam, int(rounding))

    def initial_vinf(self, teff, a, b):
        """calculates the initial vinf based on the method in Hawcroft C. 2023"""
        if 'vinf' in self.variables.keys(): #start with estimated value. Deviate from there
            start, stop, step, rounding = self.variables['vinf']
            start, stop, step, rounding = float(start), float(stop), float(step), int(rounding)
            vinf = a * teff - b
            snapped_vinf = round(start + round((vinf - start) / step) * step, rounding)
            if snapped_vinf < start:
                print('!WARNING! calculated vinf below param range')
                snapped_vinf = start
            elif snapped_vinf > stop:
                print('!WARNING! calculated vinf above param range')
                snapped_vinf = stop
            return snapped_vinf
        return float(self.fixed['vinf'])
        #if vinf is set to a value above 0, it is fixed by the user.
        #if vinf is set to negative, it should be deterministic. It gets calculated as part of creat indat



class Component:
    def __init__(self, radius, name, base_name, template, parameters):
        self.vrads = dict()
        self.radius = radius
        self.name = name
        self.base_name = base_name
        self.template = template
        self.parameters = parameters

    def set_name(self, name, retry=-1):
        self.name = name
        self.base_name = name
        if retry >= 0:
            self.name = name + '_' + str(retry)

    def append_precompute(self, precomp):
        data_line = self.name + ',' + self.base_name + ',running,' + self.name

        for param in self.parameters.keys():
            if param != 'ratio':
                data_line += ',' + str(self.parameters[param])

        data_line += ',' + str(round(self.radius, 2)) + '\n'

        with open(precomp, 'a') as the_file:
            fcntl.flock(the_file, fcntl.LOCK_EX)
            the_file.write(data_line)
            fcntl.flock(the_file, fcntl.LOCK_UN)

    @staticmethod
    def double_gauss(x, baseline, height, center, sigma):
        """
        Gaussian with continuum at 0.
        """
        center1 = center - 0.75 * sigma
        center2 = center + 0.75 * sigma

        sigma = 0.4 * sigma

        y = (baseline + height * np.exp(-(x - center1) ** 2 / (2 * sigma ** 2)) +
             height * np.exp(-(x - center2) ** 2 / (2 * sigma ** 2)))

        return y

    @staticmethod
    def gauss(x, baseline, height, center, sigma):
        """
        Gaussian with continuum at 0.
        """
        y = baseline + height * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        return y

    def gaussian_mutation(self, mut_rate, width, base, gtype, double_yn):
        """ Changes (with a certain probability) the value of parameters,
            hereby following a gaussian distribution around the current value
            of the parameter that will mutate.

            Input are the parameters of an individual, then each
            parameter has a chance of mut_rate_na to mutate, with a
            gaussian with a certain width. The width is specified either in
            terms of a fraction of the parameter space width (then determined
            for each parameter), or in terms of steps, so depending on the grid
            of each parameter ('gtype').

            Output is the mutated genome (parameters of the individual).
            """

        # Loop through all parameters of the model
        for param in self.template.variables.keys():
            # A mutation only occurs in a fraction (mut_rate_na) of
            # the genes.
            if random.random() < mut_rate:
                min_p, max_p, step_p, rounding_p = self.template.variables[param]
                min_p, max_p, step_p, rounding_p = float(min_p), float(max_p), float(step_p), int(rounding_p)

                nsteps = int(round((max_p - min_p) / step_p + 1, 0))
                param_space = np.linspace(min_p, max_p, nsteps)
                param_space = param_space[param_space != self.parameters[param]]
                if gtype == 'frac':
                    gauss_width = (max_p - min_p) * width
                else:
                    # If not 'frac', this means: gtype == 'step'
                    gauss_width = step_p * width
                if double_yn == 'yes':
                    props = self.double_gauss(param_space, base, 1.,
                                              self.parameters[param], gauss_width)
                else:
                    props = self.gauss(param_space, base, 1., self.parameters[param],
                                       gauss_width)
                props = props / np.sum(props)

                mutated_gene = np.random.choice(param_space, 1, p=props)[0]

                # The rounding is mainly done for the mass loss, which
                # has a different input format in FW than it has in the
                # parameter space (there it is in 10log)
                mutated_gene = round(mutated_gene, rounding_p)

                self.parameters[param] = mutated_gene

    def reset_defaults(self):
        parameters = list(self.parameters.keys())
        for param in parameters:
            if not param in self.template.variables.keys():
                if param in self.template.fixed.keys():
                    value = self.template.fixed[param]
                    if value == 'T' or value == 'F':
                        self.parameters[param] = value
                    elif '.' in value:
                        self.parameters[param] = float(value)
                    else:
                        self.parameters[param] = int(value)
                else:
                    del self.parameters[param]
        if self.parameters['mdot'] > 0:
            if 'mdot' in self.template.variables.keys():
                self.parameters['mdot'] = round(np.log10(self.parameters['mdot']), int(self.template.variables['mdot'][3]))

class InitComponent(Component):
    def __init__(self, template: Template, loc, inp = None ,verbose = 'N'):
        radius = 0.0
        name = 'placeholder'
        base_name = 'placeholder'
        template = template
        parameters = dict()
        for key, value in template.fixed.items():
            if value == 'T' or value == 'F':
                parameters[key] = value
            elif '.' in value:
                parameters[key] = float(value)
            else:
                parameters[key] = int(value)
        for key, value in template.variables.items():
            parameters[key] = template.draw_rand_param(key)



        if inp is not None:
            parameters['teff'] = inp[0]
            parameters['ratio'] = inp[1]
        
        if ('teff' in template.variables.keys()) and ('logg' in template.variables.keys()):
            while not self.gamma_edd_check(parameters, verbose):
                if inp is None:
                    parameters['teff'] = template.draw_rand_param('teff')
                    parameters['logg'] = template.draw_rand_param('logg')
                else:
                    parameters['logg'] = template.draw_rand_param('logg')


        #Also change these values in get_vinf in fw
        #handle vinf according to Hawcroft C. 2023
        #if vinf is a variable, set its initial value according to the relation
        #if vinf is not a variable, it gets handled in create indat
        if 'vinf' in template.variables.keys():
            if loc == 'SMC':
                a, b = 0.089, 1560  #SMC
            elif loc == 'LMC':
                a, b = 0.088, 1200  #LMC
            else:
                a, b = 0.102, 1300  #GAL
            parameters['vinf'] = template.initial_vinf(parameters['teff'], a, b)

        parameters = dict(sorted(parameters.items()))
        super().__init__(radius, name, base_name, template, parameters)

    @staticmethod
    def gamma_edd_check(parameters, verbose) -> bool:
        """
        Check whether the Eddinton limit is exceeded.
        We do not compute mR (mean atomic mass) in detail, but assume a
        conservative value (giving the lowest Gamma_Edd) of 2.2
        For Helium we use standard a low value, as FW computes GAMMA
        not based on the input HE but based on the starting model HE
        If model computation is allowed, return True, else, return False
        """
        yhe = 0.08

        sigmaB = 5.6704e-5
        speed_light = 2.997925e10
        amh = 1.67352e-24
        sigmae = 6.65e-25 / amh
        ggrav = 10 ** parameters['logg']

        ihe_start = 1.0  # Lowest value for OB stars (O stars = 2, B stars = 1)
        mu = 2.2  # Mean atomic mass. Use a rather high value to be safe

        c2 = (1.0 + ihe_start * yhe) / mu
        sigem = sigmae * c2

        gamma = sigem * (sigmaB / speed_light) * parameters['teff'] ** 4 / ggrav
        if verbose == 'Y' or verbose == 'Yes':
            print('gamma = ', gamma, 'teff =', parameters['teff'], 'logg = ', parameters['logg'], 'yhe =', yhe)

        gamma_cutoff = 1.00
        # Do not compute models that are certainly not meeting that criterion
        if gamma > gamma_cutoff:
            return False

        return True

class Retry(InitComponent):
    """
    Subclass of InitComponent. It is used to regenerate the parameters if the previous ones failed. It conserves the
    same T, logg, radius and base name
    """
    def __init__(self, old: Component, fail_counter, loc, verbose = 'N'):
        super().__init__(old.template, loc, inp=(old.parameters['teff'], old.parameters['ratio']), verbose=verbose)
        self.base_name = old.base_name
        self.name = old.base_name + '_' + str(fail_counter)
        self.radius = old.radius


class Individual:
    def __init__(self, components: list[Component], radinfo):
        self.components = components
        self.multiplicity = len(components)
        self.name = 'placeholder'
        self.radinfo = radinfo
        band, obsmag, zpsyst = radinfo
        obsmag = float(obsmag)
        teffs = []
        ratios = []
        for comp in self.components:
            teffs.append(comp.parameters['teff'])
            ratios.append(comp.parameters['ratio'])
        radii = m2r.magnitude_to_radius(teffs, ratios, band, obsmag, zpsyst)
        for i in range(len(radii)):
            self.components[i].radius = radii[i]

        self.fitting_params = dict()

    def set_name(self, name, retry=-1):
        self.name = name
        for i in range(len(self.components)):
            self.components[i].set_name(self.name + '_' + str(i), retry)

    def clone(self):
        new_components = []
        for comp in self.components:
            new = Component(0.0, 'placeholder', 'placeholder', comp.template, copy.deepcopy(comp.parameters))
            new.reset_defaults()
            new_components.append(new)
        return Individual(new_components, self.radinfo)

    def crossover(self, father, clone_fraction):
        """Generate 2 new indiviuals based on this individual of and a father.
        """

        if random.random() < clone_fraction:
            babygirl = self.clone()
            babyboy = father.clone()
        else:
            babygirl_components = []
            babyboy_components = []
            for comp_mama, comp_papa in zip(self.components, father.components):
                babygirl_component_parameters = copy.deepcopy(comp_mama.parameters)
                babyboy_component_parameters = copy.deepcopy(comp_papa.parameters)
                for param in comp_mama.template.variables.keys():
                    if np.random.choice(2, 1)[0] == 1:
                        babygirl_component_parameters[param], babyboy_component_parameters[param] = (
                            copy.deepcopy(babyboy_component_parameters[param]), copy.deepcopy(babygirl_component_parameters[param]))
                girl = Component(0.0, 'placeholder', 'placeholder', comp_mama.template, babygirl_component_parameters)
                boy = Component(0.0, 'placeholder', 'placeholder', comp_papa.template, babyboy_component_parameters)
                girl.reset_defaults()
                boy.reset_defaults()
                babygirl_components.append(girl)
                babyboy_components.append(boy)
            babygirl = Individual(babygirl_components, self.radinfo)
            babyboy = Individual(babyboy_components, self.radinfo)

        return babygirl, babyboy

    def gaussian_mutation(self, mut_rate, width, base, gtype, double_yn):
        """ Changes (with a certain probability) the value of parameters,
            hereby following a gaussian distribution around the current value
            of the parameter that will mutate.

            Input are the parameters of an individual, then each
            parameter has a chance of mut_rate_na to mutate, with a
            gaussian with a certain width. The width is specified either in
            terms of a fraction of the parameter space width (then determined
            for each parameter), or in terms of steps, so depending on the grid
            of each parameter ('gtype').

            Output is the mutated genome (parameters of the individual).
            """

        # Loop through all components and all parameters of the model
        for comp in self.components:
            comp.gaussian_mutation(mut_rate, width, base, gtype, double_yn)

    def gamma_edd_check(self, loc):
        valid = True
        for comp in self.components:
            valid = valid and InitComponent.gamma_edd_check(comp.parameters, 'N')
        return valid

    def prev_fail_check(self, precompfile):
        precompdata = pd.read_csv(precompfile)
        precompcolumns = precompdata.columns

        for component in self.components:
            params = []
            for param in precompcolumns[4:-1]:
                params.append(component.parameters[param])

            params.append(round(component.radius, 2))
            params = list(map(str, params))

            mask = (precompdata.iloc[:, 4:] == params).all(axis=1)

            if mask.any():
                # previous encounter, check status
                name_first_occurence = precompdata.iloc[mask.idxmax()].loc['name']
                status = precompdata.loc[precompdata['name'] == name_first_occurence, 'status'].iloc[0]
                if status == 'fail':
                    return False
        return True

    def set_fitting_params(self, fitinfo):
        fitmeasure, fitness, chi2_tot, rchi2_tot, dof_tot, linenames, linefitns, vrads = fitinfo
        self.fitting_params['fitmeasure'] = fitmeasure
        self.fitting_params['fitness'] = fitness
        self.fitting_params['chi2'] = chi2_tot
        self.fitting_params['rchi2'] = rchi2_tot
        self.fitting_params['dof'] = dof_tot
        for per_epoch_vrad_info in vrads:
            epoch_name = per_epoch_vrad_info[0]
            for comp_idx in range(len(self.components)):
                self.components[comp_idx].vrads[epoch_name] = per_epoch_vrad_info[comp_idx+1]
        for line_idx in range(len(linenames)):
            self.fitting_params[linenames[line_idx]] = linefitns[line_idx]

    def assure_radius(self):
        band, obsmag, zpsyst = self.radinfo
        obsmag = float(obsmag)
        teffs = []
        ratios = []
        for comp in self.components:
            teffs.append(comp.parameters['teff'])
            ratios.append(comp.parameters['ratio'])
        radii = m2r.magnitude_to_radius(teffs, ratios, band, obsmag, zpsyst)
        for i in range(len(radii)):
            self.components[i].radius = radii[i]


class InitIndividual(Individual):
    def __init__(self, paramspaces: list[Template], radinfo, verbose, loc):
        components: list[Component] = []
        for template in paramspaces:
            components.append(InitComponent(template, loc, verbose = verbose))
        super().__init__(components, radinfo)

class Population:
    _SAVE_VERSION = 1  # bump if class layout changes
    def __init__(self, population: list[Individual], gencount: int):
        self.population:list[Individual] = population
        self.name = str(gencount).zfill(4)
        self.multiplicity = population[0].multiplicity

    def save(self, filepath: str) -> None:
        """
        Persist the full Population state to disk.
        """
        payload = {
            "version": self._SAVE_VERSION,
            "population": self,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, filepath: str):
        """
        Alternative initializer that restores a Population from disk.
        """
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        version = payload.get("version")
        if version != cls._SAVE_VERSION:
            raise RuntimeError(
                f"Incompatible save version: {version} != {cls._SAVE_VERSION}"
            )

        pop = payload["population"]
        if not isinstance(pop, cls):
            raise TypeError("Save file does not contain a Population object")

        return pop

    def refresh_templates(self, paramspaces: list[Template]):
        for ind in self.population:
            for comp_idx in range(len(ind.components)):
                ind.components[comp_idx].template = paramspaces[comp_idx]

    def gen_modnames(self, retry=-1):
        """Generate model names of the format xxxx_xxxx_x, e.g.
        for generation 23 and individual 147 the 2nd components this is 0023_0147_2.
        """
        for i in range(len(self.population)):
            self.population[i].set_name(self.name + '_' + str(i).zfill(4), retry=retry)

    def construct_flat_gen(self):
        flat_gen = []
        for ind in self.population:
            flat_gen += ind.components
        return flat_gen

    def get_fitm_chi2r(self):
        fitmeasures = []
        chi2 = []
        for ind in self.population:
            fitmeasures.append(ind.fitting_params['fitmeasure'])
            chi2.append(ind.fitting_params['rchi2'])
        return fitmeasures, chi2

    def get_fittest(self, fitmeasures, rchi2s):
        """Find the fittest individual in the population."""
        # Rank the individuals according to their fitness
        order = np.argsort(fitmeasures)
        rank = np.argsort(order)
        least_fit_idx = np.argmin(rank)

        # Select fittest model
        best_ind = self.population[least_fit_idx]
        best_chi = fitmeasures[least_fit_idx]
        best_rchi2 = rchi2s[least_fit_idx]

        return best_ind, best_chi, best_rchi2

    def store_lowestchi2(self, txtfile, chi2):
        """ Write the paramters and fitness of each individual in the
        population into a textfile.
        """
        lowestchi2 = np.nanmin(chi2)
        write_lines = []

        if not os.path.isfile(txtfile):
            headerstring = '#Generation Lowest_Chi2 \n'
            write_lines.append(headerstring)

        chi2line = self.name + ' ' + str(lowestchi2) + '\n'
        write_lines.append(chi2line)

        with open(txtfile, 'a') as the_file:
            for aline in write_lines:
                the_file.write(aline)

    def print_report(self, fitmeasures, best_chi, verbose):
        if verbose == 'Y' or verbose == 'Yes' or verbose == 'M' or verbose == 'Minimal':
            print('================================================')
            print('Generation       ' + str(self.name))
            print('Best fitness     ' + str(best_chi))
            print('Median fitness   ' + str(median(fitmeasures)))

    def store_mutation(self, txtfile, mutrate):
        """ Write the mutation rate of the current generation into a
        textfile.
        """
        write_lines = []
        if not os.path.isfile(txtfile):
            headerstring = '#Generation Mutation_rate \n'
            write_lines.append(headerstring)

        mutline = self.name + ' ' + str(mutrate) + '\n'
        write_lines.append(mutline)

        with open(txtfile, 'a') as the_file:
            for aline in write_lines:
                the_file.write(aline)

    def store_charbonneaulimits(self, txtfile, thedct):
        """ Write the charbonneau limits used for adapting the mutation
        rate of the current generation into a textfile.
        """
        write_lines = []

        if not os.path.isfile(txtfile):
            headerstring = '#Generation charblim_min charblim_max \n'
            write_lines.append(headerstring)

        charbmin = thedct["fit_cutoff_min_charb"]
        charbmax = thedct["fit_cutoff_max_charb"]

        carline = self.name + ' ' + str(charbmin) + ' ' + str(charbmax) + '\n'
        write_lines.append(carline)

        with open(txtfile, 'a') as the_file:
            for aline in write_lines:
                the_file.write(aline)
                
    def reproduce(self, mut_rate, clone_fraction, precompfile, gauss_w_na, gauss_w_br,
        gauss_b_na, gauss_b_br, mut_rate_na, n_ind, na_type, br_type, dgauss, loc, gencount):
        """Given a population of individuals and a measure for their
        fitness, generate a new generation of individuals.

        IMPORTANT: models are ranked based on their so-called fitness
        measrue <fitm>. This value can be the total chi2 or some other
        measure, but:
           >> make sure that a lower value corresponds to a fitter model!
        With the current implementation (models are ranked and then
        weighted according to their ranking, but not weighted directly
        by their fitness measure) the absolute difference between the
        values of the fitness measure does not matter, but in an
        approach that uses the fitness directly for weight, it will.
        """

        fitm = []
        for ind in self.population:
            fitm.append(ind.fitting_params['fitmeasure'])

        flat_pop_orig = self.population

        # Rank the individuals according to their fitness
        order = np.argsort(fitm)
        rank = np.argsort(order)

        # Assign a reproduction probability to each indiv. using their rank
        pop_len = len(flat_pop_orig)
        repro_prop = pop_len - rank
        repro_prop = 1.0 * repro_prop / np.sum(repro_prop)

        pop_new = []
        while len(pop_new) < n_ind:
            # Pick two random parents and look up their genes
            mother_idx = np.random.choice(pop_len, 1, p=repro_prop)[0]
            father_idx = np.random.choice(pop_len, 1, p=repro_prop)[0]
            mother: Individual = flat_pop_orig[mother_idx]
            father: Individual = flat_pop_orig[father_idx]

            # Parent genomes produce two baby genomes
            baby1, baby2 = mother.crossover(father, clone_fraction)

            # Mutate the baby genomes. There are two modes of mutation.
            # Load values defining the distributions for the two types.
            gauss_w_na = float(gauss_w_na)
            gauss_w_br = float(gauss_w_br)
            gauss_b_na = float(gauss_b_na)
            gauss_b_br = float(gauss_b_br)
            mut_rate_na = float(mut_rate_na)

            # Narrow mutation: close to original value, high mutation
            # rate that is in principle fixed
            baby1.gaussian_mutation(mut_rate_na, gauss_w_na, gauss_b_na, na_type, double_yn='no')
            baby2.gaussian_mutation(mut_rate_na, gauss_w_na, gauss_b_na, na_type, double_yn='no')

            # Broad mutation: further away from original value, lower
            # mutation rate that is variable
            baby1.gaussian_mutation(mut_rate, gauss_w_br, gauss_b_br, br_type, double_yn=dgauss)
            baby2.gaussian_mutation(mut_rate, gauss_w_br, gauss_b_br, br_type, double_yn=dgauss)

            baby1.assure_radius()
            baby2.assure_radius()

            if baby1.gamma_edd_check(loc) and baby1.prev_fail_check(precompfile):
                pop_new.append(baby1)
                if len(pop_new) == n_ind:
                    break
            if baby2.gamma_edd_check(loc) and baby2.prev_fail_check(precompfile):
                pop_new.append(baby2)

        return Population(pop_new, gencount)

    def extend_precomp(self, precompfile: str):
        precompdata = pd.read_csv(precompfile)
        precompcolumns = precompdata.columns

        for individual in self.population:
            for component in individual.components:
                base_name = component.base_name

                params = []
                for param in precompcolumns[4:-1]:
                    params.append(component.parameters[param])

                params.append(round(component.radius, 2))
                params = list(map(str, params))

                mask = (precompdata.iloc[:, 4:] == params).all(axis=1)

                if mask.any():
                    #previous encounter, put status and result owner to the same as copy
                    name_first_occurence = precompdata.iloc[mask.idxmax()].loc['name']
                    status = precompdata.loc[precompdata['name'] == name_first_occurence, 'status'].iloc[0]
                    if status == 'fail' or status == 'success':
                        precompdata.loc[len(precompdata)] = [base_name, # name, no runs required so guaranteed to be its own base case
                                                             base_name,  # base case
                                                             status,  # status
                                                             precompdata.loc[precompdata['name'] == name_first_occurence, 'result_owner'].iloc[0]  # result owner
                                                             ] + params  # other parameters, should be equal to params of the ref as per check above
                    else:
                        precompdata.loc[len(precompdata)] = [base_name, # name, no runs required so guaranteed to be its own base case
                                                             base_name,  # base case
                                                             'waiting_for_ref',  # status
                                                             precompdata.loc[precompdata['name'] == name_first_occurence, 'result_owner'].iloc[0]  # result owner
                                                             ] + params  # other parameters, should be equal to params of the ref as per check above
                else:
                    #no previous encounter, append as running
                    precompdata.loc[len(precompdata)] = [base_name, base_name, 'running', base_name] + params

            precompdata.to_csv(precompfile, index=False)

    def reincarnate(self, fitm_pop, rchi2_pop, previous_best, fitm_prevbest, rchi2_prevbest):
        """ Replace worst fitting individual from generation with the best
            fitting individual of the previous generation.

            (This should happen before the reproduction takes place.)

            Principle: the overall fittest individual of the run should always
            be present in each generation before it reproduces.
            """

        # The population is only altered when the current population does
        # not contain a fitter individual than the previous one.
        if fitm_prevbest < min(fitm_pop):
            # Rank the individuals according to their fitness
            order = np.argsort(fitm_pop)
            rank = np.argsort(order)

            # Select least fit model, to be kicked out of population
            least_fit_idx = np.argmax(rank)

            # Replace least fit model by fittest model of prev. generation
            self.population[least_fit_idx] = previous_best
            fitm_pop[least_fit_idx] = fitm_prevbest
            rchi2_pop[least_fit_idx] = rchi2_prevbest

        return fitm_pop, rchi2_pop

    def get_top_x_fittest(self, fitm_pop, rchi2_pop, topx):
        """Of a population, return the topx fittest individuals.
        Used if one works with a larger first generation compared to
        the rest of the generations.
        """

        # Rank the individuals according to their fitness
        order = np.argsort(fitm_pop)
        rank = np.argsort(order)
        fitm_pop = np.array(fitm_pop)
        rchi2_pop = np.array(rchi2_pop)
        population = np.array(self.population)

        # Select fittest topx models
        best_fitm = fitm_pop[rank < topx]
        best_rchi2 = rchi2_pop[rank < topx]
        best_population = population[rank < topx]

        return best_population, best_fitm, best_rchi2

    @staticmethod
    def charbonneau_ratio(the_fitn):
        """ Measure for the fitness spread of the population"""
        best_mod = np.min(the_fitn)
        median_mod = np.median(the_fitn)

        charbratio = np.abs(best_mod - median_mod) / (best_mod + median_mod)

        return best_mod, median_mod, charbratio

    def adjust_mutation_rate_charbonneau(self, old_rate, chi2, mut_rate_factor,
                                         mut_rate_min, mut_rate_max, fit_cutoff_min, fit_cutoff_max):
        """Adjust the mutation rate based on the typical fitness
        in a population of individuals, as is suggested in
        Charbonneau (1995)"""

        # A large ratio means that the difference between the 'typical
        # model' in a generation, and the fittest model is large. In this
        # case the mutation rate is decreased, so that the genome of the
        # fitter models is better preserved, so that the fit can improve.
        # If the ratio is low, a (local) minimum has apparently been
        # explored well, and no improvements are there to be reached. The
        # mutation rate is then increased so that less well explored parts
        # of the parameter space will be probed.

        cbest, cmod, ratio = self.charbonneau_ratio(chi2)

        if ratio <= fit_cutoff_min:
            new_rate = min(mut_rate_max, old_rate * mut_rate_factor)
        elif ratio >= fit_cutoff_max:
            new_rate = max(mut_rate_min, old_rate / mut_rate_factor)
        else:
            new_rate = old_rate

        return new_rate

    @staticmethod
    def auto_charbonneau_limits(dct, charbini):
        ''' Given the initial Charbonneau ratio of the run, compute
            reasonable values for the final Charbonnau ratio '''

        ac_fit_a = float(dct['ac_fit_a'])
        ac_fit_b = float(dct['ac_fit_b'])
        ac_lowerlim = float(dct['ac_lowerlim'])
        ac_upperlim = float(dct['ac_upperlim'])
        ac_max_factor = float(dct['ac_max_factor'])

        charblim_min = ac_fit_a + charbini * ac_fit_b

        if charblim_min < ac_lowerlim:
            charblim_min = ac_lowerlim
        elif charblim_min > ac_upperlim:
            charblim_min = ac_upperlim

        charblim_max = ac_max_factor * charblim_min

        charblim_min = round(charblim_min, 3)
        charblim_max = round(charblim_max, 3)

        dct['fit_cutoff_min_charb'] = charblim_min
        dct['fit_cutoff_max_charb'] = charblim_max

        return dct

    def autoadjust_charbonneau(self, dct_ctrl, dct_files, gencount):
        ''' This function adjusts the charbonneau limits (so overrides
        the ones given in the control file) after the first generation
        by using the charbonneau ratio of the first generation as an
        indication for reasonable limits. This is because while ideally
        the ratio reflects only the convergence of the run, in practice
        it also depends on the spectrum that you fit. This has to be
        taken into account when using the limits.
        If after ac_maxgen generations the charbonneau lower limit is
        still not reached, then automatically increase mutation rate.
        '''
        with open(dct_files["genvar_out"]) as f:
            content = f.readlines()

        charbini_run = float(content[1].strip().split()[2])

        dct_ctrl = self.auto_charbonneau_limits(dct_ctrl, charbini_run)

        if gencount > float(dct_ctrl['ac_maxgen']):
            charb_list = np.genfromtxt(dct_files["genvar_out"]).T[2]
            min_charb_run = np.min(charb_list)
            last_charb_run = charb_list[-1]

            if min_charb_run > float(dct_ctrl['fit_cutoff_min_charb']):
                dct_ctrl['fit_cutoff_min_charb'] = round(last_charb_run +
                                                         float(dct_ctrl['ac_maxgen_min']), 3)
                dct_ctrl['fit_cutoff_max_charb'] = round(last_charb_run +
                                                         float(dct_ctrl['ac_maxgen_max']), 3)

        return dct_ctrl

    @staticmethod
    def update_precompute(final_genes: list[Component], precompfile):
        precompdata = pd.read_csv(precompfile)

        # all rows in final names succeeded, so their status is success
        for comp in final_genes:
            if comp is not None:
                name = comp.name
                precompdata.loc[precompdata['name'] == name, 'status'] = 'success'

        # all rows that are still marked as running failed
        precompdata.loc[precompdata['status'] == 'running', 'status'] = 'fail'

        # all rows that were waiting for ref can now assume the status of their ref
        precompdata.loc[precompdata['status'] == 'waiting_for_ref', 'status'] = (
            precompdata.loc[precompdata['status'] == 'waiting_for_ref', 'result_owner'].map(
                precompdata.set_index('name')['status']))

        precompdata.to_csv(precompfile, index=False)

    def get_individual_with_name(self, name):
        for ind in self.population:
            if ind.name == name:
                return ind
        return None

class InitPopulation(Population):
    def __init__(self, nind, gencount, paramspaces: list[Template], radinfo, verbose, loc):
        population:list[Individual] = []
        for individual in range(nind):
            population.append(InitIndividual(paramspaces, radinfo, verbose, loc))
        super().__init__(population, gencount)

    def init_precomp(self, precompfile: str):
        precompcolumns = ['name', 'base_name', 'status', 'result_owner']

        for param in self.population[0].components[0].parameters:
            if param != 'ratio':
                precompcolumns.append(param)

        precompcolumns.append('radius')

        precompdata = pd.DataFrame(columns=precompcolumns)

        for individual in self.population:
            for component in individual.components:
                base_name = component.base_name
                full_name = component.name

                params = []
                for param in precompcolumns[4:-1]:
                    params.append(component.parameters[param])

                params.append(round(component.radius, 2))
                params = list(map(str, params))

                precompdata.loc[len(precompdata)] = [full_name, base_name, 'running', full_name] + params
                precompdata.loc[len(precompdata)] = [base_name, base_name, 'base_case', full_name] + params

        precompdata.to_csv(precompfile, index=False)

    def get_normal_population(self, final_genes, radinfo):
        population = []
        for ind in range(len(self.population)):
            components = []
            for comp in range(self.multiplicity):
                i = ind*self.multiplicity + comp
                if final_genes[i] != None:
                    #put the succesfull component in the normal population
                    components.append(final_genes[i])
                else:
                    #all failed, put original failed component in the normal population
                    components.append(self.population[ind].components[comp])
            population.append(Individual(components, radinfo))
        normal_pop = Population(population, 0)
        #overwrite the names with their final names, no retries. We now have a normal population made from components of the init pop or retries.
        normal_pop.gen_modnames()
        return normal_pop

    @staticmethod
    def update_precompute(final_genes: list[Component], precompfile):
        precompdata = pd.read_csv(precompfile)

        parameter_cols = [col for col in precompdata.columns if
                          col not in ['name', 'base_name', 'status', 'result_owner']]

        # all rows in final names succeeded, so their status is success
        for comp in final_genes:
            if comp is not None:
                name = comp.name
                precompdata.loc[precompdata['name'] == name, 'status'] = 'success'
                # Update the base_name row with this row's information
                base_name = precompdata.loc[precompdata['name'] == name, 'base_name'].iloc[0]
                if base_name != name:
                    precompdata.loc[precompdata['name'] == base_name, 'status'] = 'success'
                    precompdata.loc[precompdata['name'] == base_name, 'result_owner'] = name
                    precompdata.loc[precompdata['name'] == base_name, parameter_cols] = \
                    precompdata.loc[precompdata['name'] == name, parameter_cols].iloc[0].values

        # all rows that are still marked as running failed, including their base cases
        precompdata.loc[precompdata['status'] == 'running', 'status'] = 'fail'
        precompdata.loc[precompdata['status'] == 'base_case', 'status'] = 'fail'

        # all rows that were waiting for ref can now assume the status of their ref
        precompdata.loc[precompdata['status'] == 'waiting_for_ref', 'status'] = (
            precompdata.loc[precompdata['status'] == 'waiting_for_ref', 'result_owner'].map(
                precompdata.set_index('name')['status']))

        precompdata.to_csv(precompfile, index=False)