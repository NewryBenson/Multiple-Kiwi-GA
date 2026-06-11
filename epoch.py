import numpy as np

class Epoch:
    def __init__(self, data: str, lines):
        '''
        :param data: The path to the data file of this epoch
        :param lines: The name, starts and ends of all the lines
        '''
        self.name = data
        self.wave, self.norm, self.error = np.loadtxt(data, unpack=True)
        self.lower = min(self.wave)
        self.upper = max(self.wave)
        self.lines = dict()
        for i in range(len(lines[0])):
            ind = np.argwhere((max(lines[1][i], self.lower) <= self.wave) & (self.wave <= min(lines[2][i], self.upper)))
            self.lines[lines[0][i]] = (self.wave[ind].flatten(), self.norm[ind].flatten(), self.error[ind].flatten())

    def get_line_data(self, line: str):
        '''
        For a certain line, get the data that is in that line contained in this epoch.
        :param line: The name of the desired line
        :return: A tuple containing the wavelenghts, data en errors of that line. If the line and this epoch don't overlap, this will return empty arrays.
        '''
        return self.lines[line]

    def get_line_names(self):
        return list(self.lines.keys())

    def get_residuals(self, vrads, individuals_per_line):
        '''
        calculates the residuals between a model and the data
        :param vrads: The list of radial velocities
        :param individuals_per_line: A list containing the broadened fastwind outputs of all components for all lines
        '''
        c = 299792.458  # km/s
        residuals = []

        lines = list(self.lines.keys())

        for line_idx, models in enumerate(individuals_per_line):
            line = lines[line_idx]
            wave_obs, flux_obs, error_obs = self.lines[line]

            total_flux = np.zeros_like(flux_obs)
            total_cont = np.zeros_like(flux_obs)

            if len(models) != len(vrads):
                raise ValueError("Number of models must match number of radial velocities")

            for k, vrad in enumerate(vrads):
                model = models[k].T

                wave_model, flux_model, cont_model = model

                shifted_wave = wave_model * (1.0 + vrad / c)

                flux_interp = np.interp(wave_obs, shifted_wave, flux_model)
                cont_interp = np.interp(wave_obs, shifted_wave, cont_model)

                total_flux += flux_interp * cont_interp
                total_cont += cont_interp

            valid = total_cont > 0
            combined_flux = np.zeros_like(flux_obs)
            combined_flux[valid] = total_flux[valid] / total_cont[valid]

            line_residuals = (flux_obs[valid] - combined_flux[valid]) / error_obs[valid]
            residuals.append(line_residuals)

        return np.concatenate(residuals)

    def get_active_lines(self):
        '''
        Returns the names of the linewindows that have a nonzero overlap with the data in this epoch
        '''
        result = []
        for name in self.lines.keys():
            if len(self.lines[name][0]) > 0:
                result.append(name)
        return result