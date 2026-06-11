"""
Routine to apply instumental, rotational, and macroturbulent broadening.
Script from  Michael Abdul Masih, Calum Hawcroft (2020).
Altered by Sarah Brands to account for convolving large wavelength
ranges as might be needed for usage i.c.w. FW v11 (Feb 2021).

Script revised by Frank Backs (2021). Some optimization has been done.
It runs faster now and is marginally more precise. Dependence on PyAstronomy
has been removed. Now all broadening profiles are first convolved with
each other and then with the spectrum.

Usage:
> python broaden.py -f <filename> -r <resolving power> -v <vrot> -m <vmacro>

"""

# Suppress the standard matplotlib warning
import warnings

warnings.filterwarnings("ignore")

# Import needed modules
import sys
import argparse
import numpy as np
from math import ceil
from scipy.special import erf
from scipy.signal import convolve
from scipy.stats import norm


def broaden_fwline__old__(wlc, flux, vrot, resolution, vmacro=0):
    """
    Broadens a model line profile corresponding with the given rotational velocity, spectral resolution and if provided
    macro turbulent velocity.
    :param wlc:         Wavelength points of the data, in Angstrom, array
    :param flux:        Flux of the data points (might need to be normalized), array
    :param vrot:        Rotational velocity of the star (vsini), in km/s, scalar
    :param resolution:  Desired spectral resolution at this wavelength. unitless, scalar
    :param vmacro:      Desired macro turbulent velocity, in km/s, scalar
    :return: wave, flux arrays, with the wavelength points made uniform and linear. Flux after broadening.
    """
    # Settings for rebinning
    binSize = 0.01  # Size of wavelength bins resampled spectrum (Angstrom)
    finalBins = 0.01  # Size of wavelength bins of broadened spectrum (Angstrom)

    limbDark = 0.6  # Limb darkening coefficient (rotational broadening)

    # Settings for large wavelength intervals
    # An interval of 100.0 A gives a precicion of about 0.001 % compared to
    # using the exact appropiate kernel fore ach wavelength
    range_fast = 100.  # Angstrom. Maximum width of range for using fast broadening
    overlap = 20.0  # Extend the range to account for edges


    # Resample input spectrum to even wavelength bins of <binSize> Angstrom
    newWlc = np.arange(wlc[0] + binSize, wlc[-1] - binSize, binSize)
    flux = np.interp(newWlc, wlc, flux)
    wlc = newWlc

    # Check the total width of the wavelength interval
    # If this is smaller than a certain value, apply the fast broadening,
    # i.e. one kernel for all wavelengths
    # If the wavelength interval is large, then cut the interval into
    # pieces, use an appropiate interval for each piece, and stitch together.
    totalwidth = wlc[-1] - wlc[0]
    if totalwidth < range_fast:

        # Apply the broadening.
        flux = broaden_function(wlc, flux, resolution, vrot,
                                vmacro=vmacro, limbdark=limbDark, nsig=4.)

        # Resample to <finalBins> Angstrom
        if finalBins != binSize:
            newWlc = np.arange(wlc[0] + finalBins, wlc[-1] - finalBins, finalBins)
            flux = np.interp(newWlc, wlc, flux)
        convolved_flux_all = flux
        convolved_wave_all = newWlc

    else:
        nparts = int(ceil(totalwidth / range_fast))
        partwidth = totalwidth / nparts
        lenbins = len(wlc[wlc < wlc[0] + partwidth])
        partwidthext = partwidth + overlap
        lenbinsext = len(wlc[wlc < wlc[0] + partwidthext])
        extrabins = lenbinsext - lenbins

        idxlow_list = []
        idxhigh_list = []
        for i in range(nparts):
            idxlow = max(0, i * lenbins - extrabins)
            idxhigh = min(i * lenbins + lenbins + extrabins, len(wlc))

            idxlow_list.append(idxlow)
            idxhigh_list.append(idxhigh)

        convolved_flux_all = np.array([])
        convolved_wave_all = np.array([])

        for startarg, endarg, counter in zip(idxlow_list, idxhigh_list, range(nparts)):
            wlcpart = wlc[startarg:endarg]
            fluxpart = flux[startarg:endarg]

            # Apply the broadening.
            fluxpart = broaden_function(wlcpart, fluxpart, resolution, vrot,
                                    vmacro=vmacro, limbdark=limbDark, nsig=4.)

            # Resample to <finalBins> Angstrom
            if finalBins != binSize:
                newWlcpart = np.arange(wlcpart[0] + finalBins, wlcpart[-1] - finalBins, finalBins)
                fluxpart = np.interp(newWlcpart, wlcpart, fluxpart)
                wlcpart = newWlcpart

            if counter == 0:
                wlcpart = wlcpart[0:-extrabins]
                fluxpart = fluxpart[0:-extrabins]
            elif counter == nparts - 1:
                wlcpart = wlcpart[extrabins:]
                fluxpart = fluxpart[extrabins:]
            else:
                wlcpart = wlcpart[extrabins:-extrabins]
                fluxpart = fluxpart[extrabins:-extrabins]

            convolved_flux_all = np.concatenate((convolved_flux_all, fluxpart))
            convolved_wave_all = np.concatenate((convolved_wave_all, wlcpart))

    return convolved_wave_all, convolved_flux_all


def broaden_fwline(wlc, flux, vrot, resolution, vmacro=0):
    """
    Broadens a model line profile corresponding with the given rotational velocity, spectral resolution and if provided
    macro turbulent velocity. Now rewritten to take into account large wavelength variations, to give reasonable results
    from the UV to the MIR. This is done by scaling the binning/resolution by the resolving power at the given
    wavelength.
    :param wlc:         Wavelength points of the data, in Angstrom, array
    :param flux:        Flux of the data points (might need to be normalized), array
    :param vrot:        Rotational velocity of the star (vsini), in km/s, scalar
    :param resolution:  Desired spectral resolution at this wavelength. unitless, scalar
    :param vmacro:      Desired macro turbulent velocity, in km/s, scalar
    :return: wave, flux arrays, with the wavelength points made uniform and linear. Flux after broadening.
    """
    # Settings for rebinning
    mean_wlc = np.mean(wlc)
    # Currently set to oversampling the resolving power by a factor 10 or at most 0.01 Angstrom
    binSize = max(0.01, 0.1 * mean_wlc / resolution)  # Size of wavelength bins resampled spectrum (Angstrom)
    finalBins = max(0.01, 0.1 * mean_wlc / resolution)  # Size of wavelength bins of broadened spectrum (Angstrom)

    limbDark = 0.6  # Limb darkening coefficient (rotational broadening)

    # Settings for large wavelength intervals
    # Uses 500 times the resolving power or 100 Angstrom depending on what is larger.
    range_fast = max(100, 500 * mean_wlc / resolution)  # Angstrom. Maximum width of range for using fast broadening
    overlap = 0.2 * range_fast # Extend the range to account for edges

    # Resample input spectrum to even wavelength bins of <binSize> Angstrom
    newWlc = np.arange(wlc[0] + binSize, wlc[-1] - binSize, binSize)
    flux = np.interp(newWlc, wlc, flux)
    wlc = newWlc

    # Check the total width of the wavelength interval
    # If this is smaller than a certain value, apply the fast broadening,
    # i.e. one kernel for all wavelengths
    # If the wavelength interval is large, then cut the interval into
    # pieces, use an appropiate interval for each piece, and stitch together.
    totalwidth = wlc[-1] - wlc[0]
    if totalwidth < range_fast:

        # Apply the broadening.
        flux = broaden_function(wlc, flux, resolution, vrot,
                                vmacro=vmacro, limbdark=limbDark, nsig=4.)

        # Resample to <finalBins> Angstrom
        if finalBins != binSize:
            newWlc = np.arange(wlc[0] + finalBins, wlc[-1] - finalBins, finalBins)
            flux = np.interp(newWlc, wlc, flux)
        convolved_flux_all = flux
        convolved_wave_all = newWlc

    else:
        nparts = int(ceil(totalwidth / range_fast))
        partwidth = totalwidth / nparts
        lenbins = len(wlc[wlc < wlc[0] + partwidth])
        partwidthext = partwidth + overlap
        lenbinsext = len(wlc[wlc < wlc[0] + partwidthext])
        extrabins = lenbinsext - lenbins

        idxlow_list = []
        idxhigh_list = []
        for i in range(nparts):
            idxlow = max(0, i * lenbins - extrabins)
            idxhigh = min(i * lenbins + lenbins + extrabins, len(wlc))

            idxlow_list.append(idxlow)
            idxhigh_list.append(idxhigh)

        convolved_flux_all = np.array([])
        convolved_wave_all = np.array([])

        for startarg, endarg, counter in zip(idxlow_list, idxhigh_list, range(nparts)):
            wlcpart = wlc[startarg:endarg]
            fluxpart = flux[startarg:endarg]

            # Apply the broadening.
            fluxpart = broaden_function(wlcpart, fluxpart, resolution, vrot,
                                    vmacro=vmacro, limbdark=limbDark, nsig=4.)

            # Resample to <finalBins> Angstrom
            if finalBins != binSize:
                newWlcpart = np.arange(wlcpart[0] + finalBins, wlcpart[-1] - finalBins, finalBins)
                fluxpart = np.interp(newWlcpart, wlcpart, fluxpart)
                wlcpart = newWlcpart

            if counter == 0:
                wlcpart = wlcpart[0:-extrabins]
                fluxpart = fluxpart[0:-extrabins]
            elif counter == nparts - 1:
                wlcpart = wlcpart[extrabins:]
                fluxpart = fluxpart[extrabins:]
            else:
                wlcpart = wlcpart[extrabins:-extrabins]
                fluxpart = fluxpart[extrabins:-extrabins]

            convolved_flux_all = np.concatenate((convolved_flux_all, fluxpart))
            convolved_wave_all = np.concatenate((convolved_wave_all, wlcpart))

    return convolved_wave_all, convolved_flux_all


def parseArguments():
    """
    Reads in the values from the command line.
    """
    parser = argparse.ArgumentParser(description="Applies all broadening.")
    parser.add_argument("-f", "--filename", type=str, dest="fileName", \
                        help="Filename of the input spectrum.")
    parser.add_argument("-r", "--resolution", type=float, dest="res", \
                        help="Resolving power.")
    parser.add_argument("-v", "--vrot", type=float, dest="vrot", \
                        help="Rotational velocity.")
    parser.add_argument("-m", "--vmacro", type=float, dest="vmacro", \
                        help="Macroturbulent velocity.")
    object = parser.parse_args()
    return object


def broaden_function(wave, flux, resolution, vsini, limbdark=0.6, nsig=4, vmacro=0):
    """
    Rotationally broadens part of a model spectrum. Input wavelength space must be uniform and linear.
    Wavelengths in A, velocities in km/s
    Assumes the spectrum to be normalized when preventing edge effects.
    :param wave:        Array of uniform linear wavelength points in Angstrom
    :param flux:        Array of normalized flux values (same length as wavelength array)
    :param resolution:  Scalar, the spectral resolution
    :param vsini:       Scalar, the projected rotational velocity km/s
    :param limbdark:    Scalar [0 - 1], The linear limb-darkening parameter as described in Gray. Default = 0.6
    :param nsig:        Scalar, number of standard deviations to include in the instrument profile broadening.
                                Lower is faster, higher is more precise. Default = 4
    :param vmacro:      Scalar, if larger than 0 the macro turbulent broadening is included as well. Default = 0 km/s
    :return:            Array of convolved flux values.
    """
    # Calculate the instrumental broading profile.
    instrument_profile = instrumental_broading_profile(wave, resolution, nsig=nsig)
    # Calculate the rotational broadening profile.
    rotational_profile = rotational_broadening_profile(wave, vsini, limbdark)

    # Convolve the two profiles with each other. The order does not matter
    # Note that the convolution profile is broadened by npoints, this is because the initial kernels are as
    # narrow as possible. (Especially the rotational broadening curve)
    npoints = len(rotational_profile) // 2
    instrument_profile_temp = np.concatenate((np.zeros(npoints), instrument_profile, np.zeros(npoints)))
    final_profile = convolve(instrument_profile_temp, rotational_profile, mode="same")

    if vmacro and vmacro != -1:  # This works fine as long as non-negative scalars are being put in for vmacro
        # Calculate the macro turbulent broadening profile
        macro_profile = macro_broading_profile(wave, vmacro)

        # Convolve the macro turbulent profile with the final profile
        npoints = len(macro_profile) // 2
        final_profile_temp = np.concatenate((np.zeros(npoints), final_profile, np.zeros(npoints)))
        final_profile = convolve(final_profile_temp, macro_profile, mode="same")

    # Convolve the spectrum/line with the final profile.
    # Add dummy points to counter edge effects. Dummy points are 1, as the spectrum is assumed to be normalized.
    npoints = int(len(final_profile) // 2)
    temp_flux = np.concatenate((np.ones(npoints), flux, np.ones(npoints)))
    # Do the convolution, remove the dummy points after
    convolved_flux = convolve(temp_flux, final_profile, mode="same")[npoints:-npoints]

    return convolved_flux


def rotational_broadening_profile(wave, vsini, limbdark):
    """
    Calculates the rotational broadening profile according to Gray.
    Source: "The Observation and Analysis of Stellar Photospheres"
    Numerical solution inspired by the PyAstronomy package (rotBroad)
    :param wave:        Array of wavelength uniform (linear) points in Angstrom
    :param vsini:       projected rotational velocity (km/s)
    :param limbdark:    Linear limb-darkening parameter
    :return:
    """
    w0 = np.mean(wave)
    wdiff = wave[1] - wave[0]  # Wavelength bin size (assumes it is uniform)
    dwmax = w0 * vsini / 299792.458  # The max delta lambda that can be affected by the doppler effect given vsini.

    # Parameters for the profile
    c1 = 2 * (1 - limbdark) / (np.pi * dwmax * (1 - limbdark / 3.))
    c2 = limbdark / (2 * dwmax * (1 - limbdark / 3.))

    # Calculate the size of the kernel in wavelength bins, make sure to always include 0
    dw = np.arange(0, dwmax, wdiff)
    dw = np.concatenate((-dw[::-1][:-1], dw))

    # normalize wavelength bins based on max width of rotational effects
    x = dw / dwmax

    profile = c1 * (1. - x ** 2) ** 0.5 + c2 * (1. - x ** 2)
    profile /= np.sum(profile)  # normalize the profile
    return profile


def instrumental_broading_profile(wave, resolution, nsig=4.0):
    """
    Calculates the gaussian broading profile based on the instrumental resolution.
    :param wave:        Array of wavelength uniform (linear) points in Angstrom
    :param res:         The desired spectral resolution (scalar)
    :param nsig:        The number of standard deviations to consider for the profile (default = 4.0)
                        Higher number will result in slower convolution but higher precision.
    :return:
    """
    # Calculate the instrumental broadening profile width.
    meanWvl = np.mean(wave)
    fwhm = meanWvl / float(resolution)
    sigma = fwhm / (2.0 * np.sqrt(2. * np.log(2.)))

    # Define the instrument broadening profile.
    # x is the wavelength space of the profile
    # the "loc" in the normal dist is set to the middle point of the distribution.
    # This is to make sure the line does not shift. (trick taken from the PyAstronomy
    # package (instrBroadGaussFast)) It has to do with the
    x = np.arange(-nsig * sigma, nsig * sigma + (wave[1] - wave[0]), wave[1] - wave[0])
    instrument_profile = norm.pdf(x, scale=sigma, loc=x[len(x) // 2 + len(x) % 2 - 1])
    instrument_profile /= np.sum(instrument_profile)
    return instrument_profile


def macro_broading_profile(wave, vmacro):
    """
    Calculates the macro-turbulent broadening profile for a narrow velocity range.
    Inspired by the method described here: http://dx.doi.org/10.5281/zenodo.10013
    :param wave:
    :param vmacro:
    :return:
    """
    # Some constants and such
    c = 299792.458  # Speed of light in km/s
    w0 = np.median(wave)  # Assumed mean wavelength
    wdiff = wave[1] - wave[0]  # Wavelength bin size
    dwmax = vmacro * w0 / c  # Maximum wavelength affected by doppler shifts
    ccr = 2 / (np.pi**0.5 * dwmax)  # A constant

    # Wavelength space of the kernel, relative to the rest wavelength, and in steps of the spectral resolution
    dw = np.arange(0, dwmax, wdiff)
    dw = np.concatenate((-dw[::-1][:-1], dw))  # Trick to always make it symmetric around 0

    # Normalize the wavelength space
    wave_space_norm = abs(dw) / dwmax
    profile = ccr * (np.exp(-wave_space_norm ** 2) + np.pi ** 0.5 * wave_space_norm * (erf(wave_space_norm) - 1.0))
    profile /= np.sum(profile)
    return profile


if __name__ == "__main__":

    # Read in the needed values from the command line
    arguments = parseArguments()

    # Read in the spectrum
    try:
        wlc, flux = np.loadtxt(arguments.fileName, unpack=True)
    except IOError as ioe:
        print(ioe, "Input spectrum " + arguments.fileName + " not found!")
        sys.exit()

    # check vmacro choice
    if arguments.vmacro not in (-1, None):
        vmacro = arguments.vmacro
    else:
        vmacro = 0

    new_wlc, new_flux = broaden_line(wlc, flux, arguments.vrot, arguments.res, vmacro)
    np.savetxt(arguments.fileName + ".fin", np.array([new_wlc, new_flux]).T)
