#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electronic Density of States and Bands Plotter v1.3
Copyright (C) 2022 - Andrés Megías Toledano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import ast
import copy
import glob
import pathlib
import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_wavevector(K1, K2, k1, k2, ki):
    """
    Get te wavecector corresponding to a wavenumber.
    
    Converts a numeric value of a crystalline wavenumber corresponding to just
    one path in the reciprocal space to a crystalline wavevector, if the points
    which delimite that path are known.
    
    Arguments
    ---------
    ki : float
        Crystalline wavenumber to be converted.
    k1, k2 : float
        Crystalline wavenumbers that delimite the path of the reciprocal space
        where ki is located.
    K1, K2 : array (float)
        Crystalline wavevectors associated to the points k1,k2.
        
    Returns
    -------
    Ki : array (float)
        Crystalline wavevector corresponding to ki.
    """
    K1, K2 = np.array(K1), np.array(K2)
    Ki = K1 + (K2-K1)*(ki-k1)/(k2-k1)
    
    return Ki

def find_wavevector(ki, locs, coords):
    """
    Find the wavevector corresponding to a wavenumber.
    
    Converts the numeric value of a crystalline wavenumber corresponding to
    several paths of the reciprocal space to a crystalline wavevector. It first
    finds the points which delimite the path where this wavenumber is located
    and then applies the function get_wavevector.
    
    Arguments
    ---------
    ki : float
        Crystalline wavenumber to be converted.
    locs : array (float)
        Crystalline wavenumbers of the points which delimite the paths in the
        reciprocal space.
    coords : array (float)
        Wavevectors of the points which delimite the paths in the reciprocal
        space.
        
    Returns
    -------
    Ki : array (float)
        Crystalline wavevector corresponding to ki.
    """
    Ki = []
    for i in range(len(coords) - 1):
        if ki >= locs[i] and ki <= locs[i+1]:
            Ki = get_wavevector(K1=coords[i], K2=coords[i+1],
                                k1=locs[i], k2=locs[i+1], ki=ki)
            break  
        
    return Ki

def vector_to_text(v, delimiters='()', round_digits=4,
                   zeros=True, commas=True, blank=False):
    """
    Transform a vector to text, using the specified formatting options.
    
    Arguments
    ---------
    v : array
        Numeric vector to be transformed to text.
    delimiters: str
        Text with the two characters which will be used as delimiters of the
        vector.
    round_digits : int
        Number of maximum decimals for each vector element.
    zeros : bool
        It determines if the integers are expressed like that; for example,
        '0' instead of '0.0', '2' instead of '2.0'.
    commas : bool
        It determines if commas are used to separate the vector elements.
    blank : bool
        It determines if a blank space is used between the vector elements and
        the delimiters.
        
    Returns
    -------
    s : str
        Text of vector v, with the specified formatting options.
    """
    if len(delimiters) != 2:
        raise ValueError('The delimiter must have only 2 characters.')
    del1, del2 = delimiters[0], delimiters[1]
    if blank:
        del1 = del1 + ' '
        del2 = ' ' + del2
    s = str(tuple([round(vi, round_digits) for vi in v]))
    s = s.replace('(', del1).replace(')', del2).replace(del1 + del2, '')
    if zeros:
        s = s.replace('.0,', ',').replace('.0' + del2, del2)
    if not commas or len(v) == 1:
        s = s.replace(',', '')
        
    return s

def click(event):
    """
    Allow to obtain the coordinates of a point in a plot when clicking on it.
    
    Arguments
    ---------
    event : event
    Python event corresponding to clicking on the plot.
    """
    if isinstance(event.artist, Line2D):
        selec = event.artist
        x = selec.get_xdata()
        y = selec.get_ydata()
        ind = event.ind
        x0 = float(np.take(x, ind))
        y0 = float(np.take(y, ind))
        K = find_wavevector(x0, k_locs, k_coords)
        print('Selection:', str(y0), energy_units, ('{' + str(round(x0, 4))
              + '}').replace('.0}', '}'), vector_to_text(K))

def full_path(text):
    """
    Obtain the full path described in a text string.
    
    It corrects the format of the path, allowing the use of the abbreviations
    proper of the operating system (for example, './', '../' and '~' for Unix).
    
    Argument
    --------
    text : str
        Text of the path.
        
    Resultado
    ---------
    path : str
        Text of the full path, so that Python can use it.
    """
    path = copy.copy(text)
    if path.startswith('~'):
        path = os.path.expanduser('~') + path[1:]  
    path = str(pathlib.Path(path).resolve())
    
    return path

def replace_limits(original_limits, variable):
    """
    Change the values of the text limits by numbers.
    
    It replaces the limits of the variable which are not specified ('') and
    those which references the maximum ('max') and the minimum ('min') value.
    
    Arguments
    ---------
    original_limits : list
        List with the inferior and superior limits of the corresponding
        variable.
    variable : array
        Vector with the values of the corresponding variable.
    
    Resultado
    ---------
    new_limits : list
        List with the inferior and superior limits of the corresponding
        variable, as numeric values.
    """
    new_limits = copy.copy(original_limits)
    
    for i,func_i in zip([0,1], [min,max]):
        limit = str(original_limits[i])
        if limit == '':
            new_limits[i] = func_i(variable)
        else:
            for text,func in zip(['min','max'], [min,max]):
                if text in limit:
                    new_limits[i] = func(variable)
                    if len(limit) > len(text):
                        new_limits[i] += float(limit[len(text):])
            new_limits[i] = float(original_limits[i])
            
    return new_limits

def add_margin_to_limits(limits, original_limits, width=1/25):
    """
    Add a margin to the specified limits when appropiate.
    
    If the original limits were not specified (''), it adds a margin to the
    current limits.
    
    Arguments
    ---------
    limits : list
        Current limits, both as numeric values.
    original_limits : list
        Original limits, as numeric or text values.
    width : float
        Size of the margin relative to the range defined by the limits.
    
    Returns
    -------
    new_limits : list
        New limits with the margin.
    """
    new_limits = copy.copy(limits)
    
    margin = (limits[1] - limits[0]) * width
    if original_limits[0] == '':
        new_limits[0] -= margin
    if original_limits[1] == '':
        new_limits[1] += margin
        
    return new_limits
    
def median_absolute_deviation(x, c=1.4826):
    """
    Return the median absolute deviation of the input values.
    
    Arguments
    ---------
    x : array (float)
        Input vector.
    c : float
        Normalization factor; by default it has a value so that the median
        absolute deviation of a normal distribution matches its standard
        deviation.
    
    Returns
    -------
    mad : float
        Median absolute deviation.
    """
    mad = c * np.median(abs(x - np.median(x)))
    
    return mad

def percentile_diagnostic_plots(indices, percentiles, derivative, limit_index,
                                reference, condition_limit):
    """
    Show two diagnostic plots.
    
    It shows two diagnostic plots of the algorithm of automatic search of the
    superior limit of the density of states.
    
    Arguments
    ---------
    indices : array (float)
        Percentile indices.
    percentiles : array (float)
        Percentile function as a function of the index.
    derivative : array (float)
        Derivative of the percentile function as a function of the index.
    limit_index : float
        Calculated percentile index.
    reference : float
        Reference value for calculating the limit on the derivative.
    condition_limit : float
        Number to multiply the reference value to obtain the limit of the
        derivative.
    """
    plt.figure(3, figsize=(10,8))
    
    sp1 = plt.subplot(2,1,1)
    plt.plot(indices, percentiles, color='tab:green')
    ylim1 = plt.ylim()[1]
    plt.vlines(limit_index, 0, ylim1, linestyle='--', alpha=0.6)
    plt.ylim([0, ylim1])
    plt.margins(x=0)
    plt.xlabel('percentile index')
    plt.ylabel('percentile', labelpad=6)
    
    plt.subplot(2,1,2, sharex=sp1)
    plt.plot(indices[1:], derivative, color='tab:green')
    ylim1 = plt.ylim()[1]
    plt.vlines(limit_index, 0, ylim1, linestyle='--', alpha=0.6)
    plt.hlines(reference, indices[0], indices[-1], linestyle='-.', alpha=0.6)
    plt.hlines(reference*condition_limit, indices[0], indices[-1],
               linestyle='--', alpha=0.6)
    plt.ylim([0, ylim1])
    plt.margins(x=0)
    plt.xlabel('percentile index')
    plt.ylabel('derivative of the percentile', labelpad=8)
    
    plt.tight_layout()

#%% Default variable values %%

config = configparser.ConfigParser()

config['PREAMBLE'] = {
    'bands mode': "no",
    'density mode': "yes"
}
config['COMMON OPTIONS'] = {
    'save image': "yes",
    'image format': "'jpg'",
    'font': "'DejaVu Sans'",
    'font size': "14",
    'plot scale': "1",
    'frame width': "1.2",
    'figure dimensions': "[7,7]",
    'zero energy': "0",
    'energy limits': "['','']",
    'energy label': "'energy (eV)'",
    'energy lines': "[]",
    'energy range priority': "'union'",
    'energy units': "'eV'"
}
config['BANDS MODE'] = {
    'folder': "'./'",
    'file': "'data.txt'",
    'number of empty bands': "''",
    'wavevector points': "()",
    'wavenumber offset': "0",
    'plot title': "'Energy bands'",
    'wavevector label': "'wavevector'",
    'plot colors': "(0.5,0.5,0.5), (0.4,0.4,0.8), (0.8,0.4,0.4)",
}
config['DENSITY OF STATES MODE'] = {
    'spin-orbit': "no",
    'total density of states': "no",
    'export densities of states': "no",
    'zoom in energy': "no",
    'folder': "'./'",
    'files prefix': "'pDOS.dat'",
    'projections': "()",
    'density of states limits': "['','']",
    'energy zoom parameters': "{'center': 0, 'amplitude': 3}",
    'plot title': "'Partial densities of states'",
    'density of states label': "'density of states (/eV)'",
    'density of states units': "'/eV'",
    'plot colors': ("'chocolate', 'darkgreen', 'deepskyblue', 'tab:pink', "
                    + "'tab:brown', 'blueviolet', 'orangered', 'tab:olive', "
                    + "'cadetblue'")
}

# Reading of the configuration file.
if len(sys.argv) == 1:
    config_path = full_path('./config.txt')  # default path
else:
    config_path = full_path(sys.argv[1])
config_folder = '/'.join(config_path.split('/')[:-1]) + '/'
os.chdir(config_folder)
if os.path.isfile(config_path):
    config.read(config_path)
else:
    raise FileNotFoundError('Configuration file not found.')

# Reading of the arguments.
preamble = config['PREAMBLE']
bands_mode = preamble.getboolean('bands mode')
dos_mode = preamble.getboolean('density of states mode')
if not (bands_mode or dos_mode):
    raise ValueError('No mode is selected.')
config_common = config['COMMON OPTIONS']
save_image = config_common.getboolean('save image')
font = ast.literal_eval(config_common['font'])
font_size = ast.literal_eval(config_common['font size'])
scale = ast.literal_eval(config_common['plot scale'])
frame_width = ast.literal_eval(config_common['frame width'])
figure_dimensions = ast.literal_eval(config_common['figure dimensions'])
image_format = ast.literal_eval(config_common['image format'])
energy_zero = ast.literal_eval(config_common['zero energy'])
energy_limits = ast.literal_eval(config_common['energy limits'])
energy_label = ast.literal_eval(config_common['energy label'])
energy_lines = ast.literal_eval(config_common['energy lines'])
energy_range_priority = ast.literal_eval(config_common['energy range priority'])
energy_units = ast.literal_eval(config_common['energy units'])
if bands_mode:
    config_bands = config['BANDS MODE']
    bands_folder = ast.literal_eval(config_bands['folder'])
    bands_file = ast.literal_eval(config_bands['file'])
    num_empty_bands = ast.literal_eval(config_bands['number of empty bands'])
    k_points = ast.literal_eval(config_bands['wavevector points'])
    k_offset = ast.literal_eval(config_bands['wavenumber offset'])
    bands_title = ast.literal_eval(config_bands['plot title'])
    k_label = ast.literal_eval(config_bands['wavevector label'])
    bands_colors = ast.literal_eval(config_bands['plot colors'])
if dos_mode:
    config_dos = config['DENSITY OF STATES MODE']
    spin_orbit = config_dos.getboolean('spin-orbit')
    total_dos = config_dos.getboolean('total density of states')
    export_dos = config_dos.getboolean('export densities of states')
    energy_zoom = config_dos.getboolean('zoom in energy')
    dos_folder = ast.literal_eval(config_dos['folder'])
    dos_prefix = ast.literal_eval(config_dos['files prefix'])
    projections = ast.literal_eval(config_dos['projections'])
    density_limits = ast.literal_eval(config_dos['density of states limits'])
    energy_zoom_params = ast.literal_eval(config_dos['energy zoom parameters'])
    dos_title = ast.literal_eval(config_dos['plot title'])
    density_label = ast.literal_eval(config_dos['density of states label'])
    density_units = ast.literal_eval(config_dos['density of states units'])
    dos_colors = ast.literal_eval(config_dos['plot colors'])
    
# Default graphical options.
plt.rcParams.update({'font.family': font})
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'axes.linewidth': frame_width})
for i in ['x','y']:
    plt.rcParams.update({i+'tick.major.width': frame_width})
    plt.rcParams.update({i+'tick.minor.width': 0.8*frame_width})

print()

# %% Reading and processing of the bands data %%

if bands_mode:
    
    # Reading of the data.
    bands_data = np.loadtxt(full_path(bands_folder + bands_file))
    wavenumber = bands_data[:,0]
    energy_k = bands_data[:,1] - energy_zero
    # Bands parameters.
    num_bands = list(wavenumber).count(wavenumber[0])
    inds = np.linspace(0, len(wavenumber), num_bands+1, dtype=int)
    num_band_points = np.diff(inds)[0]
    if num_empty_bands != '':
        num_occupied_bands = num_bands - num_empty_bands
    else:
        num_occupied_bands = ''
    # Points of the reciprocal space
    k_names, k_coords = [], []
    for i in range(len(k_points)):
        k_names += list(k_points[i].keys())
        k_coords += list(k_points[i].values())
    for i in range(len(k_names)):
        if k_names[i].startswith('|'):
            k_names[i] = '$\\overline{\\mathrm{' + k_names[i][1:] + '}}$'
    # Calculation of the points which delimite the paths in the wavenumber.
    if k_names != []:
        k_locs = np.zeros(len(k_names))  # límites de los tramos
        k_locs[0] = wavenumber[0]
        k_locs[-1] = wavenumber[-1]
        for i in range(1, len(k_names)-1):
            k_locs[i] = wavenumber[i*int(num_band_points/(len(k_names)-1))
                                   - k_offset]

#%% Reading and processing of the density of states data %%

if dos_mode:

    # Reading of all the density of states file.
    dos_files = glob.glob(full_path(dos_folder + dos_prefix) + '.pdos_atm*')
    original_data = []
    for file in dos_files:
        original_data += [np.loadtxt(file)]

    # Selection of the energy vector and reading of the total density of states,
    # in case that option was selected.
    energy = original_data[0][:,0] - energy_zero
    if total_dos:
        total_density = np.loadtxt(full_path(dos_folder + dos_prefix)
                                   + '.pdos_tot')[:,2]

    # Creation of a tensor with the data of the selected densities with an
    # appropiate structure for its posterior processing.
    dos_data = [[], [], [], [], [], []]
    if spin_orbit:
        dos_data += [[]]
    # Loop for the data of each file.
    for i in range(len(original_data)):
        # Loop for each column of density of states.
        for j in range(2, original_data[i].shape[1]):
            name = dos_files[i] + str(j-1)
            dos_data[0] += [name]  # file name
            dos_data[1] += [original_data[i][:,j]]  # density of states
            string1 = name.split('#')
            string2 = string1[1].split('(')
            string3 = string1[2].split('(')
            dos_data[2] += [string2[0]]  # atom number
            dos_data[3] += [string2[1][:-5]]  # element
            dos_data[4] += [string3[1][0]]  # orbital
            if not spin_orbit:
                dos_data[5] += [string3[1][-1]]  # orientation    
            else:
                string4 = string3[1].split(')')
                dos_data[5] += [string4[0][3:]]  # angular moment (j)
                dos_data[6] += [string4[1]]  # orientation
    dos_data = np.array(dos_data, object)

    #%% Selection of the desired projected densities of states according to the
    #   parameters of the configuration file %%

    # Creation of the parameter list with the appropiate format from the
    # configuration file, and creation of a similar list of names.
    params, names = [], []
    for i in range(len(projections)):        
        # Auxiliary variables for the sum mode.
        if projections[i][1] == 'sum': 
            sum_params, sum_name = [], []
        # Loop for each row of projections.
        for j in range(len(projections[i]) - 1):    
            # Parameters of the projections.
            atoms = str(projections[i][j][0])
            elements = projections[i][j][1]
            orbitals = projections[i][j][2]           
            if not spin_orbit:    
                orientations = str(projections[i][j][3])           
            else:
                momentums = str(projections[i][j][3])
                orientations = str(projections[i][j][4])
            # It sum mode, it saves the names together.
            if projections[i][-1] == 'sum':                
                if not spin_orbit:
                    sum_name += [[atoms, elements, orbitals, orientations]]
                else:
                    sum_name += [[atoms, elements, orbitals, momentums,
                                  orientations]] 
            # Loops for the abbreviated writing mode.
            for atom in atoms.split('+'):               
                for element in elements.split('+'):                
                    for orbital in orbitals.split('+'):
                        if not spin_orbit:                        
                            for orientation in orientations.split('+'):     
                                # Addition the parameters and names.
                                if projections[i][-1] == 'ind':
                                    params += [[[atom, element, orbital,
                                                 orientation]]]
                                    names += [[[atom, element, orbital,
                                                orientation]]]
                                elif projections[i][-1] == 'sum':
                                    sum_params += [[atom, element, orbital,
                                                    orientation]]
                        else:                            
                            for momentum in momentums.split('+'):
                                for orientation in orientations.split('+'):
                                    # Addition of parameters and names.
                                    if projections[i][-1] == 'ind':
                                        params += [[[atom, element, orbital,
                                                     momentum, orientation]]]
                                        names += [[[atom, element, orbital,
                                                    momentum, orientation]]]
                                    elif projections[i][-1] == 'sum':
                                        sum_params += [[atom, element, orbital,
                                                        momentum, orientation]]
        # If sum mode, now is the addition of parameters and names.                                 
        if projections[i][-1] == 'sum':            
            names += [sum_name]
            params += [sum_params]            
    
    # Calculation of the conditional masks to select the densities of states,
    # and creation of another vector with names to label each curve on the plot.
    conds, labels = [[]]*len(params), ['']*len(params)
    for i in range(len(params)):    
        conds[i] = np.zeros(dos_data[1].shape, dtype=bool)        
        for j in range(len(params[i])):
            cond_ijk = []
            # Loop for the parameters of the projections.
            for k in range(4 + int(spin_orbit)):
                if params[i][j][k] == '':  # non-specified parameter
                    cond_ijk += [np.ones(dos_data[2+k,:].shape, dtype=bool)]
                else:
                    cond_ijk += [dos_data[2+k,:] == params[i][j][k]]
            # Multiplication of the conditions for each parameter.
            cond_ij = cond_ijk[0]
            for k in range(1, len(cond_ijk)):
                cond_ij &= cond_ijk[k]
            # Addition of the conditions for the projections of the same row.
            conds[i] += cond_ij
        # Labels of the projections.
        for j in range(len(names[i])):
            labels[i] += '[' + ','.join(names[i][j]) + ']'    
        labels[i] = ']+\n['.join(labels[i].split(']['))
    # Label of the total density of states.
    if spin_orbit:
        label_total_dens = '[,,,,]'
    else:
        label_total_dens = '[,,,]'

    # Subtraction of the null selections, where none of the requirements are
    # fullfilled.
    k = 0  # count of subtracted cases
    for i in range(len(conds)):  # loop in reversed order
        if sum(conds[i-k]) == 0:   
            conds.pop(i-k)
            names.pop(i-k)
            labels.pop(i-k)
            k += 1

    # Selection of the desired projected densities of states.
    density = np.zeros((len(energy), len(conds)))
    for i in range(len(conds)):
        density[:,i] = sum(dos_data[1,conds[i]])

    #%% Creation of the output file, in case this option was selected.
       
    if export_dos and len(conds) != 0:
        
        # Data matrix.
        file_data = np.zeros((len(energy), len(conds) + 1))
        file_data[:,0] = energy   
        file_data[:,1:] = density
        # File name.
        file_name = '#'.join(labels).replace('\n', '')
        file_name = file_name.replace('#', '')
        file_name = dos_prefix + '-' + file_name
        file_name = file_name.replace('.dat', '').replace('txt', '') + '.txt'
        # Column titles.
        titles = '#'.join(labels).replace('\n', '').split('#')
        # Determination of the blank spaces for separing each column
        # (and adjusting in case any title is too much large)
        num_blanks = 20  # number of blank spaces for each column
        extra_blanks = 0  # correction blank spaces
        for i in range(len(titles)):
            # If it is enough space until the next column, there is no
            # correction.
            if (num_blanks - len(titles[i]) - extra_blanks) >= 2:
                blank_spaces = ' '*(num_blanks - len(titles[i]) - extra_blanks)
                extra_blanks = 0
            # If there is not enough space until the next column, there is a
            # slight shift in the title of that next column.
            else:  
                 blank_spaces = '  '
                 extra_blanks += len(titles[i]) - num_blanks + 2 
            titles[i] = titles[i] + blank_spaces  # títulos de las columnas               
        # Blanck spaces to align the column titles.
        num_blanks_energy = num_blanks - 5 - len(energy_units)
        num_blanks_density = num_blanks + 2 - len(density_units)
        num_blanks_delimiter = num_blanks - 9
        
        # Final title.
        title = ('E (' + energy_units + ')' + ' '*num_blanks_energy
                 + 'pDOS (' + density_units + ')\n'
                 + ' '*num_blanks_density + ''.join(titles))
        # Exporting of the data to a text file.
        np.savetxt(file_name, file_data, fmt='%1.3e', header=title,
                   delimiter=' '*num_blanks_delimiter)
        
        print('The file ' + file_name + ' has been generated.')
 
#%% Plots of the density of states mode (and both modes at the same time) %%

if dos_mode:

    if len(conds) == 0:
        raise ValueError('There is no density of states meeting the requirements.')

    # Limits of the energy and the density of states.
    original_energy_limits = copy.copy(energy_limits)
    original_density_limits = copy.copy(density_limits)
    energy_limits = replace_limits(energy_limits, energy)
    # Limits of the energy for the zoomed plot.
    energy_center, energy_amplitude = tuple(energy_zoom_params.values())
    energy_zoom_limits = [energy_center - energy_amplitude,
                          energy_center + energy_amplitude]

    # If there is also a band plot, calculation of the energy limits.
    if bands_mode:
        # Energy limits.
        bands_energy_limits = copy.copy(original_energy_limits)
        bands_energy_limits = replace_limits(bands_energy_limits, energy_k)
        # Energy range that determines the limits in that variable.
        if energy_range_priority not in ['bands', 'density of states',
                                         'intersection', 'union']:
            raise ValueError("Energy range priority should be 'bands', "
                             + "'density of states', 'intersection' or 'union'")
        if energy_range_priority == 'bands':
            energy_limits = copy.copy(bands_energy_limits)
        elif energy_range_priority == 'density of states':
            energy_limits = energy_limits
        elif energy_range_priority == 'intersection':
            energy_limits[0] = max(energy_limits[0], bands_energy_limits[0])
            energy_limits[1] = min(energy_limits[1], bands_energy_limits[1])
        elif energy_range_priority == 'union':
            energy_limits[0] = min(energy_limits[0], bands_energy_limits[0])
            energy_limits[1] = max(energy_limits[1], bands_energy_limits[1])

    # Adding of a little margin to energy limits.
    energy_limits = add_margin_to_limits(energy_limits, original_energy_limits,
                                         width=1/25)

    # Densities of states used to determine the limits in this variable.
    dens_comb = np.array([])
    cond = (energy >= energy_limits[0]) & (energy <= energy_limits[1])
    for i in range(density.shape[1]):
        dens_comb = np.concatenate((dens_comb, density[cond,i]))
    if total_dos:
        dens_comb = np.concatenate((dens_comb, total_density[cond]))
    dens_comb = dens_comb[dens_comb != 0]
        
    # Algorithm for the automatic determination of the superior limit of the
    # density of states. It calculates de percentile function for a range of
    # percentile indices and detects in which index its derivative increases
    # considerably, using that index to calculate the desired index.
    if density_limits[1] == 'auto':
        mad = median_absolute_deviation(dens_comb)
        indices = np.linspace(92, 100, int(1E3))  # percentile indices
        percentiles = np.percentile(dens_comb, indices) / mad
        derivative = np.diff(percentiles) / np.diff(indices)
        derivative_non0 = derivative[derivative != 0]
        reference = np.median(derivative_non0[:int(0.5*len(derivative_non0))])
        # Condition to obtain the desired percentile index.
        cond = derivative > 50*reference
        if cond.sum() != 0:
            limit_index = indices[1:][cond].min()
        else:
            limit_index = indices[-1]
        # Limit obtained with the limit percentile index.
        cond = dens_comb < np.percentile(dens_comb, limit_index)
        margin = (dens_comb[cond].max() - dens_comb[cond].min()) / 10
        dens_lim_perc = np.percentile(dens_comb, limit_index) + margin + 1.5*mad
        # percentile_diagnostic_plots(indices, percentiles, derivative,
        #                             limit_index, reference, 50)
        # Limit obtained with the maximum of the density of states plus a margin.
        margin = (dens_comb.max() - dens_comb.min()) / 25
        dens_lim_norm = dens_comb.max() + margin
        # Minimum of both limits.
        density_limits[1] = min(dens_lim_norm, dens_lim_perc)

    # Corrections on the density of states limits.
    density_limits = replace_limits(density_limits, dens_comb)
    density_limits = add_margin_to_limits(density_limits,
                                          original_density_limits, width=1/25)
    if original_density_limits[0] == '':
        density_limits[0] = max(0, density_limits[0])
    
   # Figures. (f = 1 for the zoomed plot)
    for f in reversed(range(1 + int(energy_zoom))):
        
        if bands_mode:
            plt.figure(f+1, figsize=(figure_dimensions[0]*2,figure_dimensions[1]))
        else:
            plt.figure(f+1, figsize=figure_dimensions)
        plt.clf()
        
        if bands_mode:
            # Bands plot (left).
            ax1 = plt.subplot(1,2,1)

            # Symbol colors, depending on if the occupied and empty bands are
            # known.
            if num_occupied_bands == '':
                num_occupied_bands = num_bands
                color1, color2 = [bands_colors[0]]*2
            else:
                color1, color2 = bands_colors[1:]
  
            # Plotting of the occupied bands
            # (or all of them if the number of occupied ones is unknown).
            for i in range(0, num_occupied_bands):  # bucle para cada banda
                wavenumber_i = wavenumber[inds[i] : inds[i+1]]
                energy_i = energy_k[inds[i] : inds[i+1]]
                plt.plot(wavenumber_i, energy_i, '.-', color=color1,
                         ms=6*scale, lw=1*scale, alpha=0.3/scale, zorder=1/3)
            # Plotting of the empty bands
            # (or none else if the number of empty bands is unksnown).
            for i in range(num_occupied_bands, num_bands):  # loop for each band
                wavenumber_i = wavenumber[inds[i] : inds[i+1]]
                energy_i = energy_k[inds[i] : inds[i+1]]
                plt.plot(wavenumber_i, energy_i, '.-', color=color2,
                         ms=6*scale, lw=1*scale, alpha=0.3/scale, zorder=1/3)       
            plt.margins(x=0)

            # Limits.
            if f == 1:
                plt.ylim(energy_zoom_limits)
            else:
                plt.ylim(energy_limits)
            xlims, ylims = plt.xlim(), plt.ylim()
            # Vertical lines to separate the paths.
            if k_names != '':
                for x in k_locs:
                    plt.vlines(x, ylims[0], ylims[1], color='black',
                               lw=0.8*frame_width, zorder=3/3)  
            # Horizontal lines for the selected energies.
            for y in energy_lines:
                plt.hlines(y, xlims[0], xlims[1], color='black',
                           linestyle=(0,(6,6)), lw=0.8*frame_width, zorder=2/3)
            # Labels of the reciprocal space.
            if k_names != []:
                plt.xticks(k_locs, k_names)
            else:
                plt.xticks([])   
            # Text of the axes and the title.
            plt.xlabel(k_label, labelpad=6)
            plt.ylabel(energy_label, labelpad=6)
            plt.title(bands_title, fontweight='bold', pad=10)
            plt.tight_layout()

            # Densities of states plot (right).
            ax2 = plt.subplot(1,2,2, sharey=ax1)

        # Plotting of the densities of states.
        for i in range(len(conds)):   
            if labels[i] != label_total_dens:
                color, ls, lw = dos_colors[i%len(dos_colors)], '-', 2
            else:
                color, ls, lw = 'black', '--', 1                    
            plt.plot(density[:,i], energy, label=labels[i], color=color,
                     linestyle=ls, linewidth=lw*scale)
        if total_dos:
            plt.plot(total_density, energy, color='black', linestyle='--',
                     linewidth=1*scale, label=label_total_dens)
            
        # Limits.
        plt.xlim(density_limits)
        if f == 1: # zoomed plot
            plt.ylim([energy_zoom_limits])
        else:
            plt.ylim(energy_limits)  
        xlims, ylims = plt.xlim(), plt.ylim()
        # Horizontal lines for the selected energies.
        for y in energy_lines:
            plt.hlines(y, xlims[0], xlims[1], color='black', lw=0.8*frame_width,
                       linestyle=(0,(6,6)), zorder=(len(conds)+2)/3)
        plt.xlim(xlims)
        plt.ylim(ylims)
        # Tweaks on the vertical axis.
        if bands_mode:
            ax2.yaxis.tick_right()
            # plt.tick_params(left=True)
        else:
            plt.ylabel(energy_label, labelpad=6)
        # Plot texts.
        plt.xlabel(density_label, labelpad=6)
        plt.legend(ncol=1, markerfirst=False)
        plt.title(dos_title, fontweight='bold', pad=12)
        plt.tight_layout()
        # Marks.
        # ax2.yaxis.set_major_locator(plt.AutoLocator())
        # ax2.minorticks_on()
    
    # Creation of an image for each figure (normal and zoomed).
    if save_image: 
        image_name = '-' + ''.join(labels).replace('\n', '')
        extension = ['','-z']  # extension for the zoomed plot
        for f in range(1 + int(energy_zoom)):
            plt.figure(f+1)
            image_name = (dos_prefix + image_name + extension[f]
                          + '.' + image_format)
            if bands_mode:
                image_name = bands_file + '-' + image_name
            image_name = image_name.replace('.dat', '').replace('.txt', '')
            plt.savefig(image_name, format=image_format, dpi=240)
            print('Figure ' + str(f+1) + ' has been saved as ' + image_name + '.')
    
    plt.show()

#%% Bands mode plot %%

elif bands_mode and not dos_mode:
    
    # Energy limits.
    original_energy_limits = copy.copy(energy_limits)
    energy_limits = replace_limits(energy_limits, energy_k)
    energy_limits = add_margin_to_limits(energy_limits, original_energy_limits,
                                         width=1/25)    
    # Figure.
    fig = plt.figure(1, figsize=figure_dimensions), plt.clf()
    # Colors.
    if num_occupied_bands != '':
        color1, color2 = bands_colors[1:]
    else:
        num_occupied_bands = num_bands
        color1, color2 = [bands_colors[0]]*2

    # Plotting of the occupied bands
    # (or all of them if the number of occupied ones is unknown).
    occupied_bands_maxs = []
    for i in range(0, num_occupied_bands):  # bucle para cada banda
        wavenumber_i = wavenumber[inds[i] : inds[i+1]]
        energy_i = energy_k[inds[i] : inds[i+1]]
        plt.plot(wavenumber_i, energy_i, '.-', color=color1, ms=6*scale,
                 lw=1*scale, alpha=0.3/scale, picker=1, zorder=1/3)
        occupied_bands_maxs += [max(energy_i)]
    # Plotting of the occupied bands
    # (or all of them if the number of occupied ones is unknown).
    if num_empty_bands != '':
        empty_bands_mins = [] 
        for i in range(num_occupied_bands, num_bands):  # loop for each band
            wavenumber_i = wavenumber[inds[i] : inds[i+1]]
            energy_i = energy_k[inds[i] : inds[i+1]]
            plt.plot(wavenumber_i, energy_i, '.-', color=color2, ms=6*scale,
                     lw=1*scale, alpha=0.3/scale, picker=1, zorder=1/3)
            empty_bands_mins += [min(energy_i)]

        # Case of distinguishing occupied and empty bands.
        if num_empty_bands > 0 and num_occupied_bands > 0:
            # Calculation of the maximum of the confuction band and the minimum
            # of the valence band.
            cond_band_min = min(occupied_bands_maxs)
            val_band_max = max(empty_bands_mins)       
            cond_band_min_k = wavenumber[energy_k == cond_band_min][0]
            val_band_max_k = wavenumber[energy_k == val_band_max][0]
            cond_band_min_point = find_wavevector(cond_band_min_k, k_locs, k_coords)
            val_band_max_point = find_wavevector(val_band_max_k, k_locs, k_coords) 
            # Terminal text.
            print('Points:', k_names)
            if len(k_coords) != 0:
                print('{' + ', '.join([vector_to_text(vec) for vec in k_coords])
                      + '}')
            print('Conduction band mimimum:', str(cond_band_min), energy_units,
                  ('{' + str(cond_band_min_k) + '}').replace('.0}', '}'),
                  vector_to_text(cond_band_min_point))
            print('Valence band maximum:', str(val_band_max), energy_units, 
                  ('{' + str(val_band_max_k) + '}').replace('.0}', '}'),
                  vector_to_text(val_band_max_point))

    # Axes limits.
    plt.margins(x=0)
    plt.ylim([energy_limits[0], energy_limits[1]])
    xlims, ylims = plt.xlim(), plt.ylim()
    # Vertical lines to separate paths.
    if k_names != []:
        for x in k_locs:
            plt.vlines(x, ylims[0], ylims[1], color='black',
                       lw=0.8*frame_width, zorder=3/3)
        plt.xticks(k_locs, k_names)
    # Horizontal lines for the selected energies.
    for y in energy_lines:
        plt.hlines(y, xlims[0], xlims[1], color='black', lw=0.8*frame_width,
                   linestyle=(0,(6,6)), zorder=2/3)     
    # Figure texts.
    plt.xlabel(k_label, labelpad=6)
    plt.ylabel(energy_label, labelpad=6)
    plt.title(bands_title, fontweight='bold', pad=12)
    plt.tight_layout()

    # Cretion of an image of the plot.
    if save_image:
        image_name = bands_file + '.' + image_format
        image_name = image_name.replace('.dat', '').replace('.txt', '')
        plt.savefig(image_name, format=image_format, dpi=240)
        print('Figure 1 has been saved as ' + image_name + '.')

    # Interactive mode.
    fig[0].canvas.mpl_connect('pick_event', click)

    plt.show()

print()
