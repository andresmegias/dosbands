#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOSBANDS.PY
1.3

Andrés Megías

El código dosbands.py, escrito en Python 3, permite representar bandas
electrónicas y densidades de estados producidas por el programa Quantum
ESPRESSO. Se necesitan los módulos os, sys, ast, copy, glob, pathlib y
configparser, de la librería estándar de Python, y también las librerías
Numpy y Matplotlib.
    Para ejecutar el código sobre un archivo de bandas o de densidades de
estados (o ambos), es necesario especificar algunos parámetros en un archivo
de configuración. Para ejecutar el código debe escribirse en la terminal,
estando en la carpeta del archivo dosbands.py:
    
		python3 dosbands.py <dirección>  .
        
    El argumento <dirección> es la ruta del archivo de configuración; si no se
especifica, se usará la ruta './config.txt', es decir,se buscará en la carpeta
actual (./) el archivo de configuración con nombre config.txt. Si el archivo
dosbands.py no está en la carpeta actual, sino en una carpeta con ruta <carpeta>,
debe escribirse '<carpeta>/dosbands.py' en lugar de 'dosbands.py'. Además, si se
incluye el archivo dosbands.py en una carpeta de ejecutables del sistema,
puede suprimirse 'python3' de la orden de ejecución.
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
    Obtiene el vector de onda correspondiente a un número de onda.
    
    Convierte un valor numérico de un número de onda cristalino correspondiente
    a un solo tramo del espacio recíproco a un vector de onda cristalino, si
    se conocen los puntos que delimitan dicho tramo.
    
    Argumentos
    ----------
    ki : float
        Número de onda cristalino que se desea convertir.
    k1, k2 : float
        Números de onda cristalinos de los puntos que delimitan el tramo del
        espacio recíproco donde se encuentra ki.   
    K1, K2 : array (float)
        Vectores de onda cristalinos asociados a los puntos k1,k2.
        
    Resultado
    ---------
    Ki : array (float)
        Vector de onda cristalino correspondiente a ki.
    """
    K1, K2 = np.array(K1), np.array(K2)
    Ki = K1 + (K2-K1)*(ki-k1)/(k2-k1)
    
    return Ki

def find_wavevector(ki, locs, coords):
    """
    Obtiene el vector de onda correspondiente a un número de onda.
    
    Convierte un valor numérico de un número de onda cristalino correspondiente
    a varios tramos del espacio recíproco a un vector de onda cristalino.
    Primero encuentra los puntos que delimitan el tramo donde se encuentra este
    número de onda y después aplica la función get_wavevector.
    
    Argumentos
    ----------
    ki : float
        Número de onda cristalino que se desea convertir.
    locs : array (float)
        Números de onda cristalinos de los puntos que delimitan
        los tramos del espacio recíproco.  
    coords : array (float)
        Vectores de onda de los puntos que delimitan los tramos del espacio
        recíproco.
        
    Resultado
    ---------
    Ki : array (float)
        Vector de onda cristalino correspondiente a ki.
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
    Transforma un vector a texto, usando las opciones de formato especificadas.
    
    Argumento
    ----------
    v : array
        Vector numérico que desea transformarse a texto.
    delimiters: str
        Texo con los dos caracteres que quieran usarse como delimitadores
        del vector.
    round_digits : int
        Número de cifras decimales máximas de cada elemento del vector.
    zeros : bool
        Determina si los números enteros se expresan como tal; por ejemplo,
        '0' en vez de '0.0', '2' en vez de '2.0'.
    commas : bool
        Determina si se utilizan comas para separar los elementos del vector.
    blank : bool
        Determina si se añade un espacio entre los elementos del vector y los
        delimitadores.
        
    Resultado
    ---------
    s : str
        Texto del vector v, con las opciones de formato especificadas.
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
    Permite obtener las coordenadas de un punto de una gráfica al hacer click.
    
    Argumento
    ---------
    event : event
    Evento de Python correspondiente a hacer click sobre la gráfica.
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
    Obtiene la ruta completa descrita en una cadena de texto.
    
    Corrige el formato de la dirección, permitiendo el uso de las abreviaciones
    propias del sistema operativo (por ejemplo, './', '../' y '~' para Unix).
    
    Argumento
    ---------
    text : str
        Texto de ruta o dirección, que puede tener abreviaciones propias del
        sistema operativo usado.
        
    Resultado
    ---------
    path : str
        Texto de la ruta completa, de modo que Python pueda usarlo.
    """
    path = copy.copy(text)
    if path.startswith('~'):
        path = os.path.expanduser('~') + path[1:]  
    path = str(pathlib.Path(path).resolve())
    
    return path

def replace_limits(original_limits, variable):
    """
    Cambia los valores de los límites de texto por números.
    
    Sustituye de los límites de la variable no especificados ('') y los que
    hacen referencia al máximo ('max') o al mínimo ('min').
    
    Argumentos
    ----------
    original_limits : list
        Lista con los límites inferior y superior de la variable en cuestión.
        Puede contener cadenas de texto, que serán traducidas a números.
    variable : array
        Vector con los valores de la variable que se está tratando.
    
    Resultado
    ---------
    new_limits : list
        Lista con los límites inferior y superior de la variable en cuestión,
        con ambos valores numéricos.
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
    Añade un margen a los límites especificados cuando sea oportuno.
    
    Si los límites originales no estaban especificados (''), añade un margen
    a los actuales límites.
    
    Argumentos
    ----------
    limits : list
        Límites actuales, ambos valores numéricos.
    original_limits : list
        Límites originales, numéricos o de texto.
    width : float
        Tamaño del margen relativo al rango determinado por los límites.
    
    Resultado
    ---------
    new_limits : list
        Nuevos límites con el margen.
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
    Desviación absoluta mediana.
    
    Argumentos
    ----------
    x : array (float)
        Vector.
    c : float
        Factor de normalización; por defecto tiene un valor tal que la
        desviación absoluta mediana de una distribución normal coincida con
        su desviación estándar.
    
    Resultado
    ---------
    mad : float
        Desviación absoluta mediana.
    """
    mad = c*np.median(abs(x - np.median(x)))
    
    return mad

def percentile_diagnostic_plots(indices, percentiles, derivative, limit_index,
                                reference, condition_limit):
    """
    Muestra dos gráficos de diagnóstico.
    
    Muestra dos gráficos de diagnóstico del algoritmo de búsqueda automática
    del límite superior de la densidad de estados.
    
    Argumentos
    ----------
    indices : array (float)
        Índices percentiles.
    percentiles : array (float)
        Función percentil en función del índice.
    derivative : array (float)
        Derivada de la función percentil en función del índice.
    limit_index : float
        Índice percentil calculado.
    reference : float
        Valor de referencia para calcular el límite en la derivada
    condition_limit : float
        Número por el que se multiplica el valor de referencia para así
        calcular el límite de la derivada
    """
    plt.figure(3, figsize=(10,8))
    
    sp1 = plt.subplot(2,1,1)
    plt.plot(indices, percentiles, color='tab:green')
    ylim1 = plt.ylim()[1]
    plt.vlines(limit_index, 0, ylim1, linestyle='--', alpha=0.6)
    plt.ylim([0, ylim1])
    plt.margins(x=0)
    plt.xlabel('índice percentil')
    plt.ylabel('percentil', labelpad=6)
    
    plt.subplot(2,1,2, sharex=sp1)
    plt.plot(indices[1:], derivative, color='tab:green')
    ylim1 = plt.ylim()[1]
    plt.vlines(limit_index, 0, ylim1, linestyle='--', alpha=0.6)
    plt.hlines(reference, indices[0], indices[-1], linestyle='-.', alpha=0.6)
    plt.hlines(reference*condition_limit, indices[0], indices[-1],
               linestyle='--', alpha=0.6)
    plt.ylim([0, ylim1])
    plt.margins(x=0)
    plt.xlabel('índice percentil')
    plt.ylabel('derivada del percentil', labelpad=8)
    
    plt.tight_layout()

#%% Valores predeterminados de las variables %%

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

# Lectura del archivo de configuración.
if len(sys.argv) == 1:
    config_path = full_path('./config.txt')  # dirección predeterminada
else:
    config_path = full_path(sys.argv[1])
config_folder = '/'.join(config_path.split('/')[:-1]) + '/'
os.chdir(config_folder)
if os.path.isfile(config_path):
    config.read(config_path)
else:
    raise FileNotFoundError('Configuration file not found.')

# Lectura de los argumentos.
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
    
# Opciones gráficas predeterminadas.
plt.rcParams.update({'font.family': font})
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'axes.linewidth': frame_width})
for i in ['x','y']:
    plt.rcParams.update({i+'tick.major.width': frame_width})
    plt.rcParams.update({i+'tick.minor.width': 0.8*frame_width})

# %% Lectura y tratamiento de los datos de las bandas %%

if bands_mode:
    
    # Lectura de los datos.
    bands_data = np.loadtxt(full_path(bands_folder + bands_file))
    wavenumber = bands_data[:,0]
    energy_k = bands_data[:,1] - energy_zero
    # Parámetros de las bandas.
    num_bands = list(wavenumber).count(wavenumber[0])
    inds = np.linspace(0, len(wavenumber), num_bands+1, dtype=int)
    num_band_points = np.diff(inds)[0]
    if num_empty_bands != '':
        num_occupied_bands = num_bands - num_empty_bands
    else:
        num_occupied_bands = ''
    # Puntos del espacio recíproco.
    k_names, k_coords = [], []
    for i in range(len(k_points)):
        k_names += list(k_points[i].keys())
        k_coords += list(k_points[i].values())
    for i in range(len(k_names)):
        if k_names[i].startswith('|'):
            k_names[i] = '$\\overline{\\mathrm{' + k_names[i][1:] + '}}$'
    # Obtención de los puntos que delimitan los tramos en el número de ondas.
    if k_names != []:
        k_locs = np.zeros(len(k_names))  # límites de los tramos
        k_locs[0] = wavenumber[0]
        k_locs[-1] = wavenumber[-1]
        for i in range(1, len(k_names)-1):
            k_locs[i] = wavenumber[i*int(num_band_points/(len(k_names)-1))
                                   - k_offset]

#%% Lectura y tratamiento de los datos de las densidades de estados %%

if dos_mode:

    # Lectura de todos los ficheros de densidades de estados.
    dos_files = glob.glob(full_path(dos_folder + dos_prefix) + '.pdos_atm*')
    original_data = []
    for file in dos_files:
        original_data += [np.loadtxt(file)]

    # Selección del vector de energías y lectura de la densidad total de estados,
    # en caso de que se haya seleccionado dicha opción.
    energy = original_data[0][:,0] - energy_zero
    if total_dos:
        total_density = np.loadtxt(full_path(dos_folder + dos_prefix)
                                   + '.pdos_tot')[:,2]

    # Creación de un tensor con los datos de las densidades seleccionadas con
    # una estructura adecuada para su posterior tratamiento.
    dos_data = [[], [], [], [], [], []]
    if spin_orbit:
        dos_data += [[]]
    # Bucle para los datos de cada archivo de datos.    
    for i in range(len(original_data)):
        # Bucle para cada columna de densidad de estados.
        for j in range(2, original_data[i].shape[1]):
            name = dos_files[i] + str(j-1)
            dos_data[0] += [name]  # nombre del fichero
            dos_data[1] += [original_data[i][:,j]]  # densidad de estados
            string1 = name.split('#')
            string2 = string1[1].split('(')
            string3 = string1[2].split('(')
            dos_data[2] += [string2[0]]  # múmero de átomo
            dos_data[3] += [string2[1][:-5]]  # elemento
            dos_data[4] += [string3[1][0]]  # orbital
            if not spin_orbit:
                dos_data[5] += [string3[1][-1]]  # orientación    
            else:
                string4 = string3[1].split(')')
                dos_data[5] += [string4[0][3:]]  # momento angular (j)
                dos_data[6] += [string4[1]]  # orientación    
    dos_data = np.array(dos_data)

    #%% Selección de las densidades de estados proyectadas deseadas según los
    #   parámetros del archivo de configuración %%

    # Creación de la lista de parámetros con el formato adecuado a partir
    # del archivo de configuración y creación de una lista similar de nombres.
    params, names = [], []
    for i in range(len(projections)):        
        # Variables auxiliares para el modo suma.
        if projections[i][1] == 'sum': 
            sum_params, sum_name = [], []
        # Bucle para cada fila de proyecciones.
        for j in range(len(projections[i]) - 1):    
            # Parámetros de las proyecciones.
            atoms = str(projections[i][j][0])
            elements = projections[i][j][1]
            orbitals = projections[i][j][2]           
            if not spin_orbit:    
                orientations = str(projections[i][j][3])           
            else:
                momentums = str(projections[i][j][3])
                orientations = str(projections[i][j][4])
            # Si está el modo suma, guarda los nombres juntos.
            if projections[i][-1] == 'sum':                
                if not spin_orbit:
                    sum_name += [[atoms, elements, orbitals, orientations]]
                else:
                    sum_name += [[atoms, elements, orbitals, momentums,
                                  orientations]] 
            # Bucles por el modo de escritura abreviado.
            for atom in atoms.split('+'):               
                for element in elements.split('+'):                
                    for orbital in orbitals.split('+'):
                        if not spin_orbit:                        
                            for orientation in orientations.split('+'):     
                                # Adición de los parámetros y los nombres.
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
                                    # Adición de los parámetros y los nombres.
                                    if projections[i][-1] == 'ind':
                                        params += [[[atom, element, orbital,
                                                     momentum, orientation]]]
                                        names += [[[atom, element, orbital,
                                                    momentum, orientation]]]
                                    elif projections[i][-1] == 'sum':
                                        sum_params += [[atom, element, orbital,
                                                        momentum, orientation]]
        # Si está el modo suma, ahora se añaden los parámetros y los nombres.                                        
        if projections[i][-1] == 'sum':            
            names += [sum_name]
            params += [sum_params]            
    
    # Determinación de las máscaras condicionales para seleccionar las
    # densidades de estados y creación de otro vector de nombres para etiquetar
    # cada curva en la gráfica.         
    conds, labels = [[]]*len(params), ['']*len(params)
    for i in range(len(params)):    
        conds[i] = np.zeros(dos_data[1].shape, dtype=bool)        
        for j in range(len(params[i])):
            cond_ijk = []
            # Bucle para los parámetros de las proyecciones.
            for k in range(4 + int(spin_orbit)):
                if params[i][j][k] == '':  # parámetro sin especificar
                    cond_ijk += [np.ones(dos_data[2+k,:].shape, dtype=bool)]
                else:
                    cond_ijk += [dos_data[2+k,:] == params[i][j][k]]
            # Producto de las condiciones para cada parámetro.
            cond_ij = cond_ijk[0]
            for k in range(1, len(cond_ijk)):
                cond_ij &= cond_ijk[k]
            # Adición de las condiciones de las proyecciones de la misma fila.
            conds[i] += cond_ij
        # Etiquetas de las proyecciones.
        for j in range(len(names[i])):
            labels[i] += '[' + ','.join(names[i][j]) + ']'    
        labels[i] = ']+\n['.join(labels[i].split(']['))
    # Etiqueta de la densidad de estados total.
    if spin_orbit:
        label_total_dens = '[,,,,]'
    else:
        label_total_dens = '[,,,]'

    # Sustracción de las selecciones nulas, donde no se cumple ninguno de los
    # requisitos especificados.
    k = 0  # recuento de casos sustraídos
    for i in range(len(conds)):  # bucle en orden inverso
        if sum(conds[i-k]) == 0:   
            conds.pop(i-k)
            names.pop(i-k)
            labels.pop(i-k)
            k += 1

    # Selección de las densidades de estados proyectadas deseadas.
    density = np.zeros((len(energy), len(conds)))
    for i in range(len(conds)):
        density[:,i] = sum(dos_data[1,conds[i]])

    #%% Creación del archivo de salida, en caso de que se haya seleccionado
    #   dicha opción %%
       
    if export_dos and len(conds) != 0:
        
        # Matriz de datos.
        file_data = np.zeros((len(energy), len(conds) + 1))
        file_data[:,0] = energy   
        file_data[:,1:] = density
        # Nombre del fichero.
        file_name = '#'.join(labels).replace('\n', '')
        file_name = file_name.replace('#', '')
        file_name = dos_prefix + '-' + file_name
        file_name = file_name.replace('.dat', '').replace('txt', '') + '.txt'
        # Títulos de las columnas.
        titles = '#'.join(labels).replace('\n', '').split('#')
        # Determinación de los espacios que separan cada columna
        # (y ajuste en caso de que algún título sea más largo de la cuenta).
        num_blanks = 20  # espacios de separación en cada columna
        extra_blanks = 0  # espacios de corrección
        for i in range(len(titles)):
            # Si hay suficiente espacio hasta la siguiente columna,
            # no hay corrección.
            if (num_blanks - len(titles[i]) - extra_blanks) >= 2:
                blank_spaces = ' '*(num_blanks - len(titles[i]) - extra_blanks)
                extra_blanks = 0
            # Si no hay suficiente espacio hasta la siguiente columna,
            # se desplaza un poco el título de dicha siguiente columna.
            else:  
                 blank_spaces = '  '
                 extra_blanks += len(titles[i]) - num_blanks + 2 
            titles[i] = titles[i] + blank_spaces  # títulos de las columnas               
        # Espacios para alinear los títulos de las columnas.
        num_blanks_energy = num_blanks - 5 - len(energy_units)
        num_blanks_density = num_blanks + 2 - len(density_units)
        num_blanks_delimiter = num_blanks - 9
        
        # Título final.
        title = ('E (' + energy_units + ')' + ' '*num_blanks_energy
                 + 'pDOS (' + density_units + ')\n'
                 + ' '*num_blanks_density + ''.join(titles))
        # Exportación de los datos a un fichero de texto.
        np.savetxt(file_name, file_data, fmt='%1.3e', header=title,
                   delimiter=' '*num_blanks_delimiter)
        
        print('The file ' + file_name + ' has been generated.')
 
#%% Gráficas del modo de densidad de estados (y de ambos modos a la vez) %%

if dos_mode:

    if len(conds) == 0:
        raise ValueError('There is no density of states meeting the requirements.')

    # Límites en la energía y la densidad de estados.
    original_energy_limits = copy.copy(energy_limits)
    original_density_limits = copy.copy(density_limits)
    energy_limits = replace_limits(energy_limits, energy)
    # Límites en la energía para la gráfica ampliada.
    energy_center, energy_amplitude = tuple(energy_zoom_params.values())
    energy_zoom_limits = [energy_center - energy_amplitude,
                          energy_center + energy_amplitude]

    # Si también hay gráfico de bandas, cálculo de los límites en la energía.
    if bands_mode:
        # Límites en la energía.
        bands_energy_limits = copy.copy(original_energy_limits)
        bands_energy_limits = replace_limits(bands_energy_limits, energy_k)
        # Rango de energías que determina los límites en dicha variable.
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

    # Añadido de un pequeño margen a los límites en la energía.
    energy_limits = add_margin_to_limits(energy_limits, original_energy_limits,
                                         width=1/25)

    # Densidades de estados usadas para determinar los límites en esta variable.
    dens_comb = np.array([])
    cond = (energy >= energy_limits[0]) & (energy <= energy_limits[1])
    for i in range(density.shape[1]):
        dens_comb = np.concatenate((dens_comb, density[cond,i]))
    if total_dos:
        dens_comb = np.concatenate((dens_comb, total_density[cond]))
    dens_comb = dens_comb[dens_comb != 0]
        
    # Algoritmo para la detección automática del límite superior de la densidad
    # de estados. Calcula la función percentil para un rango de índices
    # percentiles y detecta en qué indice su derivada aumenta considerablemente,
    # usando ese índice para calcular el límite deseado.
    if density_limits[1] == 'auto':
        mad = median_absolute_deviation(dens_comb)
        indices = np.linspace(92, 100, int(1E3))  # índices percentiles
        percentiles = np.percentile(dens_comb, indices) / mad
        derivative = np.diff(percentiles) / np.diff(indices)
        derivative_non0 = derivative[derivative != 0]
        reference = np.median(derivative_non0[:int(0.5*len(derivative_non0))])
        # Condición para obtener el índice percentil buscado.
        cond = derivative > 50*reference
        if cond.sum() != 0:
            limit_index = indices[1:][cond].min()
        else:
            limit_index = indices[-1]
        # Límite obtenido con el índice percentil límite.
        cond = dens_comb < np.percentile(dens_comb, limit_index)
        margin = (dens_comb[cond].max() - dens_comb[cond].min()) / 10
        dens_lim_perc = np.percentile(dens_comb, limit_index) + margin + 1.5*mad
#        percentile_diagnostic_plots(indices, percentiles, derivative,
#                                    limit_index, reference, 50)
        # Límite obtenido con el máximo de la densidad de estados más un margen.
        margin = (dens_comb.max() - dens_comb.min()) / 25
        dens_lim_norm = dens_comb.max() + margin
        # Mínimo límite de ambos.
        density_limits[1] = min(dens_lim_norm, dens_lim_perc)

    # Correcciones en los límites en la densidad de estados.
    density_limits = replace_limits(density_limits, dens_comb)
    density_limits = add_margin_to_limits(density_limits,
                                          original_density_limits, width=1/25)
    if original_density_limits[0] == '':
        density_limits[0] = max(0, density_limits[0])
    
   # Figuras. (f = 1 para la gráfica ampliada)
    for f in reversed(range(1 + int(energy_zoom))):
        
        if bands_mode:
            plt.figure(f+1, figsize=(figure_dimensions[0]*2,figure_dimensions[1]))
        else:
            plt.figure(f+1, figsize=figure_dimensions)
        plt.clf()
        
        if bands_mode:
            # Gráfica de bandas (izquierda).
            ax1 = plt.subplot(1,2,1)

            # Colores de los símbolos, dependiendo de si se conocen las
            # bandas ocupadas y las vacías.
            if num_occupied_bands == '':
                num_occupied_bands = num_bands
                color1, color2 = [bands_colors[0]]*2
            else:
                color1, color2 = bands_colors[1:]
  
            # Representación de las bandas ocupadas
            # (o todas si no se sabe cuántas están ocupadas).
            for i in range(0, num_occupied_bands):  # bucle para cada banda
                wavenumber_i = wavenumber[inds[i] : inds[i+1]]
                energy_i = energy_k[inds[i] : inds[i+1]]
                plt.plot(wavenumber_i, energy_i, '.-', color=color1,
                         ms=6*scale, lw=1*scale, alpha=0.3/scale, zorder=1/3)
            # Representación de las bandas vacías
            # (o ninguna más si no se sabe cuántas hay vacías).
            for i in range(num_occupied_bands, num_bands):  # bucle para cada banda
                wavenumber_i = wavenumber[inds[i] : inds[i+1]]
                energy_i = energy_k[inds[i] : inds[i+1]]
                plt.plot(wavenumber_i, energy_i, '.-', color=color2,
                         ms=6*scale, lw=1*scale, alpha=0.3/scale, zorder=1/3)       
            plt.margins(x=0)

            # Límites.
            if f == 1:
                plt.ylim(energy_zoom_limits)
            else:
                plt.ylim(energy_limits)
            xlims, ylims = plt.xlim(), plt.ylim()
            # Líneas verticales para separar los caminos.
            if k_names != '':
                for x in k_locs:
                    plt.vlines(x, ylims[0], ylims[1], color='black',
                               lw=0.8*frame_width, zorder=3/3)  
            # Líneas horizontales para las energías seleccionadas.
            for y in energy_lines:
                plt.hlines(y, xlims[0], xlims[1], color='black',
                           linestyle=(0,(6,6)), lw=0.8*frame_width, zorder=2/3)
            # Etiquetas del espacio recíproco.
            if k_names != []:
                plt.xticks(k_locs, k_names)
            else:
                plt.xticks([])   
            # Texto de los ejes y título.
            plt.xlabel(k_label, labelpad=6)
            plt.ylabel(energy_label, labelpad=6)
            plt.title(bands_title, fontweight='bold', pad=10)
            plt.tight_layout()

            # Gráfica de densidades de estado (derecha).
            ax2 = plt.subplot(1,2,2, sharey=ax1)

        # Representación de las densidades de estados.
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
            
        # Límites.
        plt.xlim(density_limits)
        if f == 1: # gráfica ampliada
            plt.ylim([energy_zoom_limits])
        else:
            plt.ylim(energy_limits)  
        xlims, ylims = plt.xlim(), plt.ylim()
        # Líneas horizontales para las energías seleccionadas.
        for y in energy_lines:
            plt.hlines(y, xlims[0], xlims[1], color='black', lw=0.8*frame_width,
                       linestyle=(0,(6,6)), zorder=(len(conds)+2)/3)
        plt.xlim(xlims)
        plt.ylim(ylims)
        # Ajustes en el eje vertical.
        if bands_mode:
            ax2.yaxis.tick_right()
#            plt.tick_params(left=True)
        else:
            plt.ylabel(energy_label, labelpad=6)
        # Texto de la gráfica.
        plt.xlabel(density_label, labelpad=6)
        plt.legend(ncol=1, markerfirst=False)
        plt.title(dos_title, fontweight='bold', pad=12)
        plt.tight_layout()
#        # Marcas.
#        ax2.yaxis.set_major_locator(plt.AutoLocator())
#        ax2.minorticks_on()
    
    # Creación de una imagen para cada figura (normal y ampliación).
    if save_image: 
        image_name = '-' + ''.join(labels).replace('\n', '')
        extension = ['','-z']  # extensión para la gráfica ampliada
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

#%% Gráfica del modo de bandas %%

elif bands_mode and not dos_mode:
    
    # Límites en la energía
    original_energy_limits = copy.copy(energy_limits)
    energy_limits = replace_limits(energy_limits, energy_k)
    energy_limits = add_margin_to_limits(energy_limits, original_energy_limits,
                                         width=1/25)    
    # Figura.
    fig = plt.figure(1, figsize=figure_dimensions), plt.clf()
    # Colores.
    if num_occupied_bands != '':
        color1, color2 = bands_colors[1:]
    else:
        num_occupied_bands = num_bands
        color1, color2 = [bands_colors[0]]*2

    # Representación de las bandas ocupadas.
    # (o todas si no se sabe cuántas están ocupadas)
    occupied_bands_maxs = []
    for i in range(0, num_occupied_bands):  # bucle para cada banda
        wavenumber_i = wavenumber[inds[i] : inds[i+1]]
        energy_i = energy_k[inds[i] : inds[i+1]]
        plt.plot(wavenumber_i, energy_i, '.-', color=color1, ms=6*scale,
                 lw=1*scale, alpha=0.3/scale, picker=1, zorder=1/3)
        occupied_bands_maxs += [max(energy_i)]
    # Representación de las bandas vacías.
    # (o ninguna más si no se sabe cuántas están vacías)
    if num_empty_bands != '':
        empty_bands_mins = [] 
        for i in range(num_occupied_bands, num_bands):  # bucle para cada banda
            wavenumber_i = wavenumber[inds[i] : inds[i+1]]
            energy_i = energy_k[inds[i] : inds[i+1]]
            plt.plot(wavenumber_i, energy_i, '.-', color=color2, ms=6*scale,
                     lw=1*scale, alpha=0.3/scale, picker=1, zorder=1/3)
            empty_bands_mins += [min(energy_i)]

        # Caso de distinguir entre bandas ocupadas y vacías.
        if num_empty_bands > 0 and num_occupied_bands > 0:
            # Cálculo del máximo de la banda de conducción y el mínimo de la
            # banda de valencia.
            cond_band_min = min(occupied_bands_maxs)
            val_band_max = max(empty_bands_mins)       
            cond_band_min_k = wavenumber[energy_k == cond_band_min][0]
            val_band_max_k = wavenumber[energy_k == val_band_max][0]
            cond_band_min_point = find_wavevector(cond_band_min_k, k_locs, k_coords)
            val_band_max_point = find_wavevector(val_band_max_k, k_locs, k_coords) 
            # Texto en la terminal.
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

    # Límites de los ejes.
    plt.margins(x=0)
    plt.ylim([energy_limits[0], energy_limits[1]])
    xlims, ylims = plt.xlim(), plt.ylim()
    # Líneas verticales para separar los caminos.
    if k_names != []:
        for x in k_locs:
            plt.vlines(x, ylims[0], ylims[1], color='black',
                       lw=0.8*frame_width, zorder=3/3)
        plt.xticks(k_locs, k_names)
    # Líneas horizontales para las energías seleccionadas.
    for y in energy_lines:
        plt.hlines(y, xlims[0], xlims[1], color='black', lw=0.8*frame_width,
                   linestyle=(0,(6,6)), zorder=2/3)     
    # Textos de la figura.
    plt.xlabel(k_label, labelpad=6)
    plt.ylabel(energy_label, labelpad=6)
    plt.title(bands_title, fontweight='bold', pad=12)
    plt.tight_layout()

    # Creación de una imagen de la figura.
    if save_image:
        image_name = bands_file + '.' + image_format
        image_name = image_name.replace('.dat', '').replace('.txt', '')
        plt.savefig(image_name, format=image_format, dpi=240)
        print('Figure 1 has been saved as ' + image_name + '.')

    # Modo interactivo.
    fig[0].canvas.mpl_connect('pick_event', click)

    plt.show()