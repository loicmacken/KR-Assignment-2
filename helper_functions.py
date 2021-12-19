import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# csv header
fieldnames = ['degree', 'n_roots', 'max_edges', 'min_edges', 'time']

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_NAME_OUTPUTS = '/results'
FILE_NAME_METHOD_NUMBER = '/method_{0}'
FILE_NAME_RESULTS_CSV = '/run_{0}.csv'

def create_lineplot_per_enemy(data, method_numbers):
    avg = {}
    max = {}
    std_max = {}
    std_avg = {}

    for i in method_numbers: 
        avg.update({ 'x' + i: [], 'y' + i: [] })
        max.update({ 'x' + i: [], 'y' + i: [] })
        std_max.update({ i: [] })
        std_avg.update({ i: [] })

        for gen in data[i]:
            # method i average
            avg['x' + i].append(gen)
            avg['y' + i].append(np.average(data[i][gen]['avg_values']))
            # method i maximum
            max['x' + i].append(gen)
            max['y' + i].append(np.average(data[i][gen]['max_values']))
            # method i std for maximum
            std_max[i].append(np.std(data[i][gen]['max_values']))
            # method i std for mean
            std_avg[i].append(np.std(data[i][gen]['avg_values']))

    plt.xlabel('Time')
    # Set the y axis label of the current axis.
    plt.ylabel('Number of variables')
    # Set a title of the current axes.
    plt.title('Case %s' % 1)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

def add_fieldnames(filename, fieldnames): 
    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def check_file_existance(foldername, filename, fieldnames): 
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        add_fieldnames(filename, fieldnames)
    elif not os.path.exists(filename):
        add_fieldnames(filename, fieldnames)

def save_data(method_number, run_number, results):
    foldername = FILE_PATH + FILE_NAME_OUTPUTS + FILE_NAME_METHOD_NUMBER.format(method_number)
    filename = foldername + FILE_NAME_RESULTS_CSV.format(run_number)

    check_file_existance(foldername, filename, fieldnames)

    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(results)
