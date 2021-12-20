import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import csv


# csv header
fieldnames = ['n_nodes', 'degree', 'n_roots', 'max_edges', 'min_edges', 'n_evidence', 'time', 'result']

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_NAME_OUTPUTS = '/results'
FILE_NAME_METHOD_NUMBER = '/method_{0}'
FILE_NAME_RESULTS_CSV = '/run_{0}.csv'
FILE_RESULTS = '/run_2.csv'


def create_lineplots(data, methods, heuristics, x_val, y_val):
    # Create a color palette
    palette = plt.get_cmap('Set1')
    # Change the style of plot
    plt.style.use('seaborn')

    res = {}

    for index, i in enumerate(heuristics): 
        i = i.value
        # print(i, index)
        # avg.update({ 'x' + i: [], 'y' + i: [] })
        res.update({ 'x' + i: [], 'y' + i: [] })
        print(data)
        for d in data[i].values():
            print('!!!!!!!', d, 'd')
            res['x' + i].append(d[x_val])
            res['y' + i].append(d[y_val])

        plt.plot(res['x' + i], res['y' + i], label = "%s Heuristic" % i, marker='o', ms=3, color=palette(index), linewidth=1, alpha=0.9) #, color=color[int(i)-1])

        
    plt.xlabel(x_val)
    # Set the y axis label of the current axis.
    plt.ylabel(y_val)
    # Set a title of the current axes.
    plt.title('%s Query' % methods)
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
    print(results)

    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows([results])

def combine_results(heuristics, methods=''):
    csvfiles = {}
    data = {}
    
    for heuristic in heuristics:
        heuristic = heuristic.value
        file_name = FILE_PATH + FILE_NAME_OUTPUTS + FILE_NAME_METHOD_NUMBER.format(heuristic) + FILE_RESULTS
        print('file_name', file_name)
        csvfiles[heuristic] = glob.glob(file_name)

        data.update({
            heuristic: {}
        })
        
        print('method_number', csvfiles, heuristic)
        for files in csvfiles[heuristic]:
            df = pd.read_csv(files).T.to_dict()
            print(df)
            data[heuristic].update(df)

    return data
