import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import csv
import json


# csv header
fieldnames = ['n_nodes', 'degree', 'n_roots', 'max_edges', 'min_edges', 'n_evidence', 'time', 'degrees_occurred', 'result']

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_NAME_OUTPUTS = '/results'
FILE_NAME_METHOD_NUMBER = '/method_{0}'
FILE_NAME_RESULTS_CSV = '/run_{0}.csv'
FILE_RESULTS = '/run_*.csv'
FILE_NAME_QUERY_CSV = '/query_{0}'

def set_fonts():
    SMALL_SIZE =12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def create_lineplots(data, query, Heuristics):
    # Create a color palette
    palette = plt.get_cmap('Set1')
    # Change the style of plot
    plt.style.use('seaborn')
    set_fonts()

    res = {} 
    for index, i in enumerate(Heuristics): 
        i = i.value

        res.update({ 'x' + i: [], 'y' + i: [] })
        for key in data[i]:
            if key == '45' or key == 45: continue
            res['x' + i].append(str(key))
            res['y' + i].append(np.average(data[i][key]['values']))

        plt.plot(res['x' + i], res['y' + i], label = "%s" % i.lower(), marker='o', ms=3, color=palette(index), linewidth=1, alpha=0.9) #, color=color[int(i)-1])

        
    plt.xlabel('# of variables (nodes)')
    # Set the y axis label of the current axis.
    plt.ylabel('Time')
    # Set a title of the current axes.
    plt.title('%s Query' % query, fontsize=16)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

def create_histchart(data, query, Heuristics, n_node):
    # Create a color palette
    palette = plt.get_cmap('Set1')
    # Change the style of plot
    plt.style.use('seaborn')
    set_fonts()

    res = {} 
    bins= set({})
    for index, i in enumerate(list(reversed(Heuristics))): 
        i = i.value

        res.update({ 'x' + i: [], 'y' + i: [] })
        res['x' + i] = data[i][n_node]['degrees']

        bins = bins | set(np.unique(res['x' + i]))
        plt.hist(x=res['x' + i], bins=np.unique(res['x' + i]), label = "%s" % i.lower(), facecolor=palette(index), alpha=0.35)

    plt.xticks(list(bins))
    # Set the x axis label of the current axis.
    plt.xlabel('Tracked degrees of each order (throughout the execution)')
    # Set the y axis label of the current axis.
    plt.ylabel('# of occurrences (in all 10 runs)')
    # Set a title of the current axes.
    plt.title('%s Query in a %s-node BN' % (query, n_node), fontsize=16)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

def create_barchart(data, query, Heuristics):
    # width of the bars
    barWidth = 0.3
    # Create a color palette
    palette = plt.get_cmap('Set1')
    # Change the style of plot
    plt.style.use('seaborn')
    set_fonts()

    res = {} 
    xtick = set({})
    for index, i in enumerate(Heuristics): 
        i = i.value

        res.update({ 'x' + i: [], 'y' + i: [] })
        for key in data[i]:
            if key == 35 or key == 45: continue
            res['x' + i].append(np.max(data[i][key]['degree']))
            res['y' + i].append(np.max(data[i][key]['degrees']))

        position = [x + index*barWidth for x in res['x' + i]]
        xtick = xtick | set(res['x' + i])
        plt.bar(x=position, height=res['y' + i], data=res['x' + i], label = "%s" % i.lower(), color=palette(index), edgecolor='black', align='center', capsize=2, width=barWidth) #, color=color[int(i)-1])

    plt.xticks(list(xtick))
    plt.xlabel('Width of BNs')
    # Set the y axis label of the current axis.
    plt.ylabel('Max Degree occurred throught the runs')
    # Set a title of the current axes.
    plt.title('%s Query' % query)
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

def save_data(query, method, run_number, results):
    foldername = FILE_PATH + FILE_NAME_OUTPUTS + FILE_NAME_QUERY_CSV.format(query) + FILE_NAME_METHOD_NUMBER.format(method)
    filename = foldername + FILE_NAME_RESULTS_CSV.format(run_number)

    check_file_existance(foldername, filename, fieldnames)

    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows([results])

def combine_results(query, Heuristics):
    csvfiles = {}
    data = {}
    
    for heuristic in Heuristics:
        file_name = FILE_PATH + FILE_NAME_OUTPUTS +  FILE_NAME_QUERY_CSV.format(query) + FILE_NAME_METHOD_NUMBER.format(heuristic.value) + FILE_RESULTS
        csvfiles[heuristic.value] = glob.glob(file_name)

        data.update({
            heuristic.value: {}
        })
        
        for files in csvfiles[heuristic.value]:
            df = pd.read_csv(files)
            for _, row in df.iterrows():
                list = json.loads(row['degrees_occurred'])
                if row['n_nodes'] in data[heuristic.value] and 'values' in data[heuristic.value][row['n_nodes']]:
                    data[heuristic.value].update({
                        row['n_nodes']: {
                            'values': (data[heuristic.value][row['n_nodes']]['values']) + [float(row['time'])],
                            'degrees': (data[heuristic.value][row['n_nodes']]['degrees']) + list,
                            'degree': (data[heuristic.value][row['n_nodes']]['values']) + [int(row['degree'])]
                        }
                    })
                else:
                    data[heuristic.value].update({
                        row['n_nodes']: {
                            'values': [float(row['time'])],
                            'degrees': list,
                            'degree': [int(row['degree'])]
                        }
                    })

    return data
