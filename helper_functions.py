import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import csv


# csv header
fieldnames = ['n_nodes', 'degree', 'n_roots', 'max_edges', 'min_edges', 'n_evidence', 'time', 'degrees_occurred', 'result']

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_NAME_OUTPUTS = '/results'
FILE_NAME_METHOD_NUMBER = '/method_{0}'
FILE_NAME_RESULTS_CSV = '/run_{0}.csv'
FILE_RESULTS = '/run_*.csv'
FILE_NAME_QUERY_CSV = '/query_{0}'


def create_lineplots(data, query, Heuristics):
    # Create a color palette
    palette = plt.get_cmap('Set1')
    # Change the style of plot
    plt.style.use('seaborn')

    res = {} 
    for index, i in enumerate(Heuristics): 
        i = i.value

        res.update({ 'x' + i: [], 'y' + i: [] })
        for key in data[i]:
            print('d', i, key, data[i][key]['values'])
            if key == '45' or key == 45: continue
            res['x' + i].append(str(key))
            res['y' + i].append(np.average(data[i][key]['values']))

        print('res', res)

        plt.plot(res['x' + i], res['y' + i], label = "%s" % i.lower(), marker='o', ms=3, color=palette(index), linewidth=1, alpha=0.9) #, color=color[int(i)-1])

        
    plt.xlabel('Number of variables (nodes)')
    # Set the y axis label of the current axis.
    plt.ylabel('Time')
    # Set a title of the current axes.
    plt.title('%s Query' % query)
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

    res = {} 
    for index, i in enumerate(Heuristics): 
        i = i.value

        res.update({ 'x' + i: [], 'y' + i: [] })
        for key in data[i]:
            print('d', i, key, data[i][key]['values'])
            if key == '45' or key == 45: continue
            res['x' + i].append(str(key))
            res['y' + i].append(np.average(data[i][key]['values']))

        print('res', res)
        # position = [x + index*barWidth for x in np.arange(len(res['x' + i]))]

        plt.bar(res['x' + i], res['y' + i], label = "%s" % i.lower(), color=palette(index), edgecolor='black', align='edge', capsize=2, width=barWidth) #, color=color[int(i)-1])

        
    plt.xlabel('Number of variables (nodes)')
    # Set the y axis label of the current axis.
    plt.ylabel('Time')
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
        print('file', file_name)

        data.update({
            heuristic.value: {}
        })
        
        print('method_number', csvfiles, heuristic.value, query)
        for files in csvfiles[heuristic.value]:
            df = pd.read_csv(files)
            # print(df)
            # data[heuristic.value].update(df)
            for _, row in df.iterrows():
                if row['n_nodes'] in data[heuristic.value] and 'values' in data[heuristic.value][row['n_nodes']]:
                    data[heuristic.value].update({
                        row['n_nodes']: {
                            'values': (data[heuristic.value][row['n_nodes']]['values']) + [float(row['time'])], # max fitness values
                        }
                    })
                else:
                    data[heuristic.value].update({
                        row['n_nodes']: {
                            'values': [float(row['time'])], # max fitness values
                        }
                    })

    return data
