import os
import sys
import copy
# add to the path the source files
sys.path.append(os.path.dirname(os.getcwd()))

from src.dataset.counting_algorithm import subgraph_counting, subgraph_counting_all, subgraph_listing
from src.ppgn.ppgn import PPGNexpl

import seml
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from networkx import Graph
import dgl
import networkx as nx
from statistics import mean, stdev
from math import sqrt
from scipy.interpolate import interp1d
import torch
import json
import torch_geometric
import scipy

budget_perc_float = [0.01, 0.05,0.1,0.25]

from math import log10, floor
def round_2(x):
    num = round(x, 1-int(floor(log10(abs(x)))))
    if num < 0.001:
        mant = int(floor(log10(abs(num))))
        base = num*10**(-mant)
        base = round(base, 1)
        return f'{base}e{mant}'
    else:
        return num

def generate_gnn_input(graph: nx.Graph, device)->torch_geometric.data.Data:
    """Creates from a networkx graph a Data instance, which is the input a a pytorch geometric model."""
    num_edges = graph.number_of_edges()
    x = torch.ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
    edge_index = torch.empty(2, 2*num_edges, dtype=torch.long)
    for i, edge in enumerate(graph.edges()):
        edge_index[0,i] = edge[0]
        edge_index[1,i] = edge[1]
        edge_index[0, i+num_edges] = edge[1]
        edge_index[1, i+num_edges] = edge[0]
    return torch_geometric.data.Data(x=x, edge_index=edge_index).to(device)

def adversarial_error(results_exp, exp_info):
    budgets = results_exp["config.budgets"].iloc[0]
    for i, budget in enumerate(budgets):
        results_exp_temp = results_exp
        results_exp_temp['Adversarial'] = results_exp_temp['result.adversarial_error'].apply(lambda l: l[i])
        results_exp_temp['Cross adversarial'] = results_exp_temp['result.cross_adversarial_errors_average'].apply(lambda l: l[i])
        results_exp_temp = pd.melt(frame=results_exp_temp, id_vars=["result.graph", 'Subgraph', 'result.seed', 'result.architecture'], value_vars=['Test', 'Adversarial', 'Cross Test', 'Cross adversarial'], var_name='Dataset', value_name=exp_info['loss_name'])
        fig, axes = plt.subplots(1, len(exp_info['subgraphs']), constrained_layout=True)
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Aversarial robsutness {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel(exp_info['loss_name'])

        for i, subgraph in enumerate(exp_info['subgraphs']):
            data = results_exp_temp[results_exp_temp['Subgraph']==subgraph]
            
            sns.boxplot(data=data, x="Subgraph", y=exp_info['loss_name'], hue='Dataset', showfliers = False, ax=axes[i], )
            #sns.despine()
            axes[i].set(xlabel=None)
            axes[i].set(ylabel=None)
            if i != len(exp_info['subgraphs'])-1:
                axes[i].get_legend().remove()
            else:
                pos = axes[i].get_position()
                axes[i].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()

def sign_adversarial_error(results_exp, exp_info, b = None):
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    for i, budget in enumerate(budgets):
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        if b is not None and b != i:
            continue
        results_exp_temp = results_exp
        results_exp_temp['Adversarial'] = results_exp_temp['result.sign_adversarial_errors'].apply(lambda l: l[i])
        # results_exp_temp['Cross adversarial'] = results_exp_temp['result.cross_sign_adversarial_errors'].apply(lambda l: mean(l[i]))
        results_exp_temp = pd.melt(frame=results_exp_temp, id_vars=["result.graph", 'Subgraph', 'result.seed', 'result.architecture'], value_vars=['Adversarial'], var_name='Dataset', value_name=exp_info['loss_name'])
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        fig.set_figwidth(3+len(exp_info['subgraphs'])*3)
        fig.set_figheight(5)
        fig.set_dpi(400)
        fig.suptitle(f"Aversarial robsutness {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel(f"{exp_info['loss_name']} with sign")

        for j, subgraph in enumerate(real_subgraphs):
            data = results_exp_temp[results_exp_temp['Subgraph']==subgraph]
            
            sns.violinplot(data=data, x="Subgraph", y=exp_info['loss_name'], hue='Dataset', showfliers = False, ax=axes[j], )
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            else:
                pos = axes[j].get_position()
                axes[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()

def subgraph_count(results_exp_old, exp_info, b=None):
    
    results_exp = copy.deepcopy(results_exp_old)
    if len(results_exp)<5:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    results_exp = results_exp.rename(columns={'Test Count': 'Test Graphs'})
    # NUMBER OF SUBSTRUCTURES
    tot_budget = 0
    for i , budget in enumerate(budgets):
        p_vals = {}
        for subgraph in exp_info['subgraphs']:
            p_vals[subgraph] = None
        p = '& ' + exp_info['budget_perc'][i]
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) >= 25]
        if len(real_subgraphs) == 0:
            continue
        tot_budget += budget
        if b is not None and b != i:
            continue
        results_exp_temp = results_exp
        results_exp_temp['Adversarial Graphs'] = results_exp_temp['result.adversarial_count'].apply(lambda l: l[i])
        results_exp_temp_2 = copy.deepcopy(results_exp_temp)
        results_exp_temp = pd.melt(frame=results_exp_temp, id_vars=["result.graph", 'Subgraph', 'result.seed', 'result.architecture'], value_vars=['Test Graphs', 'Adversarial Graphs'], var_name='Dataset', value_name='Count')
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(3+len(exp_info['subgraphs'])*3)
        fig.set_figheight(5)
        fig.set_dpi(400)
        fig.suptitle(f"Subgraph-Count of adversarial examples for {exp_info['arch']}, {exp_info['strategy']}, $\Delta=$ {exp_info['budget_perc'][i]}", fontsize = 30)
        fig.supxlabel('Pattern')
        fig.supylabel('Count')
        for j, subgraph in enumerate(real_subgraphs):
            data = results_exp_temp[results_exp_temp['Subgraph']==subgraph]
            sns.violinplot(data=data, x="Subgraph", y='Count', hue='Dataset', showfliers = False, ax=axes[j] )
            #sns.despine()
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            else:
                pos = axes[j].get_position()
                axes[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
            # statistical test
            data_2 = results_exp_temp_2[results_exp_temp_2['Subgraph']==subgraph]
            stat, p_val = scipy.stats.ttest_ind(data_2['Adversarial Graphs'], data_2['Test Graphs'], equal_var=False)
            p_vals[subgraph] = p_val
        for subgraph in exp_info['subgraphs']:
            if p_vals[subgraph] is not None:
                p = p + f' & {round_2(p_vals[subgraph])}'
            else:
                p = p + f' & NaN'
        p = p + '\\\\'
        print(p)
        # plt.show()
        # if i == 2:
        #     fig.savefig(f"figs/{exp_info['arch']}_{exp_info['strategy']}_{budget}_count.pdf")
            

def edge_count(results_exp_old, exp_info, b = None):
    results_exp = copy.deepcopy(results_exp_old)
    if len(results_exp)==0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    tot_budget = 0
    graph_ids = results_exp["result.graph"].sort_values().unique()
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    results_exp["Test edges"] = np.nan
    results_exp["Adversarial edges all"] = np.nan
    results_exp["Adversarial edges all"] = results_exp["Adversarial edges all"].astype('object')
    graphs, _ = dgl.load_graphs(dataset_path)

    test_edges = {}
    for graph_id in graph_ids:
        graph = Graph(dgl.to_networkx(graphs[graph_id]))
        test_edges[graph_id] = graph.number_of_edges() # equal for all the seeds

    for index, line in results_exp.iterrows():
        graph_id = line["result.graph"]
        subgraph = line["Subgraph"]
        seed = line["config.seed"]
        adv_edges = []
        for i, budget in enumerate(budgets):
            adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch_file']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
            adv_edges.append(adv_graph.number_of_edges())
        # assign the values
        results_exp.at[index, "Test Graphs"] = test_edges[graph_id]
        results_exp.at[index, "Adversarial edges all"] = adv_edges

    for i, budget in enumerate(budgets):
        p_vals = {}
        for subgraph in exp_info['subgraphs']:
            p_vals[subgraph] = None
        p = '& ' + exp_info['budget_perc'][i]
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) >= 25]
        if len(real_subgraphs) == 0:
            continue
        tot_budget += budget
        if b is not None and b != i:
            continue
        results_exp_temp = results_exp
        results_exp_temp['Adversarial Graphs'] = results_exp_temp['Adversarial edges all'].apply(lambda l: l[i])
        results_exp_temp_2 = copy.deepcopy(results_exp_temp)
        results_exp_temp = pd.melt(frame=results_exp_temp, id_vars=["result.graph", 'Subgraph', 'result.seed', 'result.architecture'], value_vars=['Test Graphs', 'Adversarial Graphs'], var_name='Dataset', value_name='$\#$ edges')
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(3+len(exp_info['subgraphs'])*3)
        fig.set_figheight(5)
        fig.set_dpi(400)
        fig.suptitle(f"Number of edges of adversarial examples for {exp_info['arch']}, {exp_info['strategy']}, $\Delta=$ {exp_info['budget_perc'][i]}", fontsize = 30)
        fig.supxlabel('Pattern')
        fig.supylabel('$\#$ edges')
        

        for j, subgraph in enumerate(real_subgraphs):
            data = results_exp_temp[results_exp_temp['Subgraph']==subgraph]
            sns.violinplot(data=data, x="Subgraph", y='$\#$ edges', hue='Dataset', showfliers = False, ax=axes[j], )
            #sns.despine()
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            # if b is not None:
            #     axes[i].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            else:
                pos = axes[j].get_position()
                axes[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        # plt.show()
        # if i == 3:
        #     fig.savefig(f"figs/{exp_info['arch']}_{exp_info['strategy']}_{budget}_edges.pdf")
        # statistical test
            data_2 = results_exp_temp_2[results_exp_temp_2['Subgraph']==subgraph]
            stat, p_val = scipy.stats.ttest_ind(data_2['Adversarial Graphs'], data_2['Test Graphs'], equal_var=False)
            p_vals[subgraph] = p_val
        for subgraph in exp_info['subgraphs']:
            if p_vals[subgraph] is not None:
                p = p + f' & {round_2(p_vals[subgraph])}'
            else:
                p = p + f' & NaN'
        p = p + '\\\\'
        print(p)
        
def degree_distribution(results_exp, exp_info,b):
    def dist(graph):
        degrees = np.zeros(graph.number_of_nodes())
        for n, d in graph.degree():
            degrees[d] += 1
        # for i in range(len(degrees)): # cumulative ditribution
        #     if i == 0:
        #         continue
        #     degrees[i] = degrees[i-1] + degrees[i]
        return degrees/graph.number_of_nodes()
    # number of nodes having a specific degree
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Degree distribution of adversarial examples {exp_info['arch']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Degree')
        fig.supylabel('% Nodes')

        for j, subgraph in enumerate(real_subgraphs):
            adv_deg = None
            test_deg = None
            data = results_exp[results_exp['Subgraph']==subgraph]
            for index, line in data.iterrows():
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                if adv_deg is None:
                    adv_deg = dist(adv_graph)
                else:
                    adv_deg += dist(adv_graph)
                if test_deg is None:
                    test_deg = dist(graph)
                else:
                    test_deg += dist(graph)
            adv_deg /= len(data)
            test_deg /= len(data) 
            first_non_zero = len(test_deg)-1
            while(adv_deg[first_non_zero] == 0 and test_deg[first_non_zero] == 0):
                first_non_zero  -= 1
            test_deg = test_deg[:min(first_non_zero+3, len(test_deg))]
            adv_deg = adv_deg[:min(first_non_zero+3, len(adv_deg))]
            # plot
            x = list(range(len(test_deg)))
            sns.lineplot(x = x, y = test_deg, label='Test', ax = axes[j], linewidth=0.5)
            axes[j].fill_between(x=x, y1=[0 for _ in x], y2=test_deg, alpha=0.5,zorder=10)
            x = list(range(len(adv_deg)))
            sns.lineplot(x = x, y = adv_deg, label='Adversarial', ax=axes[j], linewidth=0.5)
            axes[j].fill_between(x=x, y1=[0 for _ in x], y2=adv_deg, alpha=0.5,zorder=0)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            else:
                pos = axes[j].get_position()
                axes[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()
    

            
def best_adv_exaples(results_exp, exp_info, b=None):
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        results_exp_temp = results_exp
        results_exp_temp['Adversarial'] = results_exp_temp['result.sign_adversarial_errors'].apply(lambda l: l[i])
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Best adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        
        for j, subgraph in enumerate(real_subgraphs):
            adv_deg = None
            test_deg = None
            data = results_exp[results_exp['Subgraph']==subgraph]
            line = data.sort_values(by='Adversarial', ascending=False).iloc[0]
            graph_id = line["result.graph"]
            seed = line["config.seed"]
            adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
            graph = Graph(dgl.to_networkx(graphs[graph_id]))
            
            #define the new graph
            graph_plot = Graph()
            edges = graph.edges()
            adv_edges = adv_graph.edges()
            for e in edges:
                if e in adv_edges:
                    graph_plot.add_edge(*e, color='black')
                else:
                    graph_plot.add_edge(*e, color='r')
            for e in adv_edges:
                if e not in edges:
                    graph_plot.add_edge(*e, color='g')
            colors = [graph_plot[u][v]['color'] for u,v in graph_plot.edges()]
            nx.draw(graph_plot, ax = axes[j], edge_color = colors, node_size=100)
            axes[j].annotate(f'Adversarial error: {line["Adversarial"]:.5f}\n Test error: {line["Test"]:.5f}', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            axes[j].set_title(subgraph)
   
        plt.show()
    
def count_other_substructures(results_exp, exp_info, b=None):
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    tot_budget = 0
    graph_ids = results_exp["result.graph"].sort_values().unique()
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), tight_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Substructures count of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel('Adv Count - Test Count')
        # distribution
        fig2, axes2 = plt.subplots(1, len(real_subgraphs), tight_layout=True)
        if len(real_subgraphs) == 1:
            axes2 = [axes]
        fig2.set_figwidth(16)
        fig2.set_figheight(4.5)
        fig2.set_dpi(400)
        fig2.suptitle(f"Substructures count of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig2.supxlabel('Subgraph')
        fig2.supylabel('Adv Count - Test Count')

        for j, subgraph in enumerate(real_subgraphs):
            diff_counts = {}
            avg_diff_counts = {}
            data = results_exp[results_exp['Subgraph']==subgraph]
            for index, line in data.iterrows():
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_count = subgraph_counting_all(adv_graph)
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_count = subgraph_counting_all(graph)
                del adv_count['3-Star not ind.'] # always increases
                del test_count['3-Star not ind.']
                if diff_counts == {}:
                    for s in adv_count.keys():
                        avg_diff_counts[s] = adv_count[s] - test_count[s]
                        diff_counts[s] = [adv_count[s] - test_count[s]]
                else:
                    for s in adv_count.keys():
                        avg_diff_counts[s] += adv_count[s] - test_count[s]
                        diff_counts[s].append(adv_count[s] - test_count[s])
            # average
            df = pd.DataFrame(columns=['Subgraph', 'Count Difference'], index = list(range(len(list(avg_diff_counts.keys())))))
            df2 = pd.DataFrame(columns=['x', 'Subgraph', 'Count Difference'], index = list(range(len(list(diff_counts.keys()))*len(data))))
            for k, s in enumerate(list(avg_diff_counts.keys())):
                # df.loc[2*k] = [s, 'Adversarial', avg_adv_counts[s] / len(data)]
                # df.loc[2*k+1] = [s, 'Test', avg_test_counts[s] / len(data)]
                df.loc[k] = [s, (avg_diff_counts[s])/len(data)]
                df2.loc[k*len(data): (k+1)*len(data)-1, 'Subgraph'] = s
                df2.loc[k*len(data): (k+1)*len(data)-1, 'Count Difference'] = diff_counts[s]
            df2['x'] = subgraph

            # sns.barplot(data=df, x="Subgraph", y='Count', hue='Dataset', ax=axes[j]) 
            sns.barplot(x=df['Subgraph'], y=df['Count Difference'], ax=axes[j])
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            axes[j].set_xticklabels(axes[j].get_xticklabels(), rotation=45, horizontalalignment='right')

            sns.boxplot(data=df2, x='x', y='Count Difference', hue='Subgraph', ax=axes2[j])
            axes2[j].set(xlabel=None)
            axes2[j].set(ylabel=None)
            axes2[j].tick_params(bottom=False)
            axes2[j].set_title(subgraph)
            axes2[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes2[j].get_legend().remove()
            else:
                pos = axes2[j].get_position()
                axes2[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()

def count_other_substructures_sign(results_exp, exp_info, b=None):
    # separate the plots for graphs that have a overestimate and the ones that underestimate
    if len(results_exp) == 0:
        return
        
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    tot_budget = 0
    graph_ids = results_exp["result.graph"].sort_values().unique()
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        results_exp['Adversarial'] = results_exp['result.sign_adversarial_errors'].apply(lambda l: l[i])
        fig1, axes1 = plt.subplots(1, len(real_subgraphs), tight_layout=True)
        if len(real_subgraphs) == 1:
            axes1 = [axes1]
        fig1.set_figwidth(16)
        fig1.set_figheight(4.5)
        fig1.set_dpi(400)
        fig1.suptitle(f"Substructures count avg of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig1.supxlabel('Subgraph')
        fig1.supylabel('Adv Count - Test Count')

        # distribution
        fig2, axes2 = plt.subplots(1, len(real_subgraphs), tight_layout=True)
        if len(real_subgraphs) == 1:
            axes2 = [axes2]
        fig2.set_figwidth(16)
        fig2.set_figheight(4.5)
        fig2.set_dpi(400)
        fig2.suptitle(f"Substructures count of overestimating adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig2.supxlabel('Subgraph')
        fig2.supylabel('Adv Count - Test Count')

        for j, subgraph in enumerate(real_subgraphs):
            diff_counts_pos = {}
            diff_counts_neg = {}
            avg_diff_counts_pos = {}
            avg_diff_counts_neg = {}
            num_pos = 0
            num_neg = 0
            data = results_exp[results_exp['Subgraph']==subgraph]
            for index, line in data.iterrows():
                sign_err = line['Adversarial']
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_count = subgraph_counting_all(adv_graph)
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_count = subgraph_counting_all(graph)
                del adv_count['3-Star not ind.'] # always increases
                del test_count['3-Star not ind.']
                if sign_err > 0:
                    num_pos += 1
                    if diff_counts_pos == {}:
                        for s in adv_count.keys():
                            avg_diff_counts_pos[s] = adv_count[s] - test_count[s]
                            diff_counts_pos[s] = [adv_count[s] - test_count[s]]
                    else:
                        for s in adv_count.keys():
                            avg_diff_counts_pos[s] += adv_count[s] - test_count[s]
                            diff_counts_pos[s].append(adv_count[s] - test_count[s])
                if sign_err < 0:
                    num_neg += 1
                    if diff_counts_neg == {}:
                        for s in adv_count.keys():
                            avg_diff_counts_neg[s] = adv_count[s] - test_count[s]
                            diff_counts_neg[s] = [adv_count[s] - test_count[s]]
                    else:
                        for s in adv_count.keys():
                            avg_diff_counts_neg[s] += adv_count[s] - test_count[s]
                            diff_counts_neg[s].append(adv_count[s] - test_count[s])
           
            df = pd.DataFrame(columns=['Subgraph', 'Count Difference', 'Error'], index = list(range(2*len(list(avg_diff_counts_pos.keys())))))
            for k, s in enumerate(list(avg_diff_counts_pos.keys())):
                if s != subgraph:
                    df.loc[k] = [s, (avg_diff_counts_pos[s])/num_pos, 'Overestimating']
            k_base = len(list(avg_diff_counts_pos.keys()))
            for k, s in enumerate(list(avg_diff_counts_neg.keys())):
                if s != subgraph:
                    df.loc[k_base + k] = [s, (avg_diff_counts_neg[s])/num_neg, 'Underestimating']

            df_dist = pd.DataFrame(columns=['Subgraph', 'Count Difference', 'Error'], index = list(range(len(list(diff_counts_pos.keys()))*num_pos + len(list(diff_counts_neg.keys()))*num_neg)))
            for k, s in enumerate(list(diff_counts_pos.keys())):
                if s != subgraph:
                    df_dist.loc[k*num_pos: (k+1)*num_pos-1, 'Subgraph'] = s
                    df_dist.loc[k*num_pos: (k+1)*num_pos-1, 'Count Difference'] = diff_counts_pos[s] 
                    df_dist.loc[k*num_pos: (k+1)*num_pos-1, 'Error'] = 'Overestimating'
            k_base = (k+1)*num_pos
            for k, s in enumerate(list(avg_diff_counts_neg.keys())):
                if s != subgraph:
                    df_dist.loc[k_base + k*num_neg: k_base + (k+1)*num_neg-1, 'Subgraph'] = s
                    df_dist.loc[k_base + k*num_neg: k_base + (k+1)*num_neg-1, 'Count Difference'] = diff_counts_neg[s] 
                    df_dist.loc[k_base + k*num_neg: k_base + (k+1)*num_neg-1, 'Error'] = 'Underestimating'
            
            sns.barplot(data=df, x='Subgraph', y='Count Difference', hue='Error', ax = axes1[j])
            axes1[j].annotate(f'# ↑: {num_pos}, # ↓: {num_neg}', xy = (0.5, 0.95), xycoords='axes fraction', horizontalalignment='center',)
            axes1[j].set(xlabel=None)
            axes1[j].set(ylabel=None)
            axes1[j].set_title(subgraph)
            axes1[j].set_xticklabels(axes1[j].get_xticklabels(), rotation=45, horizontalalignment='right')
            if j != len(real_subgraphs)-1:
                axes1[j].get_legend().remove()
            else:
                pos = axes1[j].get_position()
                axes1[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )

            
            sns.boxplot(data=df_dist, x='Subgraph', y='Count Difference', hue='Error', ax=axes2[j])
            axes2[j].set(xlabel=None)
            axes2[j].set(ylabel=None)
            axes2[j].tick_params(bottom=False)
            axes2[j].set_title(subgraph)
            axes2[j].annotate(f'# ↑: {num_pos}, # ↓: {num_neg}', xy = (0.5, 0.95), xycoords='axes fraction', horizontalalignment='center',)
            axes2[j].set_xticklabels(axes1[j].get_xticklabels(), rotation=45, horizontalalignment='right')
            if j != len(real_subgraphs)-1:
                axes2[j].get_legend().remove()
            else:
                pos = axes2[j].get_position()
                axes2[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )

        plt.show()

def count_other_substructures_scatter(results_exp, exp_info, b=None):
    # separate the plots for graphs that have a overestimate and the ones that underestimate
    if len(results_exp) == 0:
        return
        
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    tot_budget = 0
    graph_ids = results_exp["result.graph"].sort_values().unique()
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        results_exp['Adversarial'] = results_exp['result.sign_adversarial_errors'].apply(lambda l: l[i])
        

        for j, subgraph in enumerate(real_subgraphs):

            fig1, axes1 = plt.subplots(1, 7, tight_layout=True)
            if len(real_subgraphs) == 1:
                axes1 = [axes1]
            fig1.set_figwidth(16)
            fig1.set_figheight(4.5)
            fig1.set_dpi(400)
            fig1.suptitle(f"Adversarial examples for {subgraph} {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
            fig1.supxlabel(f'Sign {exp_info["loss_name"]}')
            fig1.supylabel('Adv Count - Test Count')
            diff_counts = {}
            adv_err = []
            data = results_exp[results_exp['Subgraph']==subgraph]
            for index, line in data.iterrows():
                adv_err.append(line['Adversarial'])
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_count = subgraph_counting_all(adv_graph)
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_count = subgraph_counting_all(graph)
                del adv_count['3-Star not ind.'] # always increases
                del adv_count[subgraph]
                del test_count['3-Star not ind.']
                del test_count[subgraph]
                if diff_counts == {}:
                    for s in adv_count.keys():
                        diff_counts[s] = [adv_count[s] - test_count[s]]
                else:
                    for s in adv_count.keys():
                        diff_counts[s].append(adv_count[s] - test_count[s])


            for j, subgraph in enumerate(diff_counts.keys()):
                sns.scatterplot(x=adv_err, y=diff_counts[subgraph], ax=axes1[j])
                axes1[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
                axes1[j].set(xlabel=None)
                axes1[j].set(ylabel=None)
                axes1[j].set_title(subgraph)

        plt.show()


def avg_shortest_path(results_exp, exp_info, b=None):

    def compute_avg_sp(graph: nx.Graph):
        all_sp = []
        for component in (adv_graph.subgraph(c) for c in nx.connected_components(adv_graph)):
            if component.number_of_nodes() > 1:
                sp = nx.average_shortest_path_length(component)
                all_sp.extend([sp for i in range(int(component.number_of_nodes()*(component.number_of_nodes()-1)/2))])
        return mean(all_sp)

    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Average shortest paths of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel('Shortest path average')

        for j, subgraph in enumerate(real_subgraphs):

            data = results_exp[results_exp['Subgraph']==subgraph]
            adv_sp = []
            adv_con = []
            test_sp = []
            test_con = []
            for index, line in data.iterrows():
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_sp.append(compute_avg_sp(adv_graph))
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_sp.append(compute_avg_sp(graph))
            # average
            df = pd.DataFrame(columns=['Subgraph', 'Dataset', 'Shortest path'], index = list(range(2*len(test_sp))))
            
            df['Subgraph'] = subgraph
            df['Dataset'] = ['Test' for i in test_sp] + ['Adversarial' for i in adv_sp]
            df['Shortest path'] = test_sp + adv_sp
            sns.boxplot(data=df, x='Subgraph', y='Shortest path', hue='Dataset', ax = axes[j])
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            else:
                pos = axes[j].get_position()
                axes[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()

def laplacian_spectrum(results_exp, exp_info, b=None):
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Laplacian specturm of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel('Laplacian spectrum')

        for j, subgraph in enumerate(real_subgraphs):

            data = results_exp[results_exp['Subgraph']==subgraph]
            adv_spect = np.array([])
            test_spect = np.array([])
            for index, line in data.iterrows():
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_spect = np.concatenate((adv_spect, nx.laplacian_spectrum(adv_graph)))
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_spect = np.concatenate((test_spect, nx.laplacian_spectrum(graph)))
            # average
            df = pd.DataFrame(columns=['Subgraph', 'Dataset', 'Specturm'], index = list(range(len(test_spect) + len(adv_spect))))
            
            df['Subgraph'] = subgraph
            df['Dataset'] = ['Test' for i in test_spect] + ['Adversarial' for i in adv_spect]
            df['Specturm'] = np.concatenate((test_spect, adv_spect))
            # sns.histplot(data=df, x='Specturm', hue='Dataset', ax = axes[j], multiple="dodge")
            sns.kdeplot(data=df, x='Specturm', hue='Dataset', ax = axes[j], fill=True, alpha=.5, linewidth=0.5)
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            # else:
            #     axes[j].legend(
            #         loc='upper left', 
            #         bbox_to_anchor=(1, 1),
            #         ncol=1, 
            #     )
        plt.show()

def graph_connectivity(results_exp, exp_info, b=None):
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Algebraic connectivity of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel('Laplacian spectrum')

        for j, subgraph in enumerate(real_subgraphs):

            data = results_exp[results_exp['Subgraph']==subgraph]
            adv_conn = []
            test_conn = []
            for index, line in data.iterrows():
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_conn.append(nx.laplacian_spectrum(adv_graph)[1]) # second eigenvalue represents connectivity
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_conn.append(nx.laplacian_spectrum(graph)[1])
            # average
            df = pd.DataFrame(columns=['Subgraph', 'Dataset', 'Specturm'], index = list(range(len(test_conn) + len(adv_conn))))
            
            df['Subgraph'] = subgraph
            df['Dataset'] = ['Test' for i in test_conn] + ['Adversarial' for i in adv_conn]
            df['Specturm'] = np.concatenate((test_conn, adv_conn))
            # sns.histplot(data=df, x='Specturm', hue='Dataset', ax = axes[j], multiple="dodge")
            sns.boxplot(data=df,x='Subgraph', y='Specturm', hue='Dataset', ax = axes[j], showfliers = False,)
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            else:
                axes[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()

def sing_preservation(results_exp, exp_info, b=None):
    def sign(x):
        if x > 0:
            return 1
        else:
            return -1
        
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        results_exp_temp = results_exp
        results_exp_temp['Adversarial'] = results_exp_temp['result.sign_adversarial_errors'].apply(lambda l: l[i])
        results_exp_temp['Cross adversarial'] = results_exp_temp['result.cross_sign_adversarial_errors'].apply(lambda l: l[i])
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Sign preservation between Adv. err and Cross ad. err. {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel('% Adv. examples')

        for j, subgraph in enumerate(real_subgraphs):

            data = results_exp[results_exp['Subgraph']==subgraph]
            sign_pres = 0
            sign_non_pres = 0
            for index, line in data.iterrows():
                adv_sign = sign(line['Adversarial'])
                cross_adv_sign = [sign(e)*adv_sign for e in line['Cross adversarial']] # =1 if sign preserving, -1 elsewhere
                sign_pres += len([True for i in cross_adv_sign if i==1])/len(line['Cross adversarial'])
                sign_non_pres += len([True for i in cross_adv_sign if i==-1])/len(line['Cross adversarial'])

            sign_pres /= len(data)
            sign_non_pres /= len(data)

            df = pd.DataFrame(columns=['Type', 'Perc'], index = list(range(2)))
            df['Type']= ['Preserving', 'Non preserving']
            df['Perc'] = [sign_pres, sign_non_pres]
            # sns.histplot(data=df, x='Specturm', hue='Dataset', ax = axes[j], multiple="dodge")
            sns.barplot(data=df,x='Type', y='Perc', ax = axes[j],)
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            # if j != len(real_subgraphs)-1:
            #     axes[j].get_legend().remove()
            # else:
            #     axes[j].legend(
            #         loc='upper left', 
            #         bbox_to_anchor=(1, 1),
            #         ncol=1, 
            #     )
        plt.show()

def explain_grad(results_exp, exp_info, b=None):
    if len(results_exp) == 0:
        return
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        fig, axes = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes = [axes]
        fig.set_figwidth(16)
        fig.set_figheight(4.5)
        fig.set_dpi(400)
        fig.suptitle(f"Gradient values of the adjacecy matrix {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig.supxlabel('Subgraph')
        fig.supylabel('Gradient')

        fig2, axes2 = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes2 = [axes2]
        fig2.set_figwidth(16)
        fig2.set_figheight(4.5)
        fig2.set_dpi(400)
        fig2.suptitle(f"Gradient values of the adjacecy matrix {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig2.supxlabel('Subgraph')
        fig2.supylabel('Gradient')

        fig3, axes3 = plt.subplots(1, len(real_subgraphs), constrained_layout=True)
        if len(real_subgraphs) == 1:
            axes3 = [axes3]
        fig3.set_figwidth(16)
        fig3.set_figheight(4.5)
        fig3.set_dpi(400)
        fig3.suptitle(f"Gradient values of the subgraphs edges {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig3.supxlabel('Subgraph')
        fig3.supylabel('Gradient')


        for j, subgraph in enumerate(real_subgraphs):

            data = results_exp[results_exp['Subgraph']==subgraph]
            adv_grad = np.array([])
            test_grad= np.array([])
            adv_grad_s = np.array([])
            test_grad_s= np.array([])
            for index, line in data.iterrows():
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                model_architecture = line["config.model_architecture"]
                models_path = line['config.models_path']
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                model_dict = f"{models_path}/{model_architecture}_{subgraph}_{seed}.pth"
                model_params = f"{models_path}/{model_architecture}_{subgraph}_{seed}.json"
                with open(model_params, 'r') as f:
                    h_params = json.load(f)
                gnn = PPGNexpl(**h_params)
                gnn.load_state_dict(torch.load(model_dict, map_location=torch.device('cpu')))

                test_pred = gnn(generate_gnn_input(graph, 'cpu'))
                test_pred.backward()
                subgraph_edges = []
                subgraphs = subgraph_listing(graph, subgraph)
                for s in subgraphs:
                    subgraph_edges.extend(graph.subgraph(s).edges())
                g = gnn.adj.grad.squeeze()
                g = (g + g.T)*gnn.adj
                g_s = [g[i, j] for i,j in subgraph_edges]
                g = g.flatten()
                g = g[torch.nonzero(g).flatten()]
                test_grad = np.concatenate((test_grad, g.detach().numpy()))
                test_grad_s = np.concatenate((test_grad_s, g_s.detach().numpy()))
                adv_pred = gnn(generate_gnn_input(adv_graph, 'cpu'))
                adv_pred.backward()
                subgraph_edges = []
                subgraphs = subgraph_listing(adv_graph, subgraph)
                for s in subgraphs:
                    subgraph_edges.extend(graph.subgraph(s).edges())
                g = gnn.adj.grad.squeeze()
                g = (g + g.T)*gnn.adj
                g_s = [g[i, j] for i,j in subgraph_edges]
                g = g.flatten()
                g = g[torch.nonzero(g).flatten()]
                adv_grad = np.concatenate((adv_grad, g.detach().numpy()))
                adv_grad_s = np.concatenate((adv_grad_s, g_s.detach().numpy()))

            # average
            df = pd.DataFrame(columns=['Subgraph', 'Dataset', 'Gradients'], index = list(range(len(test_grad) + len(adv_grad))))
            df_s = pd.DataFrame(columns=['Subgraph', 'Dataset', 'Gradients'], index = list(range(len(test_grad_s) + len(adv_grad_s))))
            
            df['Subgraph'] = subgraph
            df['Dataset'] = ['Test' for i in test_grad] + ['Adversarial' for i in adv_grad]
            df['Gradients'] = np.concatenate((test_grad, adv_grad))

            df_s['Subgraph'] = subgraph
            df_s['Dataset'] = ['Test' for i in test_grad_s] + ['Adversarial' for i in adv_grad_s]
            df_s['Gradients'] = np.concatenate((test_grad_s, adv_grad_s))
            
            
            sns.kdeplot(data=df, x='Gradients', hue='Dataset', ax = axes[j], fill=True, alpha=.5, linewidth=0.5)
            axes[j].set(xlabel=None)
            axes[j].set(ylabel=None)
            axes[j].set_title(subgraph)
            axes[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes[j].get_legend().remove()
            # else:
            #     axes[j].legend(
            #         loc='upper left', 
            #         bbox_to_anchor=(1, 1),
            #         ncol=1, 
            #     )

            sns.boxplot(data=df, x='Subgraph', y='Gradients', hue='Dataset', ax = axes2[j])
            axes2[j].set(xlabel=None)
            axes2[j].set(ylabel=None)
            axes2[j].set_title(subgraph)
            axes2[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes2[j].get_legend().remove()
            else:
                axes2[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )

            sns.boxplot(data=df_s, x='Subgraph', y='Gradients', hue='Dataset', ax = axes3[j])
            axes3[j].set(xlabel=None)
            axes3[j].set(ylabel=None)
            axes3[j].set_title(subgraph)
            axes3[j].annotate(f'Average over {len(data)} graphs', xy = (0.5,0), xycoords='axes fraction', horizontalalignment='center',)
            if j != len(real_subgraphs)-1:
                axes3[j].get_legend().remove()
            else:
                axes3[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )
        plt.show()

def edges_sign(results_exp, exp_info, b=None):
    if len(results_exp) == 0:
        return
        
    budgets = results_exp["config.budgets"].iloc[0]
    adv_path = results_exp["config.adversarial_graphs_folder"].iloc[0]
    tot_budget = 0
    graph_ids = results_exp["result.graph"].sort_values().unique()
    dataset_path = results_exp["config.dataset_path"].iloc[0]
    
    graphs, _ = dgl.load_graphs(dataset_path)
        
        
    for i, budget in enumerate(budgets):
        if b is not None and b != i:
            continue
        real_subgraphs = [subgraph for subgraph in exp_info['subgraphs']  if len(results_exp[results_exp['Subgraph']==subgraph]) > 0]
        if len(real_subgraphs) == 0:
            continue
        results_exp['Adversarial'] = results_exp['result.sign_adversarial_errors'].apply(lambda l: l[i])
        fig1, axes1 = plt.subplots(1, len(real_subgraphs), tight_layout=True)
        if len(real_subgraphs) == 1:
            axes1 = [axes1]
        fig1.set_figwidth(16)
        fig1.set_figheight(4.5)
        fig1.set_dpi(400)
        fig1.suptitle(f"# of edges of adversarial examples {exp_info['arch']} {exp_info['exp']}, budget {exp_info['budget_perc'][i]}, {exp_info['strategy']}", fontsize = 20)
        fig1.supxlabel('Subgraph')
        fig1.supylabel('# Edges')


        for j, subgraph in enumerate(real_subgraphs):
            test_counts = []
            pos_counts = []
            neg_counts = []
            num_pos = 0
            num_neg = 0
            data = results_exp[results_exp['Subgraph']==subgraph]
            for index, line in data.iterrows():
                sign_err = line['Adversarial']
                graph_id = line["result.graph"]
                seed = line["config.seed"]
                adv_graph = Graph(np.load(os.path.join(adv_path, f"{exp_info['arch']}_{graph_id}_{subgraph}_{seed}_{i}.npy"), allow_pickle=True)[0])
                adv_count = adv_graph.number_of_edges()
                graph = Graph(dgl.to_networkx(graphs[graph_id]))
                test_counts.append(graph.number_of_edges())

                if sign_err > 0:
                    num_pos += 1
                    pos_counts.append(adv_count)
                if sign_err < 0:
                    num_neg += 1
                    neg_counts.append(adv_count)
           
            df = pd.DataFrame(columns=['Subgraph', '# Edges', 'Graph type'], index = list(range(2*len(test_counts))))
            df['Subgraph'] = subgraph
            df['# Edges'] = test_counts + pos_counts + neg_counts
            df['Graph type'] = ['Test' for _ in test_counts] + ['Overestimating' for _ in pos_counts] + ['Underestimating' for _ in neg_counts]
            
            sns.violinplot(data=df, x='Subgraph', y='# Edges', hue='Graph type', ax = axes1[j])
            axes1[j].annotate(f'# ↑: {num_pos}, # ↓: {num_neg}', xy = (0.5, 0.95), xycoords='axes fraction', horizontalalignment='center',)
            axes1[j].set(xlabel=None)
            axes1[j].set(ylabel=None)
            axes1[j].set_title(subgraph)
            if j != len(real_subgraphs)-1:
                axes1[j].get_legend().remove()
            else:
                pos = axes1[j].get_position()
                axes1[j].legend(
                    loc='upper left', 
                    bbox_to_anchor=(1, 1),
                    ncol=1, 
                )

        plt.show()

def real_adv_examples(results_exp, exp_info, delta, edge=False, degree=False, graph=False, count =False, spath=False, spectrum=False, connectivity=False, sign_pres=False, count_sign=False, count_scatter = False, sign_adv = False, grad = False, edge_sign = False, max_budget = 3, legend = False):
    # Difference between the adv err and the cross adv err
    budgets = results_exp["config.budgets"].iloc[0][:max_budget+1]
    
    b = [sum(budgets[:i+1]) for i in range(len(budgets))]
    print(b)
    adv_examples = {}
    adv_examples_seeds = {}
    adv_examples_std = {}
    adv_examples_seeds_std = {}
    for subgraph in exp_info['subgraphs']:
        adv_examples[subgraph] = []
        adv_examples_seeds[subgraph] = []
        adv_examples_std[subgraph] = []
        adv_examples_seeds_std[subgraph] = []
    for i , budget in enumerate(budgets):
        results_exp_temp = results_exp
        results_exp_temp['Adversarial'] = results_exp_temp['result.adversarial_error'].apply(lambda l: l[i])
        results_exp_temp['Adversarial Count'] = results_exp_temp['result.adversarial_count'].apply(lambda l: l[i])
        # results_exp_temp['Cross Adversarial'] = results_exp_temp['result.cross_adversarial_errors_average'].apply(lambda l: l[i])
        results_exp_temp['Cross Adversarial'] = results_exp_temp['result.cross_adversarial_errors'].apply(lambda l: l[i])
        if exp_info['loss_name'] == 'MAE/count':
            results_exp_temp['Adversarial L1'] = results_exp_temp['Adversarial']*(results_exp_temp['Adversarial Count']+1)
            # results_exp_temp['Cross Adversarial L1'] = results_exp_temp['Cross Adversarial']*(results_exp_temp['Adversarial Count'] +1)
            results_exp_temp['Cross Adversarial L1'] = results_exp_temp.apply(lambda x: [x['Cross Adversarial'][i]*(x['Adversarial Count'] +1) for i in range(4)], axis = 1)
        elif exp_info['loss_name'] == 'MAE':
            results_exp_temp['Adversarial L1'] = results_exp_temp['Adversarial']
            results_exp_temp['Cross Adversarial L1'] = results_exp_temp['Cross Adversarial']
        else:
            raise ValueError("Loss not supported!")
        results_exp_temp['ARE'] = (results_exp_temp['Adversarial'] - results_exp_temp['Test'])/results_exp_temp['Test'] #adversarial relative error
        # results_exp_temp['Cross ARE'] = (results_exp_temp['Cross Adversarial'] - results_exp_temp['Cross Test'])/results_exp_temp['Cross Test']
        results_exp_temp['Cross ARE'] = results_exp_temp.apply(lambda x: [(x['Cross Adversarial'][i] - x['Cross Test'])/x['Cross Test'] for i in  range(4)], axis = 1)
        
        #select the REAL ADVERSARIAL EXAMPLES
        results_exp_temp = results_exp_temp[results_exp_temp['Adversarial L1'] > 0.5]
        results_exp_temp = results_exp_temp[results_exp_temp['ARE'] > delta]

        for j, subgraph in enumerate(exp_info['subgraphs']):
            l =[]
            for k in range(5):
                data = results_exp_temp[results_exp_temp['Subgraph']==subgraph]
                data = data[data['config.seed']==k]
                l.append(5*len(data)/len(results_exp[results_exp['Subgraph'] == subgraph]))
            adv_examples[subgraph].append(mean(l))
            adv_examples_std[subgraph].append(stdev(l)/sqrt(5))
            
         # adversarial examples that fool also the exp_info['arch']?
        results_exp_temp['cond 1'] = results_exp_temp['Cross Adversarial L1'].apply(lambda x: True if x > [0.5 for i in range(4)] else False)
        results_exp_temp['cond 2'] = results_exp_temp['Cross ARE'].apply(lambda x: True if x  > [delta for i in range(4)] else False)
        results_exp_temp = results_exp_temp[(results_exp_temp['cond 1'] == True) & (results_exp_temp['cond 2'] == True) ]
        
        for j, subgraph in enumerate(exp_info['subgraphs']):
            l =[]
            for k in range(5):
                data = results_exp_temp[results_exp_temp['Subgraph']==subgraph]
                data = data[data['config.seed']==k]
                l.append(5*len(data)/len(results_exp[results_exp['Subgraph'] == subgraph]))
            adv_examples_seeds[subgraph].append(mean(l))
            adv_examples_seeds_std[subgraph].append(stdev(l)/sqrt(5))

        # structural information about the adv examples
        if edge:
            edge_count(results_exp_temp, exp_info, i)
        if degree:
            degree_distribution(results_exp_temp, exp_info, i)    
        if graph:
            best_adv_exaples(results_exp_temp, exp_info, i)
        if count:
            #count_other_substructures(results_exp_temp, exp_info, i)
            subgraph_count(results_exp_temp, exp_info, i)
        if spath:
            avg_shortest_path(results_exp_temp, exp_info, i)
        if spectrum:
            laplacian_spectrum(results_exp_temp, exp_info, i)
        if connectivity:
            graph_connectivity(results_exp_temp, exp_info, i)
        if sign_pres:
            sing_preservation(results_exp_temp, exp_info, i)
        if count_sign:
            count_other_substructures_sign(results_exp_temp, exp_info, i)
        if count_scatter:
            count_other_substructures_scatter(results_exp_temp, exp_info, i)
        if sign_adv:
            sign_adversarial_error(results_exp_temp, exp_info, i)
        if grad:
            explain_grad(results_exp_temp, exp_info, i)
        if edge_sign:
            edges_sign(results_exp_temp, exp_info, i)
            
        

    fig, axes = plt.subplots(1, len(exp_info['subgraphs']), constrained_layout=True)
    
    fig.set_figwidth(3*legend+len(exp_info['subgraphs'])*3)
    fig.set_figheight(5)
    fig.set_dpi(400)
    # fig.suptitle(f"% of real adversarial examples {exp_info['arch']} {exp_info['exp']}, {exp_info['strategy']}, δ = {delta}", fontsize = 20)
    fig.suptitle(f"{exp_info['arch']}, {exp_info['strategy']}", fontsize = 30) #, {exp_info['strategy']} Perturbations, $\delta$ = {delta}
    fig.supxlabel('Budget ($ \% $ of $\#$ edges)') # Δ
    fig.supylabel('$\%$ Adv. Examples')
    x_axes = [f"{exp_info['budget_perc'][i]}" for i in range(len(budgets))]
    for j, subgraph in enumerate(exp_info['subgraphs']):
        #axes[j].fill_between(x=x_axes, y1=adv_examples[subgraph], y2=[1 for i in range(len(budgets))], alpha=0.3, label='Robust', color='grey')
        #sns.lineplot(x=x_axes, y=adv_examples[subgraph], ax=axes[j], label='1 Seed')
        axes[j].errorbar(x=x_axes, y=adv_examples[subgraph], yerr = adv_examples_std[subgraph], label='Non Robust', capsize=3, fmt='.-')
        axes[j].fill_between(x=x_axes, y1= adv_examples_seeds[subgraph], y2=adv_examples[subgraph], alpha=0.3)
        
        #sns.lineplot(x=x_axes, y=adv_examples_seeds[subgraph], ax=axes[j], label='All Seeds')
        axes[j].errorbar(x=x_axes, y=adv_examples_seeds[subgraph], yerr = adv_examples_seeds_std[subgraph], label='Non Robust \n(Transfer)', capsize=3, fmt='.-')
        axes[j].fill_between(x=x_axes, y1=adv_examples_seeds[subgraph], y2=[0 for i in range(len(budgets))], alpha=0.3)  
        axes[j].set(xlabel=None)
        axes[j].set(ylabel=None)
        axes[j].set_title(subgraph)
        axes[j].set_ylim(-0.01,1.01)
        # axes[j].get_legend().remove()
        if legend is False or j != len(exp_info['subgraphs'])-1:
            pass #axes[j].get_legend().remove()
        else:
            pos = axes[j].get_position()
            axes[j].legend(
                loc='upper left', 
                bbox_to_anchor=(1, 1),
                ncol=1, 
            )
    fig.savefig(f"figs/{exp_info['arch']}_{exp_info['strategy']}.pdf")

    fig, axes = plt.subplots(1, len(exp_info['subgraphs']), constrained_layout=True)
    
    fig.set_figwidth(3+len(exp_info['subgraphs'])*3)
    fig.set_figheight(5)
    fig.set_dpi(400)
    # fig.suptitle(f"% of real adversarial examples {exp_info['arch']} {exp_info['exp']}, {exp_info['strategy']}, δ = {delta}", fontsize = 20)
    fig.suptitle(f"{exp_info['arch']}, {exp_info['strategy']}", fontsize = 28) #, {exp_info['strategy']} Perturbations, $\delta$ = {delta}
    fig.supxlabel('Budget ($ \% $ of $\#$ edges)') # Δ
    fig.supylabel('$\%$ Adv. Examples')
    x_axes = [f"{exp_info['budget_perc'][i]}" for i in range(len(budgets))]
    for j, subgraph in enumerate(exp_info['subgraphs']):
        #axes[j].fill_between(x=x_axes, y1=adv_examples[subgraph], y2=[1 for i in range(len(budgets))], alpha=0.3, label='Robust', color='grey')
        #sns.lineplot(x=x_axes, y=adv_examples[subgraph], ax=axes[j], label='1 Seed')
        axes[j].errorbar(x=x_axes, y=adv_examples[subgraph], yerr = adv_examples_std[subgraph], label='Non Robust', capsize=3, fmt='.-')
        #axes[j].fill_between(x=x_axes, y1= adv_examples_seeds[subgraph], y2=adv_examples[subgraph], alpha=0.3)
        axes[j].fill_between(x=x_axes, y1= adv_examples[subgraph], y2=[0 for i in range(len(budgets))], alpha=0.3)
        
        #sns.lineplot(x=x_axes, y=adv_examples_seeds[subgraph], ax=axes[j], label='All Seeds')
        #axes[j].errorbar(x=x_axes, y=adv_examples_seeds[subgraph], yerr = adv_examples_seeds_std[subgraph], label='Non Robust \n(Transfer)', capsize=3, fmt='.-')
        #axes[j].fill_between(x=x_axes, y1=adv_examples_seeds[subgraph], y2=[0 for i in range(len(budgets))], alpha=0.3)  
        axes[j].set(xlabel=None)
        axes[j].set(ylabel=None)
        axes[j].set_title(subgraph)
        axes[j].set_ylim(-0.01,1.01)
        # axes[j].get_legend().remove()
        if legend is False or j != len(exp_info['subgraphs'])-1:
            pass #axes[j].get_legend().remove()
        else:
            pos = axes[j].get_position()
            axes[j].legend(
                loc='upper left', 
                bbox_to_anchor=(1, 1),
                ncol=1, 
            )
    fig.savefig(f"figs/{exp_info['arch']}_{exp_info['strategy']}_no_tran.pdf")
    # compute aoc
    # 1 seed
    print('AOC 1 seed')
    for subgraph in exp_info['subgraphs']:
        aoc = 0.
        for i in range(max_budget):
            aoc += 0.5*(adv_examples[subgraph][i] + adv_examples[subgraph][i+1])*(budget_perc_float[i+1] - budget_perc_float[i])
        aoc = aoc / (budget_perc_float[-1] - budget_perc_float[0])
        print(f'{subgraph}:  {aoc}')

    print('AOC all seeds')
    for subgraph in exp_info['subgraphs']:
        aoc = 0.
        for i in range(max_budget):
            aoc += 0.5*(adv_examples_seeds[subgraph][i] + adv_examples_seeds[subgraph][i+1])*(budget_perc_float[i+1] - budget_perc_float[i])
        aoc = aoc / (budget_perc_float[-1] - budget_perc_float[0])
        print(f'{subgraph}:  {aoc}')