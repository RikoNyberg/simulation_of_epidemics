import random
from sklearn.datasets import base
import numpy as np
import networkx as nx

# Task 5
def get_immunized_random(epidemic_graph, n_nodes):
  nodes_to_immunize = random.sample(range(len(epidemic_graph)), n_nodes)
  return nodes_to_immunize


def get_immunized_neighbors(epidemic_graph, n_nodes):
  immunized_neighbors = []
  rand_nodes = random.sample(range(len(epidemic_graph)), len(epidemic_graph))
  for rand_node in rand_nodes:
      node_neighbors = set(epidemic_graph.neighbors(rand_node))
      node_neighbors = node_neighbors - set(immunized_neighbors)
      if len(node_neighbors) == 0:
        pass
      else:
        random.shuffle(list(node_neighbors))
        immunized_neighbors.append(node_neighbors.pop())
      if len(immunized_neighbors) == n_nodes:
       return immunized_neighbors
  assert False, 'Less nodes than required immunized nodes!'


def dict_to_sorted_list(input_dict, n_nodes):
  output_list = []
  for key, value in sorted(input_dict.items(), key=lambda x: x[1], reverse=True):
    if n_nodes > 0:  
      output_list.append(key)
      n_nodes -= 1
    else:
      break
  return output_list


def get_immunized_movement_info(movement_info, im_list):
  im_movement_info = np.copy(np.asarray(list(movement_info)))
  for immunized in im_list:
    im_movement_info = im_movement_info[~(im_movement_info == immunized).any(1)]
  return im_movement_info


def get_immunized_nodes_and_movement(epidemic_graph, movement_info, n_immunized_nodes):
  # im_bunch = Immunized nodes in a Bunch format:
  im_bunch = base.Bunch()

  # Choose immunization by neighbor of a random node:
  im_bunch.neighbors = get_immunized_neighbors(
      epidemic_graph, n_immunized_nodes)
  im_bunch.neighbors_move = get_immunized_movement_info(
      movement_info, im_bunch.neighbors)

  # Choose immunization by random:
  im_bunch.random = get_immunized_random(
      epidemic_graph, n_immunized_nodes)
  im_bunch.random_move = get_immunized_movement_info(movement_info, im_bunch.random)

  # Choose immunization by k-shell (core):
  core_dict = nx.core_number(epidemic_graph)
  im_bunch.core = dict_to_sorted_list(core_dict, n_immunized_nodes)
  im_bunch.core_move = get_immunized_movement_info(movement_info, im_bunch.core)

  # Choose immunization by unweighted clustering coefficient c:
  clustering_dict = nx.clustering(epidemic_graph)
  im_bunch.clustering = dict_to_sorted_list(clustering_dict, n_immunized_nodes)
  im_bunch.clustering_move = get_immunized_movement_info(movement_info, im_bunch.clustering)

  # Choose immunization by degree k:
  degree_dict = epidemic_graph.degree()
  im_bunch.degree = dict_to_sorted_list(degree_dict, n_immunized_nodes)
  im_bunch.degree_move = get_immunized_movement_info(movement_info, im_bunch.degree)

  # Choose immunization by strenght:
  strenght_dict = epidemic_graph.degree(weight='weight')
  im_bunch.strenght = dict_to_sorted_list(strenght_dict, n_immunized_nodes)
  im_bunch.strenght_move = get_immunized_movement_info(movement_info, im_bunch.strenght)

  # Choose immunization by betweeness:
  betweenness_dict = nx.betweenness_centrality(epidemic_graph)
  im_bunch.betweenness = dict_to_sorted_list(betweenness_dict, n_immunized_nodes)
  im_bunch.betweenness_move = get_immunized_movement_info(movement_info, im_bunch.betweenness)

  # Choose immunization by closeness:
  closeness_dict = nx.closeness_centrality(epidemic_graph)
  im_bunch.closeness = dict_to_sorted_list(closeness_dict, n_immunized_nodes)
  im_bunch.closeness_move = get_immunized_movement_info(movement_info, im_bunch.closeness)

  # All immunized nodes:
  neighbors_set = set(im_bunch.neighbors)
  random_set = set(im_bunch.random)
  core_set = set(im_bunch.core)
  clustering_set = set(im_bunch.clustering)
  degree_set = set(im_bunch.degree) 
  strenght_set = set(im_bunch.strenght)
  betweenness_set = set(im_bunch.betweenness) 
  closeness_set = set(im_bunch.closeness) 

  all_immunized_nodes = neighbors_set.union(
      random_set, core_set, clustering_set, degree_set, strenght_set, betweenness_set, closeness_set)
  im_bunch.all_immunized_nodes = list(all_immunized_nodes)

  return im_bunch


def get_seed_nodes(epidemic_graph, m_seed_nodes, all_immunized_nodes):
  possible_seed_nodes = list(set(range(len(epidemic_graph))) - set(all_immunized_nodes))
  random.shuffle(possible_seed_nodes)
  seed_nodes = possible_seed_nodes[:m_seed_nodes]
  return seed_nodes


def average_infection_times_for_different_seed_nodes(movement_info, bin_times, n_iter, inf_prob, infected_nodes):
  # Average infection times for different infection seed nodes:
  for inf_node in infected_nodes:
      print('First infected node:', inf_node)
      average_bin_array = average_infection_times(
          inf_node, inf_prob, movement_info, n_iter)
      plot(bin_times, average_bin_array, 'bin-times',
            'inf_node = {}'.format(inf_node))
  plt.legend()
  plt.savefig('figs/plot_of_infection_times_x_starting_nodes')
  print('Saved plot: Prevalence (fraction of infected nodes) with different infection starting nodes as a function of bin-times')
  plt.clf()
  return


def get_protection_strategies(movement_info, epidemic_graph):
  # Amount of immunized nodes and taking away immunized nodes from movement_info (creating new movement_info lists):
  n_immunized_nodes = 10
  im_bunch = get_immunized_nodes_and_movement(epidemic_graph, movement_info, n_immunized_nodes)

  # Random seed nodes where the epidemic can start:
  # (Does not include any immunized nodes in our case)
  m_seed_nodes = 20
  seed_nodes = get_seed_nodes(
      epidemic_graph, m_seed_nodes, im_bunch.all_immunized_nodes)

  ## All possible movement_info lists:
  different_strategies = [[im_bunch.neighbors, im_bunch.neighbors_move, 'random neighbor'],
  [im_bunch.random, im_bunch.random_move, 'random'],
  [im_bunch.core, im_bunch.core_move, 'top core'], 
  [im_bunch.clustering, im_bunch.clustering_move, 'top clustering'], 
  [im_bunch.degree, im_bunch.degree_move,  'top degree'], 
  [im_bunch.strenght, im_bunch.strenght_move,  'top strenght'], 
  [im_bunch.betweenness, im_bunch.betweenness_move,  'top betweenness'], 
  [im_bunch.closeness, im_bunch.closeness_move, 'top closeness']]

  print('Different immunization strategies are (with {} immunized nodes and {} different seed nodes):'.format(
      n_immunized_nodes, m_seed_nodes))
  for strategies, i in zip(different_strategies, range(1,len(different_strategies)+1)):
    print('{}. {}'.format(i, strategies[2]))
  return different_strategies, seed_nodes

