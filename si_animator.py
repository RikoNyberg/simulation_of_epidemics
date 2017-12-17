import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os
import random

from epidemic_simulation import get_epidemic_infection_times, import_data, get_epidemic_infection_times_np_array
from best_protection_strategy import get_protection_strategies


# The locations and schedule of movements:
EVENT_FNAME = "data/events_US_air_traffic_GMT.txt"
# Network file of the nodes:
NETWORK = "data/aggregated_US_air_traffic_network_undir.edg"
# Locations and names for the nodes in the visualization:
AIRPORT_INFO_CSV_FNAME = 'data/US_airport_id_info.csv'
# Background image for the animation:
BACKGROUND_IMAGE = 'data/US_air_bg.png'


class SI_AnimHelper(object):

    def __init__(self, infection_times):
        event_fname = EVENT_FNAME
        if not os.path.exists(event_fname):
            raise IOError("File " + event_fname + "could not be found")

        self.ed = np.genfromtxt(
            event_fname,
            delimiter=' ',
            dtype=None,
            names=True
        )

        self.infection_times = infection_times
        airport_info_csv_fname = AIRPORT_INFO_CSV_FNAME

        if not os.path.exists(airport_info_csv_fname):
            raise IOError("File " + event_fname + "could not be found")

        id_data = np.genfromtxt(
            airport_info_csv_fname,
            delimiter=',',
            dtype=None,
            names=True
        )
        self.xcoords = id_data['xcoordviz']
        self.ycoords = id_data['ycoordviz']
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        # ([0, 0, 1, 1])
        bg_figname = BACKGROUND_IMAGE
        img = plt.imread(bg_figname)
        self.axis_extent = (-6674391.856090588, 4922626.076444283,
                            -2028869.260519173, 4658558.416671531)
        self.img = self.ax.imshow(img, extent=self.axis_extent)
        self.ax.set_xlim((self.axis_extent[0], self.axis_extent[1]))
        self.ax.set_ylim((self.axis_extent[2], self.axis_extent[3]))
        self.ax.set_axis_off()
        self.time_text = self.ax.text(
            0.1, 0.1, "", transform=self.ax.transAxes)
        self.scat_planes = self.ax.scatter([], [], s=0.2, color="k")

        self.n = len(self.xcoords)
        self.airport_colors = np.array(
            [[0, 1, 0] for i in range(self.n)], dtype=float)
        self.scat_airports = self.ax.scatter(
            self.xcoords, self.ycoords, c=self.airport_colors, s=5, alpha=0.2)

    def draw(self, frame_time):
        """
        Draw the current situation of the epidemic spreading.

        Parameters
        ----------
        frame_time : int
            the current time in seconds
            (should lie somewhere between 1229231100 and 1230128400)
        """
        time_str = time.asctime(time.gmtime(frame_time))
        self.time_text.set_text(time_str)
        # this should be improved
        unfinished_events = (self.ed['EndTime'] > frame_time)
        started_events = (self.ed['StartTime'] < frame_time)

        # oge = 'on going events'
        oge = self.ed[started_events * unfinished_events]
        fracs_passed = (float(frame_time) - oge['StartTime']) / oge['Duration']
        ongoing_xcoords = ((1 - fracs_passed) * self.xcoords[oge['Source']]
                           + fracs_passed * self.xcoords[oge['Destination']])
        ongoing_ycoords = ((1 - fracs_passed) * self.ycoords[oge['Source']]
                           + fracs_passed * self.ycoords[oge['Destination']])
        self.scat_planes.set_offsets(
            np.array([ongoing_xcoords, ongoing_ycoords]).T)
        self.event_durations = self.ed['Duration']

        infected = (self.infection_times < frame_time)

        self.airport_colors[infected] = (1, 0, 0)  # red
        self.airport_colors[~infected] = (0, 1, 1)  # green
        self.scat_airports = self.ax.scatter(
            self.xcoords, self.ycoords, c=self.airport_colors, s=20, alpha=0.5)

    def draw_anim(self, frame_time):
        self.draw(frame_time)
        return self.time_text, self.scat_planes, self.scat_airports

    def init(self):
        return self.time_text, self.scat_planes, self.scat_airports


def visualize_si(infection_times,
                 viz_start_time=1229231100,
                 viz_end_time=1230128400,
                 tot_viz_time_in_seconds=60,
                 fps=10,
                 save_fname=None,
                 writer=None):
    """
    Animate the infection process as a function of time.

    Parameters
    ----------
    infection_times : numpy array
        infection times of the nodes
    viz_start_time : int
        start time in seconds after epoch
    viz_end_time : int
        end_time in seconds after epoch
    tot_viz_time_in_seconds : int
        length of the animation for the viewer
    fps : int
        frames per second (use low values on slow computers)
    save_fname : str
        where to save the animation
    """

    # By default, spans the whole time range of infections.
    times = np.linspace(
        viz_start_time, viz_end_time, fps * tot_viz_time_in_seconds + 1)
    siah = SI_AnimHelper(infection_times)
    ani = FuncAnimation(
        siah.fig, siah.draw_anim, init_func=siah.init, frames=times,
        interval=1000 / fps, blit=True
    )
    if save_fname is not None:
        print("Saving the animation can take quite a while.")
        print("Be patient...")
        ani.save(save_fname, codec='mpeg4')
    else:
        plt.show()


def plot_network_usa(net, xycoords, edges=None, linewidths=None):
    """
    Plot the network usa.
    The file US_air_bg.png should be located in the same directory
    where you run the code.

    Parameters
    ----------
    net : the network to be plotted
    xycoords : dictionary of node_id to coordinates (x,y)
    edges : list of node index tuples (node_i,node_j),
            if None all network edges are plotted.
    linewidths : see nx.draw_networkx documentation
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 0.9])
    # ([0, 0, 1, 1])
    bg_figname = 'US_air_bg.png'
    img = plt.imread(bg_figname)
    axis_extent = (-6674391.856090588, 4922626.076444283,
                   -2028869.260519173, 4658558.416671531)
    ax.imshow(img, extent=axis_extent)
    ax.set_xlim((axis_extent[0], axis_extent[1]))
    ax.set_ylim((axis_extent[2], axis_extent[3]))
    ax.set_axis_off()
    nx.draw_networkx_nodes(net,
                           pos=xycoords,
                           with_labels=False,
                           node_color='k',
                           node_size=5,
                           alpha=0.2)
    if linewidths == None:
        linewidths = np.ones(len(edges))

    for edge, lw in zip(edges, linewidths):
        nx.draw_networkx_edges(
            net,
            pos=xycoords,
            with_labels=True,
            edge_color='r',
            width=lw,
            edgelist=[edge],
            alpha=lw,
        )
    return fig, ax

# Tasks 2 & 3


def bin_inf_times(inf_times, bin_times):
    bins_array = []
    for time in bin_times:
        bins = sum(time > inf_times) / len(inf_times)
        bins_array.append(bins)
    return bins_array


def average_infection_times(first_infected_nodes, infection_probability, movement_info, n_iter):
    bins_array = []
    for n in range(n_iter):
        infection_times = get_epidemic_infection_times(
            first_infected_nodes, infection_probability, movement_info)
        bin_array = bin_inf_times(infection_times, bin_times)
        bins_array.append(bin_array)
    average_bin_array = np.average(bins_array, axis=0)

    return average_bin_array


def plot(x, y, x_name, value, y_name='Prevalence (fraction of infected nodes)'):
    understandable_time_format = (x - min(x)) / (60 * 60 * 24)
    time_format = 'days'
    plt.plot(understandable_time_format, y, label=value)

    plt.ylabel(y_name)
    plt.xlabel('{} ({})'.format(x_name, time_format))
    plt.title('{} as a function of {} ({})'.format(
        y_name, x_name, time_format))

    return


def average_infection_times_for_different_infection_probabilities(movement_info, bin_times, n_iter, inf_probabilities, inf_node):
    # Average infection times for different infection probabilities:
    for inf_prob in inf_probabilities:
        print('Infection probability:', inf_prob)
        average_bin_array = average_infection_times(
            inf_node, inf_prob, movement_info, n_iter)
        plot(bin_times, average_bin_array, 'bin-times',
             'inf_prob = {}'.format(inf_prob))
    plt.legend()
    plt.savefig('figs/plot_of_infection_times_x_probabilities')
    print('Saved plot: Prevalence (fraction of infected nodes) with different infection probabilities as a function of bin-times')
    plt.clf()

    return


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

# Task 4


def median_inf_time(inf_prob, movement_info, n_iter):
    infected_nodes = random.sample(range(max(movement_info['Source']) + 1), n_iter)
    times_array = []
    for inf_node in infected_nodes:
        print('First infected node:', inf_node)
        infection_times = get_epidemic_infection_times(
            inf_node, inf_prob, movement_info)
        times_array.append(infection_times)

    median_time_array = np.median(times_array, axis=0)
    # Taking away is the "max time" (10000000000)... so it never got infected
    median_time_array[median_time_array == 10000000000] = 0

    return median_time_array


def dict_to_sorted_list(input_dict):
    output_list = []
    for key in sorted(input_dict.keys()):
        output_list.append(input_dict[key])
    return output_list


def scatter(x, y, x_name, y_name='Median infection time'):

    understandable_time_format = (y - min(y)) / (60 * 60 * 24)
    time_format = 'days'
    plt.scatter(x, understandable_time_format)
    # plt.yscale('log')
    # plt.xscale('log')

    correlation, pvalue = stats.spearmanr(x, y)
    print('Correlation between {} and {} is: {}'.format(
        x_name, y_name, correlation))

    plt.ylabel('{} ({})'.format(y_name, time_format))
    plt.xlabel(x_name)
    plt.title('{} ({}) as a function of {}'.format(
        y_name, time_format, x_name))

    plt.savefig('figs/scatter_of_median_x_{}'.format(x_name))
    plt.clf()
    return


def do_and_scatter_statisticst(movement_info, epidemic_graph):
    # Mean of each node's infection times:
    n_iter = 50
    inf_prob = 0.5
    median_time_array = median_inf_time(inf_prob, movement_info, n_iter)

    # Draw the sum by index:
    #scatter(range(len(median_time_array)), median_time_array, 'airport_number')

    # Draw by k-shell (core):
    core_dict = nx.core_number(epidemic_graph)
    core_list = dict_to_sorted_list(core_dict)
    scatter(core_list, median_time_array, 'k-shell')

    # Draw by unweighted clustering coefficient c:
    clustering_dict = nx.clustering(epidemic_graph)
    clustering_list = dict_to_sorted_list(clustering_dict)
    scatter(clustering_list, median_time_array, 'clustering_coefficient')

    # Draw by degree k:
    degree_dict = epidemic_graph.degree()
    degree_list = dict_to_sorted_list(degree_dict)
    scatter(degree_list, median_time_array, 'degree')

    # Draw by strenght:
    strenght_dict = epidemic_graph.degree(weight='weight')
    strenght_list = dict_to_sorted_list(strenght_dict)
    scatter(strenght_list, median_time_array, 'strenght')

    # Draw by betweeness:
    betweenness_dict = nx.betweenness_centrality(epidemic_graph)
    betweenness_list = dict_to_sorted_list(betweenness_dict)
    scatter(betweenness_list, median_time_array, 'betweeness')

    # Draw by closeness:
    closeness_dict = nx.closeness_centrality(epidemic_graph)
    closeness_list = dict_to_sorted_list(closeness_dict)
    scatter(closeness_list, median_time_array, 'closeness_centrality')

    # Draw by the sum of each parameters normalized value:
    core_array = np.array(core_list)/max(core_list)
    #clustering_array = np.array(clustering_list) / max(clustering_list) 
    # --> This has low "Spearman rank-correlation coefficient"
    degree_array = np.array(degree_list) / max(degree_list)
    strenght_array = np.array(strenght_list) / max(strenght_list)
    betweenness_array = np.array(betweenness_list) / max(betweenness_list)
    closeness_array = np.array(closeness_list) / max(closeness_list)

    data = [core_array, degree_array,
            strenght_array, betweenness_array, closeness_array]
    data_array = np.array(data)
    sum_data = data_array.sum(axis=0)
    scatter(sum_data, median_time_array, 'normalized_SUM')

    return

# Task 5
def read_undirected_weighted_edgelist(network):
    network_info = import_data(network)
    network_list = []
    for node_1, node_2, weight in network_info:
        network_list.append(
            '{} {} {}'.format(node_1, node_2, weight))
    # print(network_list)
    epidemic_graph = nx.parse_edgelist(
        network_list, nodetype=int, data=(('weight', float),))

    return epidemic_graph


def get_average_infection_times_with_different_seeds(inf_prob, movement_info, seed_nodes, immunized_nodes_list):
  bins_array = []
  for seed in seed_nodes:
    inf_times = get_epidemic_infection_times_np_array(
        seed, inf_prob, movement_info, immunized_nodes_list)
    bin_array = bin_inf_times(inf_times, bin_times)
    bins_array.append(bin_array)
  ave_inf_times = np.average(bins_array, axis=0)

  return ave_inf_times


def get_np_array_from_tuple_list(movement_info):
    movement_info_array = []
    for movement in movement_info:
        row = []
        for mov in movement:
            row.append(mov)
        movement_info_array.append(row)
    movement_info_array = np.array(movement_info_array)

    return movement_info_array

if __name__ == "__main__":

    movement_info = import_data(EVENT_FNAME)
    movement_info_array = get_np_array_from_tuple_list(movement_info)


    bin_times = np.linspace(min(movement_info['StartTime']), max(movement_info['EndTime']), 1000)
    epidemic_graph = read_undirected_weighted_edgelist(NETWORK)
    # TODO: This import DOES NOT WORK... makes it directed network?!?!
    # (something wrong with networkx-library):
    #epidemic_graph = nx.read_weighted_edgelist(NETWORK)
    # --> Self made function for the edge file (edg) reading/import :D
    
    ## Task 2:
    ## Average infection times for different infection probabilities:
    n_iter = 10
    inf_probabilities = [0.01, 0.05, 0.1, 0.5, 1.0]
    inf_node = 0
    average_infection_times_for_different_infection_probabilities(movement_info, bin_times, n_iter, inf_probabilities, inf_node)
    
    ## Average infection times for different infection seed nodes:
    n_iter = 10
    inf_prob = 0.1
    infected_nodes = [0, 4, 41, 100, 200]
    average_infection_times_for_different_seed_nodes(movement_info, bin_times, n_iter, inf_prob, infected_nodes)

    ## Task 4:
    do_and_scatter_statisticst(movement_info, epidemic_graph)

    ## Task 5
    print('Looking for the best protection strategy...')
    different_strategies, seed_nodes = get_protection_strategies(
        movement_info_array, epidemic_graph)
    ## Average infection times for different immunization strategies and infection seed nodes:
    inf_prob = 0.5
    for strategy in different_strategies:
        print('\nCalculating infection times with immunized {} nodes...'.format(
            strategy[2]))
        ## Excluding the infected nodes from the plot:
        #ave_inf_times = get_average_infection_times_with_different_seeds(inf_prob, strategy[1], seed_nodes, [])

        ## Including the infected nodes to the plot:
        ave_inf_times = get_average_infection_times_with_different_seeds(inf_prob, movement_info_array, seed_nodes, strategy[0])
        plot(bin_times, ave_inf_times, 'bin-times',
            'strategy = {}'.format(strategy[2]))
        print('{} calculated and plotted'.format(strategy[2]))
    plt.tight_layout()
    plt.legend()
    plt.savefig('figs/plot_of_infection_times_x_different_immunization_strategies')
    print('\nSaved plot: Prevalence (fraction of infected nodes) with different immunization strategies as a function of bin-times')
    plt.clf()

    print("This is how the visualization looks like")
    inf_prob = 0.1
    inf_node = 0
    infection_times = get_epidemic_infection_times(
        inf_node, inf_prob, movement_info)

    #for saving the simulation as a video:
    visualize_si(
        infection_times,
        save_fname="si_viz_example.mp4",
        tot_viz_time_in_seconds=10,
        fps=10
    )
