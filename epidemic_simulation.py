import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time
import os

# The locations and schedule of movements:
EVENT_FNAME = "data/events_US_air_traffic_GMT.txt"
# Network file of the nodes:
NETWORK = "data/aggregated_US_air_traffic_network_undir.edg"
# Locations and names for the nodes in the visualization:
AIRPORT_INFO_CSV_FNAME = 'data/US_airport_id_info.csv'
# Background image for the animation:
BACKGROUND_IMAGE = 'data/US_air_bg.png'


def import_data(file_path):
    event_fname = file_path
    assert os.path.exists(
        event_fname), 'File {} could not be found'.format(event_fname)

    movement_info = np.genfromtxt(
        event_fname,
        delimiter=' ',
        dtype=None,
        names=True
    )
    return movement_info


def get_epidemic_infection_times(seed_node, inf_prob, movement_info):
    #start, end, start_time, end_time, duration = movement_info['Source'], movement_info['Destination'], movement_info['StartTime'], movement_info['EndTime'], movement_info['Duration']
    all_the_nodes = set(movement_info['Source']).union(
        set(movement_info['Destination']))
    infected_nodes = set([seed_node])
    inf_time = dict.fromkeys(range(len(all_the_nodes)))
    # Seed infection:
    inf_time[seed_node] = min(movement_info['StartTime']) - 1

    # The list is sorted by the time node touches the other node (EndTime, 4th element in the row/list)
    for mov in sorted(movement_info, key=lambda x: x[3]):
        # Checking
        #   1. If the Source is infected at all
        #   2. If infected before the StartTime of the movement
        #   3. If the infection is spreding with the movement, so if
        #       probability is less than Infection probability (inf_prob)
        if inf_time[mov['Source']] is None or inf_time[mov['Source']] >= mov['StartTime']:
            pass
        elif random.uniform(0, 1) > inf_prob:
            pass
        elif inf_time[mov['Destination']] is None or inf_time[mov['Destination']] > mov['EndTime']:
            inf_time[mov['Destination']] = mov['EndTime']
            infected_nodes.add(mov['Destination'])
            if len(all_the_nodes.difference(infected_nodes)) == 0:
                break
    for key, value in inf_time.items():
        if value is None:
            inf_time[key] = 10000000000
    infection_times_list = []
    for key in sorted(inf_time.keys()):
        infection_times_list.append(inf_time[key])

    return infection_times_list


def get_epidemic_infection_times_np_array(seed_node, inf_prob, movement_info_np_array, immunized_nodes_list):
    mov_inf = movement_info_np_array
    all_the_nodes = set(mov_inf[:, 1:2].reshape(1, len(mov_inf))[0]).union(
        set(mov_inf[:, :1].reshape(1, len(mov_inf))[0]))
    infected_nodes = set([seed_node])
    inf_time = dict.fromkeys(all_the_nodes)
    # Seed infection:
    inf_time[seed_node] = min(mov_inf[:, 2:3].reshape(1, len(mov_inf))[0]) - 1

    # The list is sorted by the time node touches the other node (EndTime, 4th element in the row/list)
    for mov in sorted(mov_inf, key=lambda x: x[3]):
        # Checking
        #   1. If the Source is infected at all
        #   2. If infected before the StartTime of the movement
        #   3. If the infection is spreding with the movement, so if
        #       probability is less than Infection probability (inf_prob)
        if inf_time[mov[0]] is None or inf_time[mov[0]] >= mov[2]:
            pass
        elif mov[0] in immunized_nodes_list:
            pass
        elif mov[1] in immunized_nodes_list:
            pass
        elif random.uniform(0, 1) > inf_prob:
            pass
        elif inf_time[mov[1]] is None or inf_time[mov[1]] > mov[3]:
            inf_time[mov[1]] = mov[3]
            infected_nodes.add(mov[1])
            if len(all_the_nodes.difference(infected_nodes)) == 0:
                break
    for key, value in inf_time.items():
        if value is None:
            inf_time[key] = 10000000000
    infection_times_list = []
    for key in sorted(inf_time.keys()):
        infection_times_list.append(inf_time[key])
    return infection_times_list
