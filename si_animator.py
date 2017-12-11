import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os

from epidemic_simulation import get_epidemic_infection_times, import_movement_data


EVENT_FNAME = "events_US_air_traffic_GMT.txt"
AIRPORT_INFO_CSV_FNAME = 'US_airport_id_info.csv'
BACKGROUND_IMAGE = 'US_air_bg.png'

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

if __name__ == "__main__":
    
    movement_info = import_movement_data(EVENT_FNAME)
    bin_times = np.linspace(min(movement_info['StartTime']), max(
        movement_info['EndTime']), 1000)

    ## Average infection times for different infection probabilities:
    n_iter = 10
    inf_probabilities = [0.01, 0.05, 0.1, 0.5, 1.0]
    inf_node = 0

    for inf_prob in inf_probabilities:
        print('Infection probability:', inf_prob)
        average_bin_array = average_infection_times(
            inf_node, inf_prob, movement_info, n_iter)
        plt.plot(bin_times, average_bin_array)
    plt.show()
    plt.clf()

    ## Average infection times for different infection starting nodes:
    n_iter = 10
    inf_prob = 0.1
    infected_nodes = [0, 4, 41, 100, 200]

    for inf_node in first_infected_nodes:
        print('First infected node:', inf_node)
        average_bin_array = average_infection_times(
            inf_node, inf_prob, movement_info, n_iter)
        plt.plot(bin_times, average_bin_array)
    plt.show()
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
