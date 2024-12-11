import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def main():
    d_25_r_200_file_paths = ["forage_0.25_decay_200_range_trial1.csv",
                            "forage_0.25_decay_200_range_trial2.csv",
                            "forage_0.25_decay_200_range_trial3.csv",
                            "forage_0.25_decay_200_range_trial4.csv",
                            "forage_0.25_decay_200_range_trial5.csv"
                            ]

    d_25_r_300_file_paths = ["forage_0.25_decay_300_range_trial1.csv",
                            "forage_0.25_decay_300_range_trial2.csv",
                            "forage_0.25_decay_300_range_trial3.csv",
                            "forage_0.25_decay_300_range_trial4.csv",
                            "forage_0.25_decay_300_range_trial5.csv"
                            ]

    d_25_r_400_file_paths = ["forage_0.25_decay_400_range_trial1.csv",
                            "forage_0.25_decay_400_range_trial2.csv",
                            "forage_0.25_decay_400_range_trial3.csv",
                            "forage_0.25_decay_400_range_trial4.csv",
                            "forage_0.25_decay_400_range_trial5.csv"
                            ]

    d_50_r_200_file_paths = ["forage_0.50_decay_200_range_trial1.csv",
                            "forage_0.50_decay_200_range_trial2.csv",
                            "forage_0.50_decay_200_range_trial3.csv",
                            "forage_0.50_decay_200_range_trial4.csv",
                            "forage_0.50_decay_200_range_trial5.csv"
                            ]

    d_50_r_300_file_paths = ["forage_0.50_decay_300_range_trial1.csv",
                            "forage_0.50_decay_300_range_trial2.csv",
                            "forage_0.50_decay_300_range_trial3.csv",
                            "forage_0.50_decay_300_range_trial4.csv",
                            "forage_0.50_decay_300_range_trial5.csv"
                            ]

    d_50_r_400_file_paths = ["forage_0.50_decay_400_range_trial1.csv",
                            "forage_0.50_decay_400_range_trial2.csv",
                            "forage_0.50_decay_400_range_trial3.csv",
                            "forage_0.50_decay_400_range_trial4.csv",
                            "forage_0.50_decay_400_range_trial5.csv"
                            ]


    X_d_25_r_200 = get_data_from_trials(d_25_r_200_file_paths)
    X_d_25_r_300 = get_data_from_trials(d_25_r_300_file_paths)
    X_d_25_r_400 = get_data_from_trials(d_25_r_400_file_paths)
    X_d_50_r_200 = get_data_from_trials(d_50_r_200_file_paths)
    X_d_50_r_300 = get_data_from_trials(d_50_r_300_file_paths)
    X_d_50_r_400 = get_data_from_trials(d_50_r_400_file_paths)

    X_avg_d_25_r_200 = np.mean(X_d_25_r_200, axis=0)
    X_avg_d_25_r_300 = np.mean(X_d_25_r_300, axis=0)
    X_avg_d_25_r_400 = np.mean(X_d_25_r_400, axis=0)
    X_avg_d_50_r_200 = np.mean(X_d_50_r_200, axis=0)
    X_avg_d_50_r_300 = np.mean(X_d_50_r_300, axis=0)
    X_avg_d_50_r_400 = np.mean(X_d_50_r_400, axis=0)

    X_averages_per_experiment = np.array([X_avg_d_25_r_200,
                                          X_avg_d_25_r_300,
                                          X_avg_d_25_r_400,
                                          X_avg_d_50_r_200,
                                          X_avg_d_50_r_300,
                                          X_avg_d_50_r_400
                                          ])
    experiment_legends = ["decay: 0.25, range: 200",
                          "decay: 0.25, range: 300",
                          "decay: 0.25, range: 400",
                          "decay: 0.50, range: 200",
                          "decay: 0.50, range: 300",
                          "decay: 0.50, range: 400"]

    #data: [foraged, num random steps, resources discovered, resources foraged, resources depleted]
    cmap = cm.get_cmap('tab10')
    markers = ['o', 's', '^', 'v', '+', '.']
    
    make_scatter_plot(X=X_averages_per_experiment, y_index=0, colors=cmap.colors, title="Resource Units Collected", x_label="time steps", y_label="resource units in nest", legend_labels=experiment_legends, markers=markers)
    make_scatter_plot(X=X_averages_per_experiment, y_index=1, colors=cmap.colors, title="Exploratory Steps", x_label="time steps", y_label="number of random steps taken", legend_labels=experiment_legends, markers=markers)
    make_scatter_plot(X=X_averages_per_experiment, y_index=2, colors=cmap.colors, title="Resources Discovered", x_label="time steps", y_label="number of resources discovered", legend_labels=experiment_legends, markers=markers)
    make_scatter_plot(X=X_averages_per_experiment, y_index=3, colors=cmap.colors, title="Resources Foraged", x_label="time steps", y_label="number of foraged resources", legend_labels=experiment_legends, markers=markers)
    make_scatter_plot(X=X_averages_per_experiment, y_index=4, colors=cmap.colors, title="Resources Depleted", x_label="time steps", y_label="number of resources that were depleted", legend_labels=experiment_legends, markers=markers)

def get_data_from_trials(trial_file_names):
    trials_data = []
    for trial_file in trial_file_names:
        trials_data.append(np.genfromtxt(trial_file, delimiter=',')[::50]) #reduce array so keeps every 50 data points
    return np.array(trials_data)

def make_scatter_plot(X, y_index, colors, title, x_label, y_label, legend_labels, markers):
    x = np.arange(start=1, stop=len(X[0])+1, step=1)
    for i, y in enumerate(X):
        plt.scatter(x, y[:,y_index], color=colors[i], marker=markers[i], label=legend_labels[i])
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__=="__main__":
    main()