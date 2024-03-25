
#Import the NWBHDF5IO module of the pynwb library
from pynwb import NWBHDF5IO
#Save the local file path to the NWB file
filepath = "sub-P11HMH_ses-20061101_ecephys+image.nwb"
#Open the NWB file in read mode
with NWBHDF5IO(filepath, mode='r') as io:
    #Get the contents of the NWB file
    nwb_file = io.read()
    #Get the data relevent to the Stimulus Presentation of the experiment
    stimulus_presentation = nwb_file.stimulus["StimulusPresentation"]
    #Get the stimulus data
    all_stimulus_data = stimulus_presentation.data[:]
    #Get the Stimulus onset times
    stim_on_times = stimulus_presentation.timestamps[:]
    #Get the continuous spiking information for each neuron/unit
    all_unit_spike_times = nwb_file.units["spike_times"]

#Following Tutorial: https://pynwb.readthedocs.io/en/stable/tutorials/general/plot_read_basics.html#sphx-glr-tutorials-general-plot-read-basics-py

#import libraries for data visualization
# will need to install matplotlib: conda install matplotlib
import matplotlib.pyplot as plt
# will need to install numpy: conda install numpy
import numpy as np

#install libraries for image saving
from PIL import Image

# used for reading NWD data that is in the HDF5 format
# will need to install pynwb: conda install -c conda-forge pynwb
from pynwb import NWBHDF5IO

# download the data from dandi
# will need to install dandi: conda install -c conda-forge dandi
#from dandi.download import download
#data_link = "https://api.dandiarchive.org/api/assets/0f57f0b0-f021-42bb-8eaa-56cd482e2a29/download/"
#data_pairing = "."
#download(data_link, data_pairing)

#open the file in read mode
filepath = "sub-P11HMH_ses-20061101_ecephys+image.nwb"
with NWBHDF5IO(filepath, mode='r') as io:
    nwb_file = io.read()

    # print the stimulus data
    #print(nwb_file)
    BGR_image = nwb_file.stimulus["StimulusPresentation"].data[0]
    RGB_image = BGR_image[..., ::-1]
    image = Image.fromarray(RGB_image)
    image.save("Image1.png")
    plt.imshow(RGB_image, aspect="auto")
    plt.show()

    #get the dataframe
    units_df = nwb_file.units.to_dataframe()
    #print the dataframe head
    print(units_df.head())

    #print the 
    #print(nwb_file.units["spike_times"][0])

    #visualize spiking activity relative to stimulus onset
    before = 1.5  # in seconds
    after = 1.5
    partitions_between_whole_numbers = 10

    # Get the stimulus times for all stimuli
    stimulus_presentation = nwb_file.stimulus["StimulusPresentation"]
    all_stimulus_data = stimulus_presentation.data[:]
    stim_on_times = stimulus_presentation.timestamps[:]

    print("=====")
    stim_time_diff = []
    for i in range(len(stimulus_presentation.timestamps[:100])-1):
        stim_time_diff.append(stimulus_presentation.timestamps[i+1] - stimulus_presentation.timestamps[i])
        print(stimulus_presentation.timestamps[i+1] - stimulus_presentation.timestamps[i])
    print("=====")
    stim_time_diff = np.array(stim_time_diff)
    print(np.median(stim_time_diff))
    plt.hist(stim_time_diff[stim_time_diff<1000], 20)
    plt.xlabel("Time between stimuli (s)")
    plt.ylabel("Occurrence")
    plt.title("Occurrence vs. Time between stimuli (s)")
    plt.show()

    print("---")
    print(len(nwb_file.units["spike_times"]))
    print("---")

    num_units = len(nwb_file.units["spike_times"])
    for unit in range(num_units):
        unit_spike_times = nwb_file.units["spike_times"][unit]
        trial_spikes = []
        spiking_frequncy = np.zeros(int((before+after)*partitions_between_whole_numbers))
        for time in stim_on_times[:100]:

            # Compute spike times relative to stimulus onset
            aligned_spikes = unit_spike_times - time
            #print(aligned_spikes)
            # Keep only spike times in a given time window around the stimulus onset
            aligned_spikes = aligned_spikes[
                (-before <= aligned_spikes) & (aligned_spikes <= after)
            ]
            print("-------")
            print(aligned_spikes)
            pre_count = len(aligned_spikes[aligned_spikes <= 0])/1.5
            post_count = len(aligned_spikes[aligned_spikes>0])/1.5
            print(pre_count, "|||", post_count)
            print("-------")

            trial_spikes.append(aligned_spikes)
        
        for spike in np.hstack(trial_spikes):
            index = int((spike+before)*partitions_between_whole_numbers)
            spiking_frequncy[index] += 1

        fig, axs = plt.subplots(2, 1, sharex="all")
        plt.xlabel("time (s)")
        axs[0].eventplot(trial_spikes)

        axs[0].set_ylabel("trial")
        axs[0].set_title("unit {}".format(unit))
        axs[0].axvline(0, color=[0.5, 0.5, 0.5])

        axs[1].hist(np.hstack(trial_spikes), int((spike+before)*partitions_between_whole_numbers))
        axs[1].axvline(0, color=[0.5, 0.5, 0.5])
        plt.show()
