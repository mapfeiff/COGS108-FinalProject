#import libraries for data visualization
# will need to install matplotlib: conda install matplotlib
import matplotlib.pyplot as plt
# will need to install numpy: conda install numpy
import numpy as np

#install libraries for image saving
from PIL import Image

#import os library for creating folders
import os

# used for reading NWD data that is in the HDF5 format
# will need to install pynwb: conda install -c conda-forge pynwb
from pynwb import NWBHDF5IO

def add_spike_difference_data(unit_idx, img_path, spike_difference_info, data_mode):
    #case for adding training data
    if(data_mode == "train"):
        #get the csv path
        unit_csv_path = f"data/train/unit{unit_idx}.csv"
        #write header to csv file if it has not yet been created
        if(not os.path.isfile(unit_csv_path)):
            with open(unit_csv_path, 'w') as file:
                file.write("image,label,\n")
        #add data to csv file
        with open(unit_csv_path, 'a') as file:
            file.write(f"{img_path},{spike_difference_info},\n")

    #case for adding validation data
    elif(data_mode == "validation"):
        #get the csv path
        unit_csv_path = f"data/validation/unit{unit_idx}.csv"
        #write header to csv file if it has not yet been created
        if(not os.path.isfile(unit_csv_path)):
            with open(unit_csv_path, 'w') as file:
                file.write("image,label,\n")
        #add data to csv file
        with open(unit_csv_path, 'a') as file:
            file.write(f"{img_path},{spike_difference_info},\n")

    #error case
    else:
        raise(Exception("Invalid data mode passed"))
    
def get_spike_per_sec_difference_info(nwb_file, unit_idx, stim_idx, lower_bound, upper_bound):
    #Get the spiking info
    spike_times = nwb_file.units["spike_times"][unit_idx]

    #Get stim onset time
    stim_onset_time = nwb_file.stimulus["StimulusPresentation"].timestamps[stim_idx]

    #compute the offset spiking times
    spike_times_offset = spike_times - stim_onset_time
    
    #pre-onset spiking info
    pre_onset_spikes = spike_times_offset[(-lower_bound <= spike_times_offset) & (spike_times_offset <= 0)]
    num_pre_onset_spikes = len(pre_onset_spikes)
    num_pre_onset_spikes_per_sec = num_pre_onset_spikes/lower_bound
    
    #post-onset spiking info
    post_onset_spikes = spike_times_offset[(0 < spike_times_offset) & (spike_times_offset <= upper_bound)]
    num_post_onset_spikes = len(post_onset_spikes)
    num_post_onset_spikes_per_sec = num_post_onset_spikes/upper_bound

    #compute the difference in spiking info (difference = post - pre)
    spike_per_sec_differece = num_post_onset_spikes_per_sec - num_pre_onset_spikes_per_sec
    return(spike_per_sec_differece)

#open the file in read mode
filepath = "sub-P11HMH_ses-20061101_ecephys+image.nwb"
with NWBHDF5IO(filepath, mode='r') as io:
    #read the data in the NWB file
    nwb_file = io.read()

    #save all the data corresponding to the novel images to the training or validation data set
    total_images = 100 #number of novel images presented
    validation_divisor = 10 #determines number of validation items: total_images/validation_divisor
    for stim_idx in range(total_images):
        #Get current stimulus (image)
        BGR_image = nwb_file.stimulus["StimulusPresentation"].data[stim_idx]
        RGB_image = BGR_image[..., ::-1]
        image = Image.fromarray(RGB_image)

        #save validation data
        if(stim_idx % validation_divisor == 0):
            #save image stimulus
            image_num = stim_idx//validation_divisor
            img_path = f"data/validation/images/image{image_num}.png"
            image.save(img_path)
            #See how each unit reacted differently to the stimulus
            num_units = 15
            lower_bound = 1.5 #sec before offset
            upper_bound = 1.5 #sec after offset
            for unit_idx in range(num_units):
                #compute the spiking difference info
                spike_difference_info = get_spike_per_sec_difference_info(nwb_file, unit_idx, stim_idx, lower_bound, upper_bound)    
                #add the spiking info to the validation data
                add_spike_difference_data(unit_idx, img_path, spike_difference_info, "validation")

        #save training data
        else:
            #save image stimulus
            image_num = stim_idx - (stim_idx//validation_divisor)
            img_path = f"data/train/images/image{image_num}.png"
            image.save(img_path)
            #See how each unit reacted differently to the stimulus
            num_units = 15
            lower_bound = 1.5 #sec before offset
            upper_bound = 1.5 #sec after offset
            for unit_idx in range(num_units):
                #compute the spiking difference info
                spike_difference_info = get_spike_per_sec_difference_info(nwb_file, unit_idx, stim_idx, lower_bound, upper_bound)
                #add the spiking info to the train data
                add_spike_difference_data(unit_idx, img_path, spike_difference_info, "train")
    
    #Generate the data for each neuron recording
    #num_units = 15
    #for unit_num in range(len(num_units)):