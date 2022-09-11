'''Code to view averages across repetitions for scan data.

Alan Kastengren, XSD, APS
Started June 16, 2021
'''
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.signal
from scipy.integrate import trapz
import h5py
import ArrayBinModule as abm

data_path = Path.cwd().parent.joinpath('data')
PIN_PV = 'ADQ-07414:COM:CH0:DATA'
BIM_PV = 'ADQ-07414:COM:CH1:DATA'
PIN_offset_PV = 'ADQ-07414:COM:CH0:DCBIAS-RB'
sample_skip_pv = 'ADQ-07414:COM:SAMPSKIP-RB'
y_PV = '7bmb1:aero:m2.RBV'
x_PV = '7bmb1:m25.RBV'
sample_skip = 8
norm_points = 50
trans_data = None
time_array = None
y = None
x = None

def print_hdf_contents(scan_num):
    file_path = data_path.joinpath('Scan_{0:04d}.h5'.format(scan_num))
    with h5py.File(file_path, 'r') as hdf_file:
        for i in hdf_file.keys():
            print(i)

def compute_average_trace(hdf_file, point_num, dark_offset):
    #sample_skip = hdf_file[sample_skip_pv][0]
    PIN_offset_ADC = hdf_file[PIN_offset_PV][0]
    delta_t = 2e-9 * sample_skip
    PIN_data = hdf_file[PIN_PV][point_num,:,:]
    av_PIN = np.mean(PIN_data,axis=1,dtype=np.float64)
    av_PIN -= PIN_offset_ADC
    dec_PIN, pulse_time, dummy = abm.fbin_signal_by_pulse(av_PIN,delta_t,repeat_num=24)
    #Bin by orbit
    num_orbits = dec_PIN.shape[0] // 24
    reshaped_PIN = dec_PIN[:num_orbits*24].reshape((num_orbits,24))
    orbit_binned_PIN = np.mean(reshaped_PIN, axis=1, dtype=np.float64)
    real_delta_t = pulse_time * 24
    if dark_offset:
        orbit_binned_PIN -= (dark_offset - PIN_offset_ADC)
    return orbit_binned_PIN, np.linspace(0, real_delta_t * (num_orbits - 1), num_orbits)


def norm_trace(data):
    return data / np.mean(data[:norm_points], dtype=np.float64)


def process_dark_scan(scan_num):
    file_path = data_path.joinpath('Scan_{0:04d}.h5'.format(scan_num))
    with h5py.File(file_path, 'r') as hdf_file:
        PIN_data = hdf_file[PIN_PV][...]
    return np.mean(PIN_data, dtype=np.float64)


def load_data(scan_num, dark_scan=3):
    file_path = data_path.joinpath('Scan_{0:04d}.h5'.format(scan_num))
    file_path_processed = data_path.joinpath('Scan_{0:04d}_Processed.h5'.format(scan_num))
    global y
    global time_array
    global trans_data
    #Check if data have already been loaded and processed
    if file_path_processed.exists():
        with h5py.File(file_path_processed, 'r') as hdf_processed:
            global trans_data
            trans_data = hdf_processed['Transmission'][...]
            y = hdf_processed[y_PV][...]
            global time_array
            time_array = hdf_processed['Time'][...]
            return
    with h5py.File(file_path, 'r') as hdf_file:
        num_pts = hdf_file[PIN_PV].shape[0]
        y = hdf_file[y_PV][...]
        if dark_scan:
            dark_offset = process_dark_scan(dark_scan)
        else:
            dark_offset = 0
        #Handle first point
        first_pt, time_array = compute_average_trace(hdf_file, 0, dark_offset)
        trans_data = np.zeros((num_pts, first_pt.shape[0]))
        trans_data[0,:] = norm_trace(first_pt)
        for i in range(1,num_pts):
            print('Loading data for y = {0:5.2f} mm'.format(y[i]))
            trans_data[i,:] = norm_trace(compute_average_trace(hdf_file, i, dark_offset)[0])
            #plt.plot(time_array, full_data[i,:])
            #plt.show()
        #Write the processed data file, except for the raw PIN and BIM data
        with h5py.File(file_path_processed, 'w') as hdf_processed:
            for k in hdf_file.keys():
                if k == PIN_PV or k == BIM_PV:
                    continue
                hdf_file.copy(k, hdf_processed)
            hdf_processed['Transmission'] = trans_data
            hdf_processed['Time'] = time_array
    return



def load_multiple_scans(scan_list):
    #Extract data from first scan
    global x, y, trans, time_array
    x, y, trans, time_array = extract_scan_data(scan_list[0])
    for i in scan_list[1:]:
        temp_x, temp_y, temp_trans, _ = extract_scan_data(i)
        x = np.concatenate((x, temp_x))
        y = np.concatenate((y, temp_y))
        trans = np.concatenate((trans, temp_trans))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom = 0.35)
    min_level = np.max(trans) - 0.75 * (np.max(trans) - np.min(trans))
    levs = np.linspace(min_level, np.max(trans), 50)
    p = ax.tricontourf(x, y, trans[:,0], levs, extend='both')
    fig.colorbar(p)
    plt.xlabel('X, mm')
    plt.ylabel('Y, mm')
    ax_slider = plt.axes([0.25, 0.05, 0.45, 0.03])
    allowed_times = time_array*1e6
    time_slider = Slider(ax_slider, r'Time, $\mu$s', np.min(allowed_times), np.max(allowed_times[:-1]), 0)

    def update(val):
        time_step = np.argmin(np.abs(time_array*1e6 - val))
        p = ax.tricontourf(x, y, trans[:,time_step], levs, extend='both')
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()
    return


def extract_scan_data(scan_num):
    file_path = data_path.joinpath('Scan_{0:04d}_Processed.h5'.format(scan_num))
    with h5py.File(file_path, 'r') as hdf_file:
        x = hdf_file[x_PV][...]
        y = hdf_file[y_PV][...]
        time = hdf_file['Time'][...]
        trans = hdf_file['Transmission'][...]
    return x, y, trans, time


def view_time_trace(point_num):
    plt.figure()
    plt.plot(time_array, trans_data[point_num,:], '.-')
    plt.xlabel('Time, s')
    plt.ylabel('Transmission')
    plt.title('Time trace for point #{0:d}'.format(point_num))
    plt.show()


def view_transverse(desired_time):
    plt.figure()
    time_step = np.argmin(np.abs(time_array - desired_time))
    plt.plot(y, trans_data[:,time_step])
    plt.xlabel('Y, mm')
    plt.ylabel('Transmission')
    plt.title('Transverse distribution t={0:7.4f} ms'.format(desired_time * 1e3), fontsize=12)
    plt.show()


def view_transverse_slider():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom = 0.35)
    line_plot, = plt.plot(y, trans_data[:,0], 'r.-')
    plt.ylim(np.min(trans_data),np.max(trans_data))
    plt.ylabel('Transmission')
    plt.xlabel('Y, mm')
    ax_slider = plt.axes([0.25, 0.05, 0.45, 0.03])
    allowed_times = time_array*1e6
    time_slider = Slider(ax_slider, r'Time, $\mu$s', np.min(allowed_times), np.max(allowed_times[:-1]), 0)

    def update(val):
        time_step = np.argmin(np.abs(time_array*1e6 - val))
        line_plot.set_ydata(trans_data[:,time_step])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()

    
def view_time_slider():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom = 0.35)
    line_plot, = plt.plot(time_array*1e6, trans_data[0,:], 'r.-')
    plt.ylim(np.min(trans_data),np.max(trans_data))
    plt.ylabel('Transmission')
    plt.xlabel(r'Time, $\mu$s ')
    ax_slider = plt.axes([0.25, 0.05, 0.45, 0.03])
    allowed_y = y
    y_slider = Slider(ax_slider, 'Y, mm', np.min(y), np.max(y[:-1]), np.min(y))

    def update(val):
        y_arg = np.argmin(np.abs(y - val))
        line_plot.set_ydata(trans_data[y_arg,:])
        fig.canvas.draw_idle()

    y_slider.on_changed(update)
    plt.show()


def view_TIM():
    plt.figure()
    TIM = trapz(-np.log(trans_data),y,axis=0)
    plt.plot(time_array, TIM, 'r.-')
    plt.ylabel('TIM')
    plt.xlabel(r'Time, $\mu$s ')
    plt.show()


def view_all():
    plt.figure()
    plt.contourf(time_array*1e6, y, trans_data,51)
    plt.colorbar()
    plt.xlabel(r'Time,$\mu$s')
    plt.ylabel('Y, mm')
    plt.title('Transmission vs. t and Y')
    plt.show()
    return 


def view_all_el():
    plt.figure()
    plt.contourf(time_array*1e6, y, -np.log(trans_data),51)
    plt.colorbar()
    plt.xlabel(r'Time,$\mu$s')
    plt.ylabel('Y, mm')
    plt.title('Ext. Lengths vs. t and Y')
    plt.show()
    return 
