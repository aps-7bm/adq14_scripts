'''Code to view averages across repetitions for scan data.

Alan Kastengren, XSD, APS
Started June 16, 2021
'''
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.signal
import h5py
import ArrayBinModule as abm

data_path = Path.cwd().parent.joinpath('data')
PIN_PV = 'ADQ-07414:COM:CH0:DATA'
PIN_offset_PV = 'ADQ-07414:COM:CH0:DCBIAS-RB'
sample_skip_pv = 'ADQ-07414:COM:SAMPSKIP-RB'
y_PV = '7bmb1:aero:m1.RBV'
sample_skip = 8
norm_points = 50
trans_data = None
time_array = None
y = None

def print_hdf_contents(scan_num):
    file_path = data_path.joinpath('Scan_{0:04d}.h5'.format(scan_num))
    with h5py.File(file_path, 'r') as hdf_file:
        for i in hdf_file.keys():
            print(i)

def compute_average_trace(hdf_file, point_num):
    #sample_skip = hdf_file[sample_skip_pv][0]
    PIN_offset_ADC = hdf_file[PIN_offset_PV][0]
    print(PIN_offset_ADC)
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
    return orbit_binned_PIN, np.linspace(0, real_delta_t * (num_orbits - 1), num_orbits)


def norm_trace(data):
    return data / np.mean(data[:norm_points], dtype=np.float64)


def load_data(scan_num):
    file_path = data_path.joinpath('Scan_{0:04d}.h5'.format(scan_num))
    with h5py.File(file_path, 'r') as hdf_file:
        num_pts = hdf_file[PIN_PV].shape[0]
        global y
        y = hdf_file[y_PV][...]
        #Handle first point
        global time_array
        first_pt, time_array = compute_average_trace(hdf_file, 0)
        global trans_data
        trans_data = np.zeros((num_pts, first_pt.shape[0]))
        trans_data[0,:] = norm_trace(first_pt)
        for i in range(1,num_pts):
            trans_data[i,:] = norm_trace(compute_average_trace(hdf_file, i)[0])
            #plt.plot(time_array, full_data[i,:])
            #plt.show()
    return

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
    
def load_scanList(listName):
    file_path = Path.cwd().joinpath(listName)
    with open(file_path) as f:
        array = [line.split() for line in f]
    # print(array[0][0])
    load_data(int(array[0][0]))
    time_new = np.asarray(time_array)
    print(time_new)
    # view_all()
    
    return

    
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


def view_all():
    plt.figure()
    plt.contourf(time_array*1e6, y, trans_data,51)
    plt.colorbar()
    plt.xlabel(r'Time,$\mu$s')
    plt.ylabel('Y, mm')
    plt.title('Transmission vs. t and Y')
    plt.show()
    return 
