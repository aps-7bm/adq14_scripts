'''Code to run a scan with the new ADQ14 EPICS support.

Alan Kastengren
Started June 15, 2021
'''
from pathlib import Path
import threading
import epics
import time
import numpy as np
import h5py

#Global variables
global_args = {'scan':'7bmb1:scan1',
                'handshake':'7bmb1:rad:get_data',
                'digitizer_state_set':'ADQ-07414:COM:STATEMACHINE:SETSTATE',
                'digitizer_state_get':'ADQ-07414:COM:STATEMACHINE:GETSTATE',
                'repetitions':32,
                'active_channels':[0,1]}


def get_ancillary_PVs(args):
    '''Makes a list of PVs to record.
    '''
    ancillary_PVs = []
    scan_readback = str(epics.caget(args['scan']+'.R1PV'))
    if scan_readback != 'time':
        ancillary_PVs.append(scan_readback)
    for i in range(1,71):
        detector_PV = str(epics.caget('{0:s}.D{1:02d}PV'.format(
                        args['scan'], i)))
        if detector_PV == "":
            continue
        if epics.caget(detector_PV, timeout = 0.2):
            ancillary_PVs.append(detector_PV)
    args['ancillary_PVs'] = ancillary_PVs
    print('Ancillary PVs set up')
    return 
            

def get_waveform_info(args):
    #Get sampling info from the digitizer
    args['sampling_info'] = {}
    args['sampling_info']['samples_per_waveform'] = int(epics.caget('ADQ-07414:COM:SAMPCNT'))
    args['sampling_info']['sampling_rate'] = float(epics.caget('ADQ-07414:INFO:SAMPRATEDEC'))
    args['sampling_info']['sample_skip'] = int(epics.caget('ADQ-07414:COM:SAMPSKIP-RB'))

    #Get triggering info from the digitizer
    args['trig_info'] = {}
    args['trig_info']['pretrigger_samples'] = int(epics.caget('ADQ-07414:COM:PRETRIGSAMP-RB'))
    args['trig_info']['trigger_holdoff_samples'] = int(epics.caget('ADQ-07414:COM:TRIGHOLDOFFSAMP-RB'))
    args['trig_info']['trigger_mode'] = int(epics.caget('ADQ-07414:COM:TRIGMODE-RB'))
    if args['trig_info']['trigger_mode'] == 3:
        args['trig_info']['internal_trig_freq'] = float(epics.caget('ADQ-07414:COM:INTERNTRIGFREQ'))
    elif args['trig_info']['trigger_mode'] == 1:
        args['trig_info']['ext_trigger_edge'] = int(epics.caget('ADQ-07414:COM:EXTERNTRIGEDGE-RB'))
        args['trig_info']['ext_trigger_threshold_V'] = float(epics.caget('ADQ-07414:COM:EXTERNTRIGTHRESHOLD-RB'))
    
    #Get the PVs for the active channels
    args['waveform_PVs'] = []
    args['digitizer_meta_PVs'] = []
    for i in range(4):
        if i in args['active_channels']:
            args['waveform_PVs'].append('ADQ-07414:COM:CH{0:d}:DATA'.format(i))
            args['digitizer_meta_PVs'].append('ADQ-07414:COM:CH{0:d}:INPUTRANGE-RB'.format(i))
            args['digitizer_meta_PVs'].append('ADQ-07414:COM:CH{0:d}:DCBIAS-RB'.format(i))
    print('Waveform info set up')
    return
    

def setup_arrays(args):
    ancillary_arrays = {} 
    num_scan_points = int(epics.caget(args['scan']+'.NPTS'))
    for ancillary_PV in args['ancillary_PVs']:
        ancillary_arrays[ancillary_PV] = np.zeros(num_scan_points)
    args['ancillary_arrays'] = ancillary_arrays
    digitizer_meta_arrays = {} 
    for digitizer_meta_PV in args['digitizer_meta_PVs']:
        digitizer_meta_arrays[digitizer_meta_PV] = np.zeros(num_scan_points)
    args['digitizer_meta_arrays'] = digitizer_meta_arrays
    data_arrays = {}
    for data_array in args['waveform_PVs']:
        data_arrays[data_array] = np.zeros((num_scan_points, args['sampling_info']['samples_per_waveform'], args['repetitions']))
    args['data_arrays'] = data_arrays
    print('Finished setting up arrays')
    print('Scanning for {0:d} points'.format(num_scan_points))
    print('Doing {0:d} repetitions per point'.format(args['repetitions']))
    return


def update_arrays(args, current_point):
    print('Taking data at point {0:d}'.format(current_point))
    #Get the data PVs first, since they should be the most important
    take_repeated_data(args, current_point)
    #Get the ancillary data and add it to the arrays
    for a_pv in args['ancillary_arrays'].keys():
        args['ancillary_arrays'][a_pv][current_point] = epics.caget(a_pv)
    #Get the digitizer meta data
    for dm_pv in args['digitizer_meta_arrays'].keys():
        args['digitizer_meta_arrays'][dm_pv][current_point] = epics.caget(dm_pv)
 

def take_repeated_data(args, current_point):
    waveform_length = args['sampling_info']['samples_per_waveform']
    repetitions = args['repetitions']
    data_temp_arrays = []
    for i in range(repetitions):
        #If the digitizer isn't ready, wait for it
        while epics.caget(args['digitizer_state_get']) != 4:
            time.sleep(0.001)
        #Start the digitizer taking data
        epics.caput(args['digitizer_state_set'],7)
        time.sleep(0.01)
        #Wait for the digitizer to be done
        while epics.caget(args['digitizer_state_get']) != 4:
            time.sleep(0.001)
        #Read the data into the arrays
        for d_pv in args['waveform_PVs']:
            args['data_arrays'][d_pv][current_point,:,i] = epics.caget(d_pv)
    return


def scan_loop(args):
    scan_started = False
    start_time = time.time()
    handshake_pv = epics.PV(args['handshake'])
    while True:
        #If the scan hasn't started ...
        if epics.caget(args['scan'] + '.EXSC') == 0:
            scan_started = False
            time.sleep(0.01)
            continue
        #If we get here, the scan is running
        #At the start of a scan, set up the arrays
        if not scan_started:
            print('Starting scan')
            setup_arrays(args)
            current_point = 0
            scan_started = True
            start_time = time.time()
            time.sleep(0.01)
            continue
        #If the scan is running
        if handshake_pv.get() == 1:
            update_arrays(args, current_point)
            current_point += 1
            #Handle the end of the scan
            if (current_point == epics.caget(args['scan'] + '.NPTS') or
                epics.caget(args['scan'] + '.EXSC') ==  0):
                print('Scan finished: writing data')
                #threading.Thread(target=write_hdf, args=args)
                write_hdf(args)
                print('Total time = {0:7.3f} s'.format(time.time() - start_time))
            handshake_pv.put(0)
            time.sleep(0.1)
            continue
        

def write_hdf(args):
    '''Called to write out the saved arrays to disk.
    '''
    scan_num = int(epics.caget('7bmb1:saveData_scanNumber')) - 1
    file_name = 'Scan_{0:04d}.h5'.format(scan_num)
    print(Path.cwd())
    file_path = Path.cwd().parent.joinpath('data',file_name)
    with h5py.File(file_path, 'w') as hdf_file:
        #Write the data PVs
        for d_pv in args['data_arrays'].keys():
            hdf_file[d_pv] = args['data_arrays'][d_pv]
        for a_pv in args['ancillary_arrays'].keys():
            hdf_file[a_pv] = args['ancillary_arrays'][a_pv]
        #Get the digitizer meta data
        for dm_pv in args['digitizer_meta_arrays'].keys():
            hdf_file[dm_pv] = args['digitizer_meta_arrays'][dm_pv]


def main_loop():
    #Set up the ancillary PVs
    get_ancillary_PVs(global_args)
    get_waveform_info(global_args)
    scan_loop(global_args)


if __name__ == '__main__':
    main_loop()
