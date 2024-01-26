import sys
import numpy as np
import h5py
from os.path import basename, splitext

if sys.argv[1] == '-':
    stream = sys.stdin
else:
    stream = open("/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/streams/"+sys.argv[1]+".stream", 'r')

PixelResolution = 1 / (75 * 1e-3)

reading_geometry = False
reading_chunks = False
reading_peaks = False
is_a_hit = False
n_hits = 0
max_fs = -1
max_ss = -1
max_num_peaks=0
max_num_hits=1000
file_index=0
file_label= splitext(basename(sys.argv[1]))[0]


def initialize_collecting_hits():
    peakXPosRaw_agg = []
    peakYPosOriginal_agg = []
    peakXPosOriginal_agg = []
    peakYPosRaw_agg = []
    peaknPeak_agg = []
    peakTotalIntensity_agg = []
    peak_list = {"nPeak": 0,"peakTotalIntensity":[],"peakXPosRaw": [], "peakXPosOriginal": [],"peakYPosOriginal": [],"peakYPosRaw": []}
    file_name_list=[]
    event_list=[]
    shift_vertical_mm_list=[]
    
    return  peakXPosOriginal_agg, peakXPosRaw_agg, peakYPosOriginal_agg,peakYPosRaw_agg, peaknPeak_agg, peakTotalIntensity_agg, peak_list, file_name_list, event_list, shift_vertical_mm_list

def write_agg_file(file_label: str, file_index: int, n_hits:int):
    f = h5py.File(f"/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/hits_v8/{file_label}-{file_index}-agg_hit.h5", 'w')
    f.create_dataset('/entry/data/powder', data=powder[:max_ss+1,:max_fs+1])
    f.create_dataset('/entry/data/data', data=agg_hits[:n_hits,:,:], compression="gzip")
    f.create_dataset('/entry/data/peakXPosRaw', data=peakXPosRaw)
    f.create_dataset('/entry/data/peakXPosOriginal', data=peakYPosOriginal)
    f.create_dataset('/entry/data/peakYPosOriginal', data=peakYPosOriginal)
    f.create_dataset('/entry/data/peakYPosRaw', data=peakYPosRaw)
    f.create_dataset('/entry/data/nPeaks', data=peaknPeak)
    f.create_dataset('/entry/data/peakTotalIntensity', data=peakTotalIntensity)
    f.create_dataset('/entry/data/file_id', data=file_name_list)
    f.create_dataset('/entry/data/event_id', data=event_list)
    f.create_dataset('/entry/data/shift_vertical_mm', data=shift_vertical_mm_list)
    f.close()


peakXPosOriginal_agg, peakXPosRaw_agg, peakYPosOriginal_agg,peakYPosRaw_agg, peaknPeak_agg, peakTotalIntensity_agg, peak_list, file_name_list, event_list, shift_vertical_mm_list = initialize_collecting_hits()

for count, line in enumerate(stream):
    print(line)
    if reading_chunks and n_hits<max_num_hits:
        if line.startswith('End of peak list'):
            reading_peaks = False
            if is_a_hit:
                if peak_list["nPeak"]>max_num_peaks:
                    max_num_peaks = peak_list["nPeak"]
                peakXPosRaw_agg.append(peak_list["peakXPosRaw"])
                peakXPosOriginal_agg.append(peak_list["peakXPosOriginal"])
                peakYPosOriginal_agg.append(peak_list["peakYPosOriginal"])
                peakYPosRaw_agg.append(peak_list["peakYPosRaw"])
                peaknPeak_agg.append(peak_list["nPeak"])
                peakTotalIntensity_agg.append(peak_list["peakTotalIntensity"])
                peak_list = {
                "nPeak": 0,
                "peakTotalIntensity":[],
                "peakXPosRaw": [],
                "peakXPosOriginal": [],
                "peakYPosOriginal": [],
                "peakYPosRaw": []
                }
                file_name_list.append(file_name)
                event_list.append(event)
                shift_vertical_mm_list.append(shift_vertical_mm)
                f = h5py.File(file_name, 'r')
                agg_hits[n_hits,:,:] = np.array(f['entry/data/data'][event])
                f.close()
                n_hits += 1
                if n_hits==max_num_hits:
                    print(f"Saving {file_label}-{file_index}-agg_hit.h5")
                    peakXPosRaw=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
                    peakXPosOriginal=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
                    peakYPosOriginal=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
                    peakYPosRaw=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
                    peakTotalIntensity=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
                    peaknPeak=np.zeros((n_hits,), dtype=np.int64)
                    for i in range(n_hits):
                        peaknPeak[i]=peaknPeak_agg[i]
                        peakXPosRaw[i,:peaknPeak_agg[i]]=peakXPosRaw_agg[i]
                        peakXPosOriginal[i,:peaknPeak_agg[i]]=peakXPosOriginal_agg[i]
                        peakYPosOriginal[i,:peaknPeak_agg[i]]=peakYPosOriginal_agg[i]
                        peakYPosRaw[i,:peaknPeak_agg[i]]=peakYPosRaw_agg[i]
                        peakTotalIntensity[i,:peaknPeak_agg[i]]=peakTotalIntensity_agg[i]
                    write_agg_file(file_label, file_index, n_hits)
                    file_index+=1
                    n_hits=0
                    peakXPosOriginal_agg, peakXPosRaw_agg,peakYPosOriginal_agg,peakYPosRaw_agg, peaknPeak_agg, peakTotalIntensity_agg, peak_list, file_name_list, event_list, shift_vertical_mm_list = initialize_collecting_hits()
        elif line.startswith('  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel'):
            reading_peaks = True
        elif reading_peaks and is_a_hit:
            fs, ss, dump, intensity = [float(i) for i in line.split()[:4]]
            peak_y_pos = ss - 0.5
            peak_y_pos_original = ss - 0.5
            peak_x_pos = fs - shift_vertical_px - 0.5
            peak_x_pos_original = fs - 0.5
            if peak_x_pos <= max_fs:
                peak_list["peakXPosRaw"].append(peak_x_pos)
                peak_list["peakXPosOriginal"].append(peak_x_pos_original)
                peak_list["peakYPosOriginal"].append(peak_y_pos_original)
                peak_list["peakYPosRaw"].append(peak_y_pos)
                peak_list["peakTotalIntensity"].append(intensity)
                peak_list["nPeak"] += 1
                powder[int(round(peak_y_pos,0)), int(round(peak_x_pos,0))] += intensity   
        elif line.startswith('hit = 1'):
            is_a_hit = True 
        elif line.startswith('hit = 0'):
            is_a_hit = False
        elif line.split(': ')[0]=='Image filename':
                file_name = line.split(': ')[-1][:-1]
        elif line.split(': ')[0]=='Event':
            event=int(line.split(': //')[-1])
            print(file_name, event)
            f = h5py.File(file_name, 'r')
            shift_vertical_mm = -1* float(f['shift_vertical_mm'][event])
            shift_vertical_px = shift_vertical_mm * PixelResolution
            f.close()
    elif line.startswith('----- End geometry file -----'):
        print("starting")
        reading_geometry = False
        end_of_geometry_file_line_in_stream=count+1
        if file_index==0:
            powder = np.zeros((max_ss + 1, max_fs + 1))
        agg_hits = np.zeros((max_num_hits, max_ss + 1, max_fs + 1))
    elif reading_geometry:
        try:
            par, val = line.split('=')
            if par.split('/')[-1].strip() == 'max_fs' and int(val) > max_fs:
                max_fs = int(val)
            elif par.split('/')[-1].strip() == 'max_ss' and int(val) > max_ss:
                max_ss = int(val)
        except ValueError:
            pass
    elif line.startswith('----- Begin geometry file -----'):
        reading_geometry = True
    elif line.startswith('----- Begin chunk -----'):
        reading_chunks = True


print(f"Saving {file_label}-{file_index}-agg_hit.h5")
peakXPosRaw=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
peakXPosOriginal=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
peakYPosOriginal=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
peakYPosRaw=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
peakTotalIntensity=np.zeros((n_hits, max_num_peaks), dtype=np.float32)
peaknPeak=np.zeros((n_hits,), dtype=np.int64)

for i in range(n_hits):
    peaknPeak[i]=peaknPeak_agg[i]
    peakXPosRaw[i,:peaknPeak_agg[i]]=peakXPosRaw_agg[i]
    peakXPosOriginal[i,:peaknPeak_agg[i]]=peakXPosOriginal_agg[i]
    peakYPosOriginal[i,:peaknPeak_agg[i]]=peakYPosOriginal_agg[i]
    peakYPosRaw[i,:peaknPeak_agg[i]]=peakYPosRaw_agg[i]
    peakTotalIntensity[i,:peaknPeak_agg[i]]=peakTotalIntensity_agg[i]

write_agg_file(file_label, file_index, n_hits)
