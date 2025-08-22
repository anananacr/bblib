adu_per_eV = 1.0
clen = 50.0e-3

dim0 = %
dim1 = ss
dim2 = fs
data = /data/data

mask_file = /Users/minoru/Documents/ana/scripts/beambusters/docs/example/mask.h5
mask = /data/data
mask_good = 1
mask_bad = 0

; Upper panel
0/min_fs = 0
0/max_fs = 1023
0/min_ss = 512
0/max_ss = 1023
0/corner_x = -512.00
0/corner_y = 10.00
0/fs = x
0/ss = y
0/res = 13333.3  ; 75 micron pixel size

; Lower panel
1/min_fs = 0
1/max_fs = 1023
1/min_ss = 0
1/max_ss = 511
1/corner_x = -512.00
1/corner_y = -522.00
1/fs = x
1/ss = y
1/res = 13333.3  ; 75 micron pixel size
