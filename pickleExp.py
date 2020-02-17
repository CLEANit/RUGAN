import numpy as np

import pickle

datafolder = './data/'

Ts = [0,2,4,6,8,10,12,14,16,18,20,24,26,28,30,32]
dopings = [0.04065004, 1.26791777, 3.74087229, 6.02810036, 7.96623677, 9.95879863, 11.4017542, 13.72206745, 15.8631948, 17.93418327, 19.70910948,  23.91076375,  25.77354837,  28.23699057, 29.95799697,31.15944951]
configs = np.empty((18778,12,12))
labels = np.empty((18778))
ind = 0
for i in range(len(Ts)):
    with open(datafolder + 'cold_D'+str(Ts[i])+ '.pkl','rb') as f:
        configs_list = pickle.load(f, encoding="latin1")

    singles = np.array(configs_list[0])
    up = np.array(configs_list[1])
    configs[ind:ind+len(up)] = up
    labels[ind:ind+len(up)] = dopings[i]
    ind += len(up)
    down = np.array(configs_list[2])
    configs[ind:ind+len(down)] = down
    labels[ind:ind+len(down)] = dopings[i]
    ind += len(down)


pFname = datafolder+ "configs"+".p"
with open(pFname, 'wb') as output_file:
    pickle.dump(configs,output_file)

pFname = datafolder+ "labels_reg"+".p"
with open(pFname, 'wb') as output_file:
    pickle.dump(labels,output_file)
