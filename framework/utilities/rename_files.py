import os
path = 'Database/Family/40_to_100/all_dictionaries/'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.pkl'])))