import os
path = 'Database/Family/161_to_220/all_dictionaries/'
path2 = 'Database/Family/161_to_220/all_dictionaries2/'
files = os.listdir(path)


for index, file in enumerate(files):
    # if index > 95:
        os.rename(os.path.join(path, file), os.path.join(path2, ''.join([str(index), '.pkl'])))