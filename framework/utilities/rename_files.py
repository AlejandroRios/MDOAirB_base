import os
path = 'Database/Family/161_to_220/all_dictionaries/'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.plk'])))