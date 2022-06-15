import os

def cleanOneFamilyData(dataset_collection_folder):
    for subdir_name in os.listdir(dataset_collection_folder):
        subdir_path = os.path.join(dataset_collection_folder, subdir_name)
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if int(file_name.split('_')[0]) < 0:
                os.remove(file_path)
    return

cleanOneFamilyData('./persons')
