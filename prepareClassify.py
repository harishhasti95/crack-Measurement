import os, shutil
import glob
from pathlib import Path

for root, directories, filenames in os.walk('data'): 
    for filename in filenames:  
        temp_path = os.path.join(os.getcwd(), os.path.join(root,filename))
        if 'Cracked' in temp_path:
            shutil.copy2(temp_path, 'dataClassification/Cracked/')
        elif 'Non-cracked' in temp_path:
            shutil.copy2(temp_path, 'dataClassification/Non-cracked/')