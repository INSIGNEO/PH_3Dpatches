#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import patch_build

# network info
model='densenet121'
moel_path= '/jmain02/home/J2AD003/txk56/mxm87-txk56/SOFTWARE/classifier/ph_pipeline/Model/weights_patch_ph32v2_densenet121.h5'

# 3D patches size shape
width=32 
height=32 
depth=1

threshold=0.8 # ratio of non zero pixels
split_data=1 # the patient number of analysis
folder='nii' # folder to store the results
classes=6.  # number of classes

pb=patch_build.patch_build('train','covid_rgunet')
pb.patch_3dbuilder(model, model_path, width, height, depth, threshold, split_data, folder, classes)

#P.S. to run the framework need to use the .config file so: python3 3D_patch_framework_run.py train_patch.config
