#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from ph_pipeline import patch_build

pb=patch_build.patch_build('train','covid_rgunet')
pb.patch_3dbuilder(model='densenet121', model_path='/jmain02/home/J2AD003/txk56/mxm87-txk56/SOFTWARE/classifier/ph_pipeline/Model/weights_patch_ph32v2_densenet121.h5', width=32, height=32, depth=1, threshold=0.8,split_data=1,folder="nii",classes=6)

