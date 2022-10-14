from __future__ import division, print_function
import numpy as np
import sys
import torch
import os
import cv2
import math
import argparse
import logging
import tensorflow as tf
from keras import backend as K
from ph_pipeline import  create_net, regularization, run_model, config, datasetnet, store_model, handle_data, threed_rebuilder
from ph_pipeline.datasetnet import *
#from skimage.metrics import structural_similarity
import pylab 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
import time
from tensorflow.keras import utils
import os
from tensorflow.keras.utils import to_categorical
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, jaccard_score, auc, matthews_corrcoef, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d

class patch_build(object):

	def __init__ (self, rmn,mmn) :

		args = config.parse_arguments()
		self.param_reg=0.0001
		self.normalize_image=args.normalize
		self.roi_model_name=rmn
		self.main_model_name=mmn
		self.batch_size=args.batch_size
		self.bst=args.batch_size_test
		self.gancheckpoint='checkpoint'
		self.original_image_shape_roi=args.original_image_shape_roi
		self.original_image_shape_main=args.original_image_shape
		self.data_augm=args.data_augm
		self.data_augm_classic=args.data_augm_classic
		self.store_model_path=args.store_txt
		self.validation=args.validation	
		self.main_model=args.main_model
		self.STORE_PATH1=args.store_data_test
		self.STORE_PATH2=self.STORE_PATH1+'/ROI/test/'
		self.data_extention=args.data_extention
		self.counter_extention=args.counter_extention
		self.loss_type=args.loss_main

	def build_data_structure(self,clas=0,path='none'):
		if path!='none':
			directory=path+'/'+clas
		else:
			directory=self.STORE_PATH1+'/'+clas
		if not os.path.exists(directory):
			os.mkdir(directory)


	def patch_extract(self,class_list='off', patch_export='on',width=32,height=32, depth=1, labels=[0,1], threshold=0.75, tr_class=0.1,store_format='jpeg',split_data=1,folder="base_folder"):
		ο=1
		οο=1
		y_array=[]	
		self.build_data_structure(clas=folder,path=self.STORE_PATH2)
                #set the shape of roi analysis for the output lab
		print("Patch analysis pre-processing!!!!!")

		for i in create_dataset_itter(split=split_data,analysis='train',path_case='main'):
			Xn, X_total,Youtn, contour_mask = [],[],[],[]
			Xn.append(i[0])
			X_total.append(i[1])
			Youtn.append(i[2])
			contour_mask.append(i[3])
			storetxt=(i[4])
			itter=(i[5])
			X=np.array(Xn)
			Yout=np.array(Youtn)
			X=np.reshape(X,[X.shape[1],X.shape[2],X.shape[0]*X.shape[3]])
			print("Length of classification labels: ",len(labels))
			print("MAX value Y: ",np.max(Yout), "Min value Y: ",np.min(Yout))
			print("MAX value X: ",np.max(X), "Min value X: ",np.min(X))
			Y=np.reshape(Yout,[Yout.shape[1],Yout.shape[2],Yout.shape[0]*Yout.shape[3]])
			print(Y.shape,X.shape)
			height2, width2, _ = Y.shape
			height1, width1, _=X.shape
			if class_list=='on':
				tr=tr_class
				for i in range(Y.shape[2]):
					y=Y[:,:,i]
					clas=[] #multi-label
					countc=0
					label_size=[]
					#no multi-label casses only multi-class
					tr_max=int(tr*tr*height2*width2)
					for z in range(1,len(labels)):
						y1=y[:,:]
						occur_clas = np.count_nonzero((y1<=(z+1)) & (y1>z))
						label_size.append(occur_clas)
						# all the labels multi-label case
						if label_size[countc]>=tr_max:
							clas.append(z)	
							print("achieved class: ",z," image: ",i)
						countc=countc+1
					clas=np.array(clas)
					run=[]
					for io in range(len(clas)): 
						run.append(str(clas[io]))
					class_name=('_'.join(run))
					self.build_data_structure('class_'+class_name)
					x=(X[:,:,i]) #z-axis
					str3=self.STORE_PATH1 +'/class_'+class_name+ '/Image_%s_%s.%s' % (i,class_name,store_format)
					str4=self.STORE_PATH1 +'/class_'+class_name+ '/NII_Image_%s_%s.%s' % (i,class_name,"nii.gz")
					x1=np.squeeze(x)
					imgn=nib.Nifti1Image(x1, affine=np.eye(4))
					nib.save(imgn,str4)
					plt.imsave(str3, x1, format='jpeg',cmap = cm.gray)


			if patch_export=='on':
				split_gpu=1
				start=0
				totalX=X.shape[2]
				end=int(totalX/split_gpu)
				Y=np.reshape(Y,[1,height2,width2,totalX,1])
				X=np.reshape(X,[1,height1,width1,totalX,1])
				print("start the  patch extraction.")
				print("MAX value Y: ",np.max(Y), "Min value Y: ",np.min(Y))
				u=0
				shape= self.original_image_shape_main
				thr=1 
				xs=int(height*thr)
				ys=int(width*thr) 
				td=int(depth*thr)
				Xtotal, Xtot = [], []
				Xtotal=X[start:end]
				y_array=[]
				Ytotal, Ytot = [], []
				Ytotal=Y[start:end]
				start=end
				end=end+int(totalX/split_gpu)
				print(Xtotal.shape,Ytotal.shape)
				patch_size_h = [1, height, width, depth, 1] #channels]
				patch_size_w= [1, height, width, depth, 1] #classes]		
				Xtotal=tf.convert_to_tensor(Xtotal,dtype=tf.int32)
				Ytotal=tf.convert_to_tensor(Ytotal,dtype=tf.int32)
				x=tf.extract_volume_patches(input=Xtotal,ksizes=patch_size_h, strides=[1,ys,xs,td,1], padding='VALID')
				y1=tf.extract_volume_patches(input=Ytotal,ksizes=patch_size_w, strides=[1,ys,xs,td,1], padding='VALID')
				y=y1	
				number1=int(x.shape[2]*x.shape[1])
				number2=int(y.shape[2]*y.shape[1])
				number3=int(number1*x.shape[3])
				number4=int(number2*y.shape[3])
				print(x.shape,y.shape,number1, number2)
				xphs=tf.reshape(x,[x.shape[3]*x.shape[1]*x.shape[2],width,height,depth])
				yout=tf.reshape(y,[y.shape[3]*y.shape[1]*y.shape[2],width,height,depth])
				print(yout.shape,xphs.shape)	
				# Check if the threshold number of the patch is == with only one label if yes save if not discard
				y_save=[]
				x_save=[]
				label_size=[]
				print(number4,yout.shape[0])
				for i in range(0,number4):
					Xphs=(xphs[i,:,:,:])
					Yphs=yout[i,:,:,:] #[i:i+yout.shape[0],:,:,:]
					patch_window=np.array(Yphs)
					count=0
					z=1
					if (np.max(patch_window)<=0) and (np.min(patch_window)<=0):
						z_max=0
			#	print("Empty patch......")
					else:
						z_max=len(labels)
					#	print("RUN: ",i," MAX: ",np.max(Yphs)," MIN: ",np.min(Yphs),Yphs.shape)

					label_size=[]
					patch=np.squeeze(patch_window)
					while z<z_max:
						occure=0			
						occure = np.count_nonzero(((patch<=(z+1)) & (patch>z)))
						label_size.append(occure)
						thrpass=int(threshold*height*threshold*width*threshold*depth)
						if label_size[count]>=thrpass:
							print("non zero detected: ",label_size[count], " threshold to pass: ",int(thrpass))								
							x=np.array(Xphs,dtype='f')
							print("RUN: ",i," MAX: ",np.max(Yphs)," MIN: ",np.min(Yphs))#,x.shape)
							print("image detected in class ",z, "and saved")
							class_f='class_%s' %(z)
							class_f3=class_f+"/3D"
							new_fold=self.STORE_PATH2+folder+"/"
							self.build_data_structure(class_f,new_fold)
							self.build_data_structure(class_f3,new_fold)
							twd=0
							#x.astype(np.int8)
							#x=np.dtype(float) 
							str4=new_fold + class_f3+ '/X3d_%s__%s.%s' % (z,i,'nii.gz')
							imgnthree=nib.Nifti1Image(x, affine=np.eye(4))
							imgnthree.header.set_data_dtype(np.uint8)
							nib.save(imgnthree,str4)		
							while twd<depth:
								str2=new_fold + class_f+ '/X_%s_%s_%s.%s' % (z,twd,i,'jpg')           
								str3=new_fold + class_f+ '/X_%s_%s_%s.%s' % (z,twd,i,'nii.gz')
								xo=x[:,:,twd]
								imgn=nib.Nifti1Image(xo, affine=np.eye(4))
								imgn.header.set_data_dtype(np.uint8)
								nib.save(imgn,str3)
							
								plt.imsave(str2, xo, format='jpg', cmap = "gray")			
								twd=twd+1
						count=count+1
						if z>(np.max(patch)):
							z=100000
						else:
							z=z+1
			print('finish split step')						


	def patch_3dbuilder(self, batch=16,model='densenet121', model_path='/jmain02/home/J2AD003/txk56/mxm87-txk56/SOFTWARE/classifier/ph_pipeline/Model/weights_patch_ph32_densenet121.h5', width=32, height=32, depth=1, threshold=0.8,split_data=1,folder="3d",classes=11,crop='off',normalize='true'):

		ο=1
		οο=1
		y_array=[]
		channels=1
		self.build_data_structure(clas=folder,path=self.STORE_PATH1)
                #set the shape of roi analysis for the output lab
		print("Patch 3D builder processing!!!!!")
		Y_final=[]
		split_gpu=1
		for i in create_dataset_itter(split=split_data,analysis='train',path_case='main'):
			
			Xn, X_total,Youtn, contour_mask = [],[],[],[]
			Xn.append(i[0])
			X_total.append(i[1])
			Youtn.append(i[2])
			contour_mask.append(i[3])
			storetxt=(i[4])
			itter=(i[5])
			Xi=np.array(Xn)
			Yout=np.array(Youtn)
			Xi=np.reshape(Xi,[Xi.shape[1],Xi.shape[2],Xi.shape[0]*Xi.shape[3]])
			Y=np.reshape(Yout,[Yout.shape[1],Yout.shape[2],Yout.shape[0]*Yout.shape[3]])
			print(Y.shape,Xi.shape)
			X=Xi
			height2, width2, _ = Y.shape
			height1, width1, _=Xi.shape
			totalX=Xi.shape[2]
			start=0
			end=int(totalX/split_gpu)
			Y=np.reshape(Y,[1,height2,width2,totalX,1])
			Xi=np.reshape(Xi,[1,height1,width1,totalX,1])
			Y1=np.array(Y,dtype=np.uint8)
			Y1=np.where(Y1>=1,1,0)
			Y1=np.array(Y1,dtype=np.uint8)
			Ybase=np.reshape(Y1,[1,height1,width1,totalX,1])
			Y_patient_sl=np.zeros([totalX,height1,width1])
			Y_patient_un=np.zeros([totalX,height1,width1])
			#if crop=='off':
			#	X=Xi #no crop
			#else: 
			#	X=cv2.bitwise_and(Xi,Xi,mask=Y1)
			print("start the  3D patch extraction of patient.")
			print("MAX value Y binary: ",np.max(Y1), "Min value Y: ",np.min(Y1))
			print("MAX value Y test full: ",np.max(Y), "Min value Y: ",np.min(Y))
			thr=1
			xs=int(height*thr)
			ys=int(width*thr)            
			#load model
			cn=create_net.create_net(model)
			print("Load model from: ")
			file_store=model_path
			print(file_store)
			if height<32:
				ms=cn.net([], [], model ,32,channels,(classes),32)
			else:
				ms=cn.net([], [], model ,height,channels,(classes),width)
			model_structure=ms[0]
			rm=run_model.run_model('main')
			loss=rm.load_loss(self.loss_type)
			model_structure.compile(optimizer=rm.optimizer,loss=loss, metrics=rm.metrics_generator())
			model_structure.load_weights(file_store)#,by_name=True, skip_mismatch=True)
			rm=run_model.run_model(model)
			mask_counter=np.zeros((classes))
			total_mask=0
			zero_count=0
			Ys=np.reshape(Y,[height1,width1,totalX])
			Ys=self.class_transform(Ys)
			Xx=np.reshape(Xi,[height1,width1,totalX])
			max_pat=int(height/height1)
			Ysp=[]
			Ypat1=[]
			for testx in range(totalX):
			
				Xsk=Xx[:,:,testx]
				Ysk=Ys[:,:,testx]
				Xsk=np.reshape(Xsk,[height1,width1])
				Ysk=np.reshape(Ysk,[height1,width1])
				patch_size= [height,width] 
				sizex=int(height1/height)
				sizey=int(width1/width)
				Ypt=torch.tensor(Ysk)
				Xpt=torch.tensor(Xsk)
				Ypat=Ypt.unfold(0,height,xs).unfold(1,width,ys)
				Xpat=Xpt.unfold(0,height,xs).unfold(1,width,ys)
				y=np.array(Ypat)
				x=np.array(Xpat)
				
				print(testx,": 2d image processing of initial extraction patches ",x.shape," shape")
				#x=np.reshape(x,[sizex,sizey,height,width,1])
				#y=np.reshape(y,[sizex,sizey,height,width,1])
				number=x.shape[0]
				st=batch
				x_batch=[]
				index=[]
				position=[]
				position2=[]
				Ytest=np.array(y)
				#Ytest=self.class_transform(Ytest)#transform same space with GT classes task
				Yspb=(Ysk)
				one_count=0
				number4=x.shape[1]
				number3=x.shape[0]
				healthy_only=True
				for io in range(0,number3):	
					for i2 in range(0,number4):
						occure = np.count_nonzero(Ytest[io,i2,:,:])
						thrpass=int((threshold)*height*(threshold)*width*threshold)
						nonhealthy=np.where(Ytest[io,i2,:,:]>1)
						if len(nonhealthy[0])>0:#for normal use (no 5-slice validation) comment the unhealthy sequence
							healthy_only=False
						if (occure<thrpass or healthy_only==True):
							zero_count=zero_count+1
						else:
							one_count=one_count+1
							uniq=i2+io+testx
							xj=x[io,i2]
							xn=self.transf_jpg(xj,uniq)
							x_batch.append(xn)
							position.append(io)
							position2.append(i2)

				if one_count!=0:
					x_b=np.array(x_batch)
					if (one_count % batch)==0:
						batch1=int(one_count/batch)
						#print(one_count % batch,batch1)
					else:
						batch1=one_count
					print(x_b.shape)
					x_b=np.reshape(x_b, [x_b.shape[0],x_b.shape[1],x_b.shape[2],1])
					y_pred=[]
					if height<32:
						x_b=np.resize(x_b,(one_count,32,32,1))
						#print(x_b.shape)
					for i in range(0,one_count,batch1):
						x_test=ImageDataGenerator().flow(x=x_b[i:(i+batch1)], batch_size=batch1, shuffle=False )
						y_pred_keras = model_structure.predict(x_test)
						y_p=y_pred_keras
						print(y_p)
						y_pred.append(y_p)
					y_pred=np.array(y_pred)
					y_pred=np.reshape(y_pred,[y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2]])
					class_num=0

					for u in range(one_count):
						yp=y_pred[u]
						y_max=np.max(yp)
						yp=np.where(yp==y_max, 1.0,0.0 )
						index=np.where(yp==1.0)
						pos=position[u]
						pos2=position2[u]
						#print(y_max)
						if len(index[0])>1:
							#indo=np.where(index[0]==0,0,5) # healthy if not abnormal
							ind=0
							print(len(index[0]))
						else:
							ind=index[0]
						Ytest[pos,pos2]=np.where(Ytest[pos,pos2]>0,ind+1.0,0.0)
						class_num=int(ind)
						mask_counter[class_num]=mask_counter[class_num]+1	
						total_mask=total_mask+1
					

					#RECOSTRUCTION OF PATCHES
					print("recostruction of xy image...")
					Y2d=torch.tensor(Ytest)
					Y2d=Y2d.squeeze()
					unfold_shape = Y2d.size()
					Ypatches = Y2d.contiguous().view(-1, height,height)
					Ypatch=Ypatches.view(unfold_shape)
					output_h = unfold_shape[0] * unfold_shape[2]
					output_w = unfold_shape[1] * unfold_shape[3]
					Ypatd = Ypatch.permute(0, 2, 1, 3).contiguous()
					rexy=Ypatd.view(output_h, output_w)
					rexy=rexy.squeeze()
					rexy=np.array(rexy)
					Y_patient_sl[testx]=(rexy)
					Ypat1.append(rexy)

					
				
			Ys=np.reshape(Ys,[totalX,height1,width1])
			z_r=Y_patient_sl
			z_r=np.reshape(z_r,[totalX,height1,width1])
			#to undone the full comment all, fo one tap back the recostruction, comment theexclude healthy up and down
			Ysp=np.array(Ysp)
			Ypat1=np.array(Ypat1)
			print(Ysp.shape,Ypat1.shape)
			Ypat1=np.where(Ypat1==7,4,Ypat1)
			Ypat1=np.where(Ypat1==10,5,Ypat1)
			Ypat1=np.where(Ypat1==6,5,Ypat1)

########################################################################################################################
########################## METRICS EXTRACTION ########################
			Ys1=Ysp
			z_r1=Ypat1
			
			MSE = np.square(np.subtract(Ys1,z_r1)).mean()
			RMSE=math.sqrt(MSE)
			s1=np.array(Ys1)
			s2=np.array(z_r1)
			if s2.shape[0]!=0:
				s1=np.reshape(s1,[s1.shape[0]*s1.shape[1]*s1.shape[2]])#*s1.shape[3]*s1.shape[4]])
				s2=np.reshape(s2,[s2.shape[0]*s2.shape[1]*s2.shape[2]])#*s2.shape[3]*s2.shape[4]])	
				f1=f1_score(s1, s2, average='weighted')#reduce based on the number of max class to normalize)
				recall=recall_score(s1, s2, average='weighted')
				auc1=accuracy_score(s1, s2)
				jac=jaccard_score(s1, s2, average='weighted')
				pre=precision_score(s1, s2, average='weighted')
				hamm=distance.hamming(s1,s2)
				mcc=matthews_corrcoef(s1,s2)
				print("Root Mean Square Error: ", RMSE)
				print("F1-score score: ",f1)
				print("Recall: ",recall)
				print("Precision: ",pre)
				print("Accuracy: ",auc1)
				print("Jaccard score: ",jac)
				print("MCC: ",mcc)
				print("Hamming distances: ",hamm)
#######################################################################################################################
				print("Statistic profile of patient")
				print("MAX pred: ",np.max(z_r1))			
				print("MAX GT: ",np.max(Ys))
				Ysl1=(z_r1)
				Ysl2=np.transpose(Ysl1)
				Ysl2=np.fliplr(Ysl2)
				Ysl2=np.rot90(Ysl2)
				Ysl20=(Ysl2*256)
				Ys0=(Ys*256)#*(np.max(Ys)/classes)
				#Yslu0=256*Yslu1*6
				Y10=np.reshape(Ys0, [height1,width1,totalX])
				print(" Max of recostructed area: ",np.max(Ysl1))
				print(" Max of ground truth area: ",np.max(Ys0))
				print("final shape (after reshape) : ", Ysl20.shape)
				print("total patches: ",number4*totalX)
				print("total patches tested: ",total_mask)
				print("zero patches: ",zero_count)
				for dc in range(0,classes):
					print("total patches class : ",dc, " ",mask_counter[dc])
				print("he/she has the following clinical profile: ")

				for d in range(0,classes):
					print("Detected disease of class :",d)
					print( (mask_counter[d]/total_mask)*100,"%")
				totalYd=0
				clt=np.zeros(classes)
				

				#save patients
				new_fold=self.STORE_PATH1+folder+"/"
				str6=new_fold + '/Y3d_predic_grayscale_patient_%s.%s' % (itter,'nii.gz')
				imgnthree3=nib.Nifti1Image(Ysl20, affine=np.eye(4))
				imgnthree3.header.set_data_dtype(np.uint8)
				nib.save(imgnthree3,str6)

				str5=new_fold + '/Y3d_ground_truth_patient_%s.%s' % (itter,'nii.gz')
				imgnthree2=nib.Nifti1Image(Y10, affine=np.eye(4))
				imgnthree2.header.set_data_dtype(np.uint8)
				nib.save(imgnthree2,str5)
			else:
				print("no disease is detected...")
                                                   
			print("finish patient nii!!")


	def class_transform(self,X):

		X=np.where(X==1,1,X) #exclude the healthy lungs
		X=np.where(X==2,1,X)
		X=np.where(X==4,2,X)
		X=np.where(X==6,3,X)
		X=np.where(X==7,4,X)
		X=np.where(X==5,6,X)
		X=np.where(X==10,5,X)        
		X=np.where(X==8,6,X)
		X=np.where(X==9,6,X)
		X=np.where(X==11,6,X)

		return X

	def transf_jpg(self,x,uniq=1):
		new_fold=self.STORE_PATH1+'/torch_patch'
		xf=np.array(x,dtype='f')
		str2=new_fold + '/X_patch_%s.%s' % (uniq,'jpg')
		plt.imsave(str2, xf, format='jpg', cmap = "gray")
		xloadj=Image.open(str2).convert('L')
		xj=np.array(xloadj,dtype='f')
		return xj
