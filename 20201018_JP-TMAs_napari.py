# 2020-05-05 Need to threshold with Napari!
# author engje
# ipython --gui=qt
#%run 20200504_JPTMAs_napari.py
import napari
import os
import skimage
from skimage import io
import numpy as np
import copy
import pandas as pd
import tifffile

#paths
codedir = 'C:\\Users\\engje\\Documents\\Data\\2020\\20200217_JPTMAs'

regdir = f'{codedir}\\Cropped_highres'
regdir = f'{codedir}\\Cropped'

os.chdir('..\..')
from mplex_image import visualize as viz
from mplex_image import process, analyze

#load positive and intensity data
os.chdir(codedir)
#load positive and intensity data
os.chdir(codedir)

s_man = '20211207_JP-TMA1_GatedCellTypes_M1M2.csv'
df_man = pd.read_csv(f'data/{s_man}',index_col=0)
df_man['cellid'] = [item.split('_cell')[1] for item in df_man.index]
#df_pos = analyze.celltype_to_bool(df_man,'celltype')
#df_pos['slide_scene'] = [item.split('_cell')[0] for item in df_pos.index]

#df_mi = pd.read_csv('./data/20211207_JP-TMA1_LeidenClustering_neighbors15_resolution0.6_markers34_ER+.csv',index_col=0)
df_mi = pd.read_csv('./data/20211207_JP-TMA1_LeidenClustering_neighbors15_resolution0.6_markers32_ER+.csv',index_col=0)
df_pos = analyze.celltype_to_bool(df_mi,'leiden')
df_pos = df_pos.merge(df_man.loc[:,['DAPI_X','DAPI_Y']], left_index=True,right_index=True)


d_crop = {'JE-TMA-42-Scene-01':(2000,2000),
 'JE-TMA-42-Scene-02':(2500,2500),
 'JE-TMA-42-Scene-03':(2000,2000),
 'JE-TMA-42-Scene-04':(1600,2000),
 'JE-TMA-42-Scene-05':(2000,2000),
 'JE-TMA-42-Scene-06':(2000,2000),
 'JE-TMA-42-Scene-07':(2500,2000),
 'JE-TMA-42-Scene-08':(2500,2000),
 'JE-TMA-42-Scene-09':(2000,2000),
 'JE-TMA-42-Scene-10':(2000,2000),
 'JE-TMA-42-Scene-11':(2000,1000),
 'JE-TMA-42-Scene-12':(3000,3000),
 'JE-TMA-42-Scene-13':(2800,2500),
 'JE-TMA-42-Scene-14':(2200,2200),
 'JP-TMA1-1-Scene-001': (1801,873),#(3000, 3000),
 'JP-TMA1-1-Scene-002': (2000, 2000),
 'JP-TMA1-1-Scene-003': (1600, 2000),
 'JP-TMA1-1-Scene-004': (1000, 1000),
 'JP-TMA1-1-Scene-005': (1600, 2600),
 'JP-TMA1-1-Scene-006': (3000, 3000),
 'JP-TMA1-1-Scene-007': (2000, 2000),
 'JP-TMA1-1-Scene-008': (4000, 2000),
 'JP-TMA1-1-Scene-009': (3200, 1500),
 'JP-TMA1-1-Scene-011': (2000, 2000),
 'JP-TMA1-1-Scene-015': (2000, 500),
 'JP-TMA1-1-Scene-016': (1000, 1000),
 'JP-TMA1-1-Scene-017': (2200, 2200),
 'JP-TMA1-1-Scene-018': (2000, 2000),
 'JP-TMA1-1-Scene-019': (1200, 2200),
 'JP-TMA1-1-Scene-020': (2000, 2000),
 'JP-TMA1-1-Scene-021': (2000, 2000),
 'JP-TMA1-1-Scene-022': (2000, 2000),
 'JP-TMA1-1-Scene-023': (500, 3000),
 'JP-TMA1-1-Scene-024': (1500, 1000),
 'JP-TMA1-1-Scene-025': (500, 1000),
 'JP-TMA1-1-Scene-026': (500, 1000),
 'JP-TMA1-1-Scene-027': (2000, 2000),
 'JP-TMA1-1-Scene-028': (1800,800), #(1000,1000) 
 'JP-TMA1-1-Scene-029': (1000, 1000),
 'JP-TMA1-1-Scene-030': (2000, 1500),
 'JP-TMA1-1-Scene-031': (2500, 2500),
 'JP-TMA1-1-Scene-032': (2000, 2000),
 'JP-TMA1-1-Scene-033': (2000, 3000),
 'JP-TMA1-1-Scene-034': (2000, 2000),
 'JP-TMA1-1-Scene-035': (2000, 2000),
 'JP-TMA1-1-Scene-036': (2000, 2000),
 'JP-TMA1-1-Scene-037': (2000, 2500),
 'JP-TMA1-1-Scene-038': (2000, 2000),
 'JP-TMA1-1-Scene-039': (2000, 2000),
 'JP-TMA1-1-Scene-040': (500, 2500),
 'JP-TMA1-1-Scene-041': (2000, 2000),
 'JP-TMA1-1-Scene-042': (500, 2500),
 'JP-TMA1-1-Scene-043': (2000, 2000),
 'JP-TMA1-1-Scene-044': (2000, 2000),
 'JP-TMA1-1-Scene-045': (500, 1000),
 'JP-TMA1-1-Scene-046': (2000, 3500),
 'JP-TMA1-1-Scene-047': (2000, 2000),
 'JP-TMA1-1-Scene-048': (2000, 2000),
 'JP-TMA1-1-Scene-049': (2000, 2000),
 'JP-TMA1-1-Scene-050': (2500, 2000),
 'JP-TMA1-1-Scene-051': (2000, 2000),
 'JP-TMA1-1-Scene-052': (2000, 2500),
 'JP-TMA1-1-Scene-053': (2000, 2000),
 'JP-TMA1-1-Scene-054': (3000, 5000),
 'JP-TMA1-1-Scene-055': (2000, 2000),
 'JP-TMA1-1-Scene-056': (2200, 800),
 'JP-TMA1-1-Scene-057': (2000, 2000),
 'JP-TMA1-1-Scene-058': (2000, 2000),
 'JP-TMA1-1-Scene-059': (2000, 2000),
 'JP-TMA1-1-Scene-060': (2000, 2000),
 'JP-TMA1-1-Scene-061': (2000, 2000),
 'JP-TMA1-1-Scene-062': (2000, 2000),
 'JP-TMA1-1-Scene-063': (2000, 2000),
 'JP-TMA1-1-Scene-064': (3000, 100),
 'JP-TMA1-1-Scene-065': (2500, 1000),
 'JP-TMA1-1-Scene-066': (3000, 100),
 'JP-TMA1-1-Scene-067': (500, 1500),
 'JP-TMA1-1-Scene-068': (2000, 2000),
 'JP-TMA1-1-Scene-069': (2500, 1000),
 'JP-TMA1-1-Scene-070': (2000, 2000),
 'JP-TMA1-1-Scene-071': (2000, 2000),
 'JP-TMA1-1-Scene-072': (200, 2000),
 'JP-TMA1-1-Scene-073': (2000, 2000),
 'JP-TMA1-1-Scene-074': (2000, 2000),
 'JP-TMA1-1-Scene-075': (2000, 2000),
 'JP-TMA1-1-Scene-076': (2000, 2000),
 'JP-TMA1-1-Scene-077': (2000, 2000),
 'JP-TMA1-1-Scene-078': (3000, 1000),
 'JP-TMA1-1-Scene-080': (2000, 2000),
 'JP-TMA1-1-Scene-081': (2000, 3000),
 'JP-TMA1-1-Scene-082': (2000, 3000),
 'JP-TMA1-1-Scene-083': (2000, 2000),
 'JP-TMA1-1-Scene-085': (2000, 2000),
 'JP-TMA1-1-Scene-086': (2000, 2000),
 'JP-TMA1-1-Scene-087': (2000, 2000),
 'JP-TMA1-1-Scene-088': (200, 2000),
 'JP-TMA1-1-Scene-089': (2000, 500),
 'JP-TMA1-1-Scene-090': (2000, 2000),
 'JP-TMA1-1-Scene-091': (2000, 2000),
 'JP-TMA1-1-Scene-092': (2500, 2500),
 'JP-TMA1-1-Scene-093': (2000, 2000),
 'JP-TMA1-1-Scene-094': (2000, 2000),
 'JP-TMA1-1-Scene-095': (2500, 2500),
 'JP-TMA1-1-Scene-097': (1000, 2500),
 'JP-TMA1-1-Scene-098': (3000, 3000),
 'JP-TMA1-1-Scene-099': (2000, 2000),
 'JP-TMA1-1-Scene-100': (2000, 2000),
 'JP-TMA1-1-Scene-103': (1000, 2000),
 'JP-TMA1-1-Scene-104': (1000, 1000),
 'JP-TMA1-1-Scene-105': (2000, 2000),
 'JP-TMA1-1-Scene-106': (2000, 2000),
 'JP-TMA1-1-Scene-107': (1000, 1000),
 'JP-TMA1-1-Scene-108': (500, 3200),
 'JP-TMA1-1-Scene-109': (2000, 2000),
 'JP-TMA1-1-Scene-110': (2000, 2000),
 'JP-TMA1-1-Scene-112': (2000, 2000),
 'JP-TMA1-1-Scene-113': (2000, 2000),
 'JP-TMA1-1-Scene-115': (2000, 2000),
 'JP-TMA1-1-Scene-116': (500, 2000),
 'JP-TMA1-1-Scene-117': (2000, 2000),
 'JP-TMA1-1-Scene-118': (2000, 2000),
 'JP-TMA1-1-Scene-119': (2000, 2000),
 'JP-TMA1-1-Scene-120': (2000, 2000),
 'JP-TMA1-1-Scene-121': (1000, 1600),
 'JP-TMA1-1-Scene-122': (2000, 1000),
 'JP-TMA1-1-Scene-123': (2000, 2000),
 'JP-TMA1-1-Scene-124': (1000, 1000),
 'JP-TMA1-1-Scene-125': (2000, 2000),
 'JP-TMA1-1-Scene-126': (2000, 1000),
 'JP-TMA1-1-Scene-127': (3000, 3000),
 'JP-TMA1-1-Scene-128': (2000, 2000),
 'JP-TMA1-1-Scene-130': (2000, 2000),
 'JP-TMA1-1-Scene-131': (1000, 3500),
 'JP-TMA2-1-Scene-01': (2000, 2000),
 'JP-TMA2-1-Scene-02': (1000, 2000),
 'JP-TMA2-1-Scene-03': (1000, 2000),
 'JP-TMA2-1-Scene-04': (1500, 1500),
 'JP-TMA2-1-Scene-05': (1000, 2000),
 'JP-TMA2-1-Scene-06': (1300, 2300),
 'JP-TMA2-1-Scene-07': (2600, 2600),
 'JP-TMA2-1-Scene-08': (1000, 1000),
 'JP-TMA2-1-Scene-09': (1000, 2000),
 'JP-TMA2-1-Scene-10': (2000, 2000),
 'JP-TMA2-1-Scene-11': (1000, 2000),
 'JP-TMA2-1-Scene-12': (800, 2300),
 'JP-TMA2-1-Scene-13': (1000, 2000),
 'JP-TMA2-1-Scene-14': (1000, 2000),
 'JP-TMA2-1-Scene-15': (1000, 2000),
 'JP-TMA2-1-Scene-16': (200, 2000),
 'JP-TMA2-1-Scene-17': (2000, 2000),
 'JP-TMA2-1-Scene-18': (1000, 2000),
 'JP-TMA2-1-Scene-19': (2000, 2000),
 'JP-TMA2-1-Scene-20': (2000, 2000),
 'JP-TMA2-1-Scene-21': (2000, 2000),
 'JP-TMA2-1-Scene-22': (1000, 2000),
 'JP-TMA2-1-Scene-23': (2000, 2000),
 'JP-TMA2-1-Scene-24': (2000, 2000),
 'JP-TMA2-1-Scene-25': (2000, 1000),
 'JP-TMA2-1-Scene-26': (2400, 1000),
 'JP-TMA2-1-Scene-27': (2000, 1000),
 'JP-TMA2-1-Scene-28': (2000, 1000),
 'JP-TMA2-1-Scene-29': (2000, 1000),
 'JP-TMA2-1-Scene-30': (2000, 2000),
 'JP-TMA2-1-Scene-31': (2000, 2000),
 'JP-TMA2-1-Scene-32': (2000, 2000),
 'JP-TMA2-1-Scene-33': (2000, 2000),
 'JP-TMA2-1-Scene-34': (1500, 2000),
 'JP-TMA2-1-Scene-35': (1500, 2000),
 'JP-TMA2-1-Scene-36': (1500, 2000),
 'JP-TMA2-1-Scene-37': (1500, 2000),
 'JP-TMA2-1-Scene-38': (2000, 2000),
 'JP-TMA2-1-Scene-39': (1500, 2000),
 'JP-TMA2-1-Scene-40': (1500, 1500),
 'JP-TMA2-1-Scene-41': (1500, 1500),
 'JP-TMA2-1-Scene-42': (2000, 2000)
 
 }

#sample
s_slide = 'JE-TMA-42-Scene-13'
#s_slide = 'JP-TMA1-1-Scene-105'
#s_slide = 'JP-TMA2-1-Scene-38' #nice example
s_slide ='JP-TMA2-1-Scene-07'
s_slide ='JP-TMA1-1-Scene-041'
s_slide ='JP-TMA1-1-Scene-053'
s_slide ='JP-TMA1-1-Scene-007'

# survival samples
'''
#diff state
s_slide = 'JP-TMA1-1-Scene-028' #not a bad core: has necrotic region but that doesn't stain for tumor
# just has 1 little normal duct 
s_slide = 'JP-TMA1-1-Scene-030' #true lum Ecad-, cells are not adhering to each other
s_slide = 'JP-TMA1-1-Scene-097' #true lum Ecad-, cells are not adhering
s_slide = 'JP-TMA1-1-Scene-055' #Lum, Ecad+: its out of focus
s_slide = 'JP-TMA1-1-Scene-120' #Lum, Ecad+: its out of focus
s_slide = 'JP-TMA1-1-Scene-039' #Lum, Ecad+: its out of focus
s_slide = 'JP-TMA1-1-Scene-017' #EGFR atrifact
s_slide = 'JP-TMA1-1-Scene-001' #EGFR atrifact
s_slide = 'JP-TMA1-1-Scene-067' #EGFR atrifact
s_slide = 'JP-TMA1-1-Scene-006' #bad core
s_slide = 'JP-TMA1-1-Scene-072' #EGFR atrifact
#Ecad

#non tumor
#s_slide = 'JP-TMA1-1-Scene-034' #PDPN real
#s_slide = 'JP-TMA1-1-Scene-051' #PDPN real
#s_slide = 'JP-TMA1-1-Scene-116' #PDPN real
#s_slide = 'JP-TMA1-1-Scene-028' #bad core
#non tumor minus 28 TNBC
#s_slide = 'JP-TMA1-1-Scene-001' #asma/PDPN artifact
s_slide = 'JP-TMA1-1-Scene-017' #true Ecad+, CK7/CK19- tumor asma/PDPN artifact
#celltype
#s_slide = 'JP-TMA1-1-Scene-004' #true Ecad+, CK7/CK19- tumor,CD31 real
#tumor
#s_slide = 'JP-TMA1-1-Scene-002' #Ecad+, luminal EGFR+, tumor nests, short survival
#s_slide = 'JP-TMA1-1-Scene-076' #Ecad+, luminal tumor nests
#s_slide = 'JP-TMA1-1-Scene-001' #Ecad+ luminal
#s_slide = 'JP-TMA1-1-Scene-040' #long survivor, Ecad+, EGFR+, some few CK5, tumor is not CK7/CK19 negative
#s_slide = 'JP-TMA1-1-Scene-104' #same pt as 40, lots of immune; CD8+ (rethresh ck19/7)

#s_slide = 'JP-TMA1-1-Scene-021' #short, tumor is CK low, lots of immune
#s_slide = 'JP-TMA1-1-Scene-029' #short, luminal lots of CD8/CD4/CD45; CD31 real
#s_slide = 'JP-TMA1-1-Scene-095'  #short, luminal, heterogenous CK17, lots of CD8/CD4/CD45; CD31 real
#s_slide = 'JP-TMA1-1-Scene-033' #CD45 and CD3 out of focus
#s_slide = 'JP-TMA1-1-Scene-100' #CD45 and CD3 out of focus, maybe underest immune? some norm duct
#non tumor (ER+)
#s_slide = 'JP-TMA1-1-Scene-046' #PDPN tumor
#s_slide = 'JP-TMA1-1-Scene-110' #real+ artefact
#s_slide = 'JP-TMA1-1-Scene-053' #real+ artefact
#s_slide = 'JP-TMA1-1-Scene-118' #pdpn artefact
#s_slide = 'JP-TMA1-1-Scene-024' #real+ artefact/ ER+ real
#s_slide = 'JP-TMA1-1-Scene-023' #ER+ real
#s_slide = 'JP-TMA1-1-Scene-032' #long surv, ER looks weak/funny, #PDPN real, only 8%, probaly lyphatics
#s_slide = 'JP-TMA1-1-Scene-099' #long surv: more than 0% ER+ #PDPN weak, everywhere
#s_slide = 'JP-TMA1-1-Scene-085' #short, ER low
#s_slide = 'JP-TMA1-1-Scene-027' #short, ER+ but regions of ER-
#s_slide = 'JP-TMA1-1-Scene-081' #short, looks mostly ER+
s_slide = 'JP-TMA1-1-Scene-001'
'''

#has artifact
'''
s_slide ='JP-TMA1-1-Scene-035'
s_slide ='JP-TMA1-1-Scene-108'

s_slide ='JP-TMA1-1-Scene-092'
s_slide ='JP-TMA1-1-Scene-071'
s_slide ='JP-TMA1-1-Scene-081'
s_slide ='JP-TMA1-1-Scene-046'
s_slide ='JP-TMA1-1-Scene-023' #?
s_slide ='JP-TMA1-1-Scene-091'
s_slide ='JP-TMA1-1-Scene-081'
'''
s_slide ='JP-TMA1-1-Scene-032'
s_cell = 28
#load images
os.chdir(regdir)

for s_file in os.listdir():
    if s_file.find(f'{s_slide}_x') > -1:
        s_crop = s_file.split('_')[1]
#s_crop = f'x{d_crop[s_slide][0]}y{d_crop[s_slide][1]}'
viewer = napari.Viewer()
label_image = viz.load_crops(viewer,s_crop,s_slide)

#certain centroids

ls_index = ['JP-TMA1-1_scene080_cell1795']
points,x_arr = viz.add_points(s_crop,df_man.loc[ls_index],s_slide.replace('-Scene-','_scene'))
viewer.add_points(points)

#all centroids
points,x_arr = viz.add_points(s_crop,df_man,s_slide.replace('-Scene-','_scene'))
viewer.add_points(points)

#celltype centroids

cell_points, __ = viz.add_celltype_points(s_crop,df_pos,s_slide.replace('-Scene-','_scene'),s_cell,i_scale = 2)
viewer.add_points(cell_points,face_color='r')

#select points by drawing shape
'''
point_properties, b_poly = viz.points_in_shape(viewer,points,df_man,s_slide.replace('-Scene-','_scene'))
points_layer = viewer.add_points(points, properties=point_properties, face_color='in_poly',size=10)

'''

#select mask
'''
#inpoly
df_exclude = x_arr.loc[b_poly]

#out of poly
df_exclude = x_arr.loc[~b_poly]

#add inpoly
df_exclude = df_exclude.append(x_arr.loc[b_poly])

#add out of poly
df_exclude = df_exclude.append(x_arr.loc[~b_poly])

df_exclude.to_csv(f"exclude_{s_slide.replace('-Scene-','_scene')}.csv")
'''

#show positive results
'''
ls_cell = ls_cell = ['CK7_Ring','HER2_Ring', 'CD44_Ring','Vim_Ring']
df_scene = df_pos[df_pos.slide_scene==f"{s_slide.split('-Scene-')[0]}_scene{s_slide.split('-Scene-')[1]}"]

for s_cell in ls_cell: 
    label_image_cell = viz.pos_label(viewer,df_scene,label_image,s_cell)
'''
os.chdir(codedir)


