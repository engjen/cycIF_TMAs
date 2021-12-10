# image processing for with mlpex_image
# date: 2020-08-18
# author: engje
# language Python 3.8
# license: GPL>=v3

#libraries

import os
import sys
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import re
from skimage import io, measure, segmentation, morphology
import scipy
import math

# cd /home/groups/graylab_share/Chin_Lab/ChinData/Cyclic_Workflow/cmIF_2019-12-02_JP-TMA1

####  Paths  ####

codedir = os.getcwd()
rootdir = f'{codedir}'
czidir = rootdir.replace('_Workflow','_Images')

#automatically generated
tiffdir = f'{rootdir}/RawImages'
qcdir = f'{rootdir}/NewQC'
regdir = f'{rootdir}/RegisteredImages'
subdir = f'{rootdir}/SubtractedRegisteredImages'
segdir = f'{rootdir}/Segmentation'
cropdir = f'{rootdir}/Cropped'


# Start Preprocessing
#os.chdir('/home/groups/graylab_share/OMERO.rdsStore/engje/Data/')
from mplex_image import preprocess, mpimage, cmif
preprocess.cmif_mkdir([tiffdir,qcdir,regdir,segdir,subdir,cropdir])

os.chdir(codedir)

ls_sample = ['JP-TMA1-1',
 #'JP-TMA2-1',
 #'JE-TMA-42',
 ]

#### 3 QC raw images ####
'''
preprocess.cmif_mkdir([f'{qcdir}/RawImages'])

ls_scene = [ '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057',
 '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074',
 '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091',
 '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108',
 '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125',
 '126', '127', '128', '129', '130', '131']
ls_scene = [ '013']
ls_scene = ['129', '130', '131']
ls_scene = [ '128']
ls_scene = [ '096']
for s_sample in ['JP-TMA1-1']:     #'JP-TMA1-1','HER2B-K176', 'JP-TMA2-1', 'JE-TMA-42'
    os.chdir(f'{tiffdir}/{s_sample}')
    #sort and count images
    df_img = mpimage.parse_org(s_end = "ORG.tif",type='raw')
    #cmif.count_images(df_img[df_img.slide==s_sample])
    #investigate tissues
    cmif.visualize_raw_images(df_img[(df_img.slide==s_sample) & (df_img.scene.isin(ls_scene))],qcdir,color='c1')
'''

#### 5 Check Registration Visualization #### 
'''
for s_sample in ls_sample:
    cmif.visualize_reg_images(f'{regdir}',qcdir,color='c1',s_sample=s_sample)
'''
#### 6 Create AF Subtracted Images #### 
'''
#parameters
#d_channel = {'c2':'R8Qc2','c3':'R8Qc3','c4':'R8Qc4','c5':'R8Qc5'}
#d_early={'c2':'R0c2','c3':'R0c3','c4':'R0c4','c5':'R0c5'}
d_channel = {'c2':'R5Qc2','c3':'R5Qc3','c4':'R5Qc4','c5':'R5Qc5'}
d_early = {}

for s_sample in ls_sample:
    preprocess.cmif_mkdir([f'{subdir}/{s_sample}'])
    os.chdir(f'{regdir}/{s_sample}')
    for s_file in os.listdir():
        print(s_file)
        if s_file.find(s_sample) > -1:
            os.chdir(s_file)
            df_img = mpimage.parse_org()
            ls_exclude = sorted(set(df_img[df_img.color=='c5'].marker)) + ['DAPI'] + [item for key, item in d_channel.items()] + [item for key, item in d_early.items()]
            #subtract
            #df_markers = cmif.autofluorescence_subtract(s_sample,df_img,f'{codedir}/data/PipelineExample',d_channel,ls_exclude,subdir=f'{subdir}/{s_sample}') #
            cmif.autofluorescence_subtract(s_sample,df_img,codedir,d_channel,ls_exclude,f'{subdir}/{s_sample}',d_early)
            os.chdir('..')
'''
#generate channel/marker metadata csv
#cmif.metadata_table(regdir,segdir)

#### 7 Cellpose segmentation ####
'''
from mplex_image import segment
import time

nuc_diam = 30 #nuclei 30 looks good; flow threshold 0
cell_diam = 30 # cell 30, flow thresh 0.6

s_seg_markers = "['CK7']"  # out of focus Ecad, CK7 good, flow 0.6 looks good. but missing lots of cells at flow = 0.4 and 0.2, mistakes at flow 0.
s_type ='cell' #'nuclei'#

print(f'Predicting {s_type}')
for s_sample in ls_sample:
    preprocess.cmif_mkdir([f'{segdir}/{s_sample}Cellpose_Segmentation'])
    os.chdir(f'{regdir}/{s_sample}')
    for s_file in os.listdir():
        if s_file.find(s_sample) > -1:
            os.chdir(f'{regdir}/{s_sample}/{s_file}')
            print(f'Processing {s_file}')
            df_img = segment.parse_org()
            for s_scene in sorted(set(df_img.scene)):
                s_slide_scene= f'{s_sample}-Scene-{s_scene}'
                s_find = df_img[(df_img.rounds=='R1') & (df_img.color=='c1') & (df_img.scene==s_scene)].index[0]
                segment.cellpose_segment_job(s_file,s_slide_scene,
                 s_find,f'{segdir}/{s_sample}Cellpose_Segmentation',
                 f'{regdir}/{s_sample}/{s_slide_scene}',nuc_diam,cell_diam,
                 s_type,s_seg_markers,s_match='match')#,s_job='gpu' ,s_match= 'seg' or 'match' 
                os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
                os.system(f'sbatch cellpose_{s_type}_{s_slide_scene}.sh')
                time.sleep(5)
                print('Next')
'''
#### 7 Cellpose segmentation ####
'''
from mplex_image import segment
nuc_diam = 30
cell_diam = 30 

s_seg_markers = "['CK7']"
s_type = 'nuclei'# 'cell' #''both' #'
if s_type == 'nuclei':
    s_match='seg'
else:
    s_match='both'

print(f'Predicting {s_type}')
for s_sample in ls_sample:
    segment.segment_spawner(s_sample,segdir,f'{regdir}/{s_sample}',nuc_diam,cell_diam,s_type,s_seg_markers,s_job='short',s_match=s_match)


# check seg done
for s_sample in ls_sample:
    df = pd.read_csv(f'{segdir}/features_{s_sample}_FilteredMeanIntensity_DAPI12_DAPI2.csv',index_col=0)
    es_scene =  set([item.replace('_scene','-Scene-') for item in df.slide_scene.unique()]) 
    os.chdir(f'{regdir}/{s_sample}')
    ls_scene = os.listdir() 
    os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
    print('\n nuc')
    for s_scene in ls_scene:
        if not os.path.exists(f'{s_scene} nuclei30 - Nuclei Segmentation Basins.tif'):
            print(f'x sbatch cellpose_nuclei_{s_scene}.sh')
        elif len(set([s_scene]).intersection(es_scene))==0:
            print(f'sbatch cellpose_nuclei_{s_scene}.sh')
    print('\n cell')
    for s_scene in ls_scene:
        if not os.path.exists(f'{s_scene}_CK7 cell30 - Cell Segmentation Basins.tif'):
            print(f'x sbatch cellpose_cell_{s_scene}.sh')
        elif len(set([s_scene]).intersection(es_scene))==0:
            print(f'sbatch cellpose_cell_{s_scene}.sh')
'''
#### 8 Extract Cellpose Features ####
'''
from mplex_image import features

nuc_diam = 30
cell_diam = 30 
ls_seg_markers = ['CK7']

for s_sample in ls_sample: 
    df_sample, df_thresh = features.extract_cellpose_features(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam,b_big=True)
    df_sample.to_csv(f'{segdir}/features_{s_sample}_MeanIntensity_Centroid_Shape.csv')
    df_thresh.to_csv(f'{segdir}/thresh_{s_sample}_ThresholdLi.csv')
'''
#8.1 Top 25% pixels
'''
from mplex_image import features

nuc_diam = 30
cell_diam = 30 
ls_seg_markers = ['CK7']
ls_membrane = ['HER2','EGFR','AR','ER','Ecad']

for s_sample in ls_sample: 
    df_sample = features.extract_bright_features(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam,ls_membrane)
    df_sample.to_csv(f'{segdir}/features_{s_sample}_BrightMeanIntensity.csv')
'''
### filter cellpose features 12/6/21 #######
'''
from mplex_image import process, features

nuc_diam = 30
cell_diam = 30 
ls_seg_markers = ['CK7']
s_thresh='CK7'
ls_membrane = ['HER2','EGFR','Ecad']
ls_marker_cyto = ['CK14','CK5','CK17','CK19','CK7','CK8','Ecad','HER2']
ls_custom = ['ER_nuclei25','AR_nuclei25','EGFR_cellmem25','HER2_cellmem25','Ecad_cellmem25','CD44_nucadj2','Vim_nucadj2']
ls_filter = ['DAPI12_nuclei','DAPI2_nuclei']
ls_shrunk = ['CD44_nucadj2','Vim_nucadj2']
man_thresh = 400

for s_sample in ls_sample: 
    # long
    os.chdir(segdir)
    df_img_all = process.load_li([s_sample],s_thresh, man_thresh)
    df_mi_full = process.load_cellpose_df([s_sample], segdir)
    df_xy = process.filter_cellpose_xy(df_mi_full)
    df_mi_full, i_max = process.drop_last_rounds(df_img_all,ls_filter,df_mi_full)
    df_mi_filled = process.fill_cellpose_nas(df_mi_full,ls_marker_cyto,s_thresh=s_thresh,man_thresh=man_thresh)
    
    df_mi_filled = process.shrink_seg_regions(df_mi_filled,s_thresh,ls_celline=[],ls_shrunk=ls_shrunk)
    df_mi_mem_fill = process.fill_bright_nas(ls_membrane,s_sample,s_thresh,df_mi_filled,segdir)
    df_mi,es_standard = process.filter_loc_cellpose(df_mi_mem_fill, ls_marker_cyto, ls_custom,filter_na=False)
    #096 Her2 problem
    df_mi.loc[df_mi.slide_scene=='JP-TMA1-1_scene096','HER2_cytoplasm'] = df_mi.loc[df_mi.slide_scene=='JP-TMA1-1_scene096','Her2_perinuc5']
    df_mi.loc[df_mi.slide_scene=='JP-TMA1-1_scene096','HER2_cellmem25']= df_mi.loc[df_mi.slide_scene=='JP-TMA1-1_scene096','Her2_perinuc5'] #not ideal but only 161 cells
    df_mi = df_mi.drop(['Her2_nuclei','Her2_perinuc5'],axis=1)
    df_mi = df_mi.dropna()
    df_pos_auto,d_thresh_record = process.auto_threshold(df_mi,df_img_all)
    #ls_color = process.plot_thresh_results(df_img_all,df_pos_auto,d_thresh_record,df_xy,i_max,s_thresh,qcdir)
    ls_color = df_pos_auto.columns[df_pos_auto.columns.str.contains('DAPI')]
    #df_mi_filter = process.filter_dapi_cellpose(df_pos_auto,ls_color,df_mi,ls_filter,qcdir)
    #df_mi_filter.to_csv(f'{segdir}/features_{s_sample}_FilteredMeanIntensity_{"_".join([item.split("_")[0] for item in ls_filter])}.csv')
    df_out = df_mi.merge(df_pos_auto.loc[:,ls_color],left_index=True,right_index=True,suffixes=('','pos'))
    df_out.to_csv(f'{segdir}/features_{s_sample}_FilteredMeanIntensity.csv')
    df_xy.loc[df_mi.index].to_csv(f'{segdir}/features_{s_sample}_CentroidXY.csv')
    
    #Expand nuclei without matching cell labels for cells touching analysis
    #just need to run fill_cellpose_nas and use df_mi_filled
    labels,combine,dd_result = features.combine_labels(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam, df_mi_filled,s_thresh)
    #process.marker_table(df_img_all,qcdir)
'''

#### 9 Filter cellpose features #### 
'''
def check_scenes(df_mi_full):
    ls_scene = []
    for s_scene in sorted(set(df_mi_full.slide_scene)):
        df_scene = df_mi_full[df_mi_full.slide_scene == s_scene]
        if len(df_scene.dropna()) == 0:
            print(s_scene)
            ls_scene.append(s_scene)
    return(ls_scene)

from mplex_image import process, features
#parameters
nuc_diam = 30
cell_diam = 30 
ls_seg_markers = ['CK7']
s_thresh='CK7'
ls_membrane = ['HER2','EGFR','Ecad']
ls_marker_cyto = ['CK14','CK5','CK17','CK19','CK7','CK8','Ecad','HER2']
ls_custom = ['ER_nuclei25','AR_nuclei25','EGFR_cellmem25','HER2_cellmem25','Ecad_cellmem25','CD44_nucadj2','Vim_nucadj2']
ls_filter = ['DAPI12_nuclei','DAPI2_nuclei']
ls_shrunk = ['CD44_nucadj2','Vim_nucadj2']
man_thresh = 400
'''
#filtering normal
'''
for s_sample in ls_sample: 
    os.chdir(segdir)
    #replace nas, select segmentation region and filter cells negative for dapi
    df_mi_full,df_img_all = process.filter_cellpose_df(s_sample,segdir,qcdir,s_thresh,ls_membrane,ls_marker_cyto,
     ls_custom,ls_filter,ls_shrunk,man_thresh)
    #Expand nuclei without matching cell labels for cells touching analysis
    #se_neg = df_mi_full[df_mi_full.slide == s_sample].loc[:,f'{s_thresh}_negative']
    #se_neg = df_mi_full[df_mi_full.slide.str.contains(s_sample)].loc[:,f'{s_thresh}_negative']
    #labels,combine,dd_result = features.combine_labels(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam, se_neg)
    #process.marker_table(df_img_all,qcdir)
'''

#check bad
'''
for s_sample in ls_sample: 
    df_result = features.check_combined(segdir,s_sample,cell_diam,ls_seg_markers)
    df_result.to_csv(f'{segdir}/features_{s_sample}_BadMatchCells{cell_diam}.csv')
'''

#filtering JP-TMA1
'''
for s_sample in ls_sample: 
    os.chdir(segdir)
    df_img_all = process.load_li([s_sample])
    df_mi_full = process.load_cellpose_df([s_sample], segdir)
    ls_scene = check_scenes(df_mi_full)
    df_mi_full = df_mi_full[~df_mi_full.slide_scene.isin(ls_scene)]
    df_xy = process.filter_cellpose_xy(df_mi_full)
    d_scene = {'a':sorted(set(df_mi_full.slide_scene))[:60],'b':sorted(set(df_mi_full.slide_scene))[60:]}
    for key, scenes in d_scene.items():
        df_half = df_mi_full[df_mi_full.slide_scene.isin(scenes)]
        df_half.to_csv(f'features_{s_sample}{key}_MeanIntensity_Centroid_Shape.csv')
'''

#filtering JP-TMA1 cont'd
'''
key = 'a'
for s_sample in ls_sample: 
    os.chdir(segdir)
    df_img_all = process.load_li([s_sample])
    s_sample_name = f'{s_sample}{key}'
    df_mi_full = process.load_cellpose_df([s_sample_name], segdir)
    df_xy = process.filter_cellpose_xy(df_mi_full)
    df_mi_full.slide = s_sample
    df_mi_full.slide_scene = df_mi_full.slide + '_' + df_mi_full.scene
    #manually override too low Ecad thresh
    df_img_all.loc[df_img_all[(df_img_all.marker==s_thresh) & (df_img_all.threshold_li < man_thresh)].index, 'threshold_li'] = man_thresh
    df_mi_filled = process.fill_cellpose_nas(df_img_all,df_mi_full,ls_marker_cyto,s_thresh=s_thresh,ls_celline=[],
      ls_shrunk = ls_shrunk,qcdir=qcdir)
    if len(ls_membrane) > 0:
        print(f'Loading features_{s_sample}_BrightMeanIntensity.csv')
        df_mi_mem = pd.read_csv(f'{segdir}/features_{s_sample}_BrightMeanIntensity.csv',index_col=0)
        df_mi_mem_fill = process.fill_membrane_nas(df_mi_filled, df_mi_mem,s_thresh=s_thresh,ls_membrane=ls_membrane)
    else:
        df_mi_mem_fill = df_mi_filled
    df_mi = process.filter_loc_cellpose(df_mi_mem_fill, ls_marker_cyto, ls_custom)
    df_pos_auto,d_thresh_record = process.auto_threshold(df_mi,df_img_all)
    ls_color = [item + '_nuclei' for item in df_img_all[(df_img_all.slide_scene==df_img_all.slide_scene.unique()[0]) & (df_img_all.marker.str.contains('DAPI'))].marker.tolist()]
    process.positive_scatterplots(df_pos_auto,d_thresh_record,df_xy,ls_color + [f'{s_thresh}_cytoplasm'],qcdir)
    df_mi_filter = process.filter_dapi_cellpose(df_pos_auto,ls_color,df_mi,ls_filter,df_img_all,qcdir)
    df_mi_filter.to_csv(f'{segdir}/features_{s_sample_name}_FilteredMeanIntensity_{"_".join([item.split("_")[0] for item in ls_filter])}.csv')
    df_xy.to_csv(f'{segdir}/features_{s_sample_name}_CentroidXY.csv')
    se_neg = df_mi_full[df_mi_full.slide.str.contains(s_sample)].loc[:,f'{s_thresh}_negative']
    labels,combine,dd_result = features.combine_labels(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam, se_neg)
'''

'''
os.chdir(segdir)
df_both = pd.DataFrame()
s_sample = 'JP-TMA1-1'
for key in ['a','b']: ##
    df = pd.read_csv(f'{segdir}/features_{s_sample}{key}_FilteredMeanIntensity_{"_".join([item.split("_")[0] for item in ls_filter])}_good.csv',index_col=0)
    df_both = df_both.append(df)
df_both['slide_scene'] = [item.split('_cell')[0] for item in df_both.index]
df_both['cell'] = [int(item.split('_cell')[1]) for item in df_both.index]
df_both.sort_values(['slide_scene','cell'],inplace=True)
df_both.drop(['cell'],axis=1)
df_both.to_csv(f'{segdir}/features_{s_sample}_FilteredMeanIntensity_{"_".join([item.split("_")[0] for item in ls_filter])}_good_both.csv')
'''
#filtering with bad 

def replace_bad(df_bad, ls_marker_cyto,df_mi_full):
    '''
    replace bad cytoplasms with good perinuc5
    '''
    print('For cells that had cytoplasm from multiple cells')
    for s_marker in ls_marker_cyto:
        print(f'Replace  {s_marker}_cytoplasm bad')
        df_mi_full.loc[df_mi_full.index.isin(df_bad.index),f'{s_marker}_cytoplasm'] = df_mi_full.loc[df_mi_full.index.isin(df_bad.index),f'{s_marker}_perinuc5'] 
        print(f'with {s_marker}_perinuc5')
    return(df_mi_full)

'''
key = 'b'
for s_sample in ls_sample: 
    os.chdir(segdir)
    df_img_all = process.load_li([s_sample])
    s_sample_name = f'{s_sample}{key}' #f'{s_sample}' #
    df_mi_full = process.load_cellpose_df([s_sample_name], segdir)
    df_xy = process.filter_cellpose_xy(df_mi_full)
    df_bad = pd.read_csv(f'features_{s_sample_name}_BadMatchCells30.csv',index_col=0)
    df_mi_full.slide = s_sample
    df_mi_full.slide_scene = df_mi_full.slide + '_' + df_mi_full.scene
    df_img_all.loc[df_img_all[(df_img_all.marker==s_thresh) & (df_img_all.threshold_li < man_thresh)].index, 'threshold_li'] = man_thresh
    df_mi_filled = process.fill_cellpose_nas(df_img_all,df_mi_full,ls_marker_cyto,s_thresh=s_thresh,ls_celline=[],
      ls_shrunk = ls_shrunk,qcdir=qcdir)
    df_good = replace_bad(df_bad,ls_marker_cyto,df_mi_filled)
    if len(ls_membrane) > 0:
        print(f'Loading features_{s_sample}_BrightMeanIntensity.csv')
        df_mi_mem = pd.read_csv(f'{segdir}/features_{s_sample}_BrightMeanIntensity.csv',index_col=0)
        df_mi_mem_fill = process.fill_membrane_nas(df_good, df_mi_mem,s_thresh=s_thresh,ls_membrane=ls_membrane)
    else:
        df_mi_mem_fill = df_good
    df_mi = process.filter_loc_cellpose(df_mi_mem_fill, ls_marker_cyto, ls_custom)
    df_pos_auto,d_thresh_record = process.auto_threshold(df_mi,df_img_all)
    ls_color = [item + '_nuclei' for item in df_img_all[(df_img_all.slide_scene==df_img_all.slide_scene.unique()[0]) & (df_img_all.marker.str.contains('DAPI'))].marker.tolist()]
    process.positive_scatterplots(df_pos_auto,d_thresh_record,df_xy,ls_color + [f'{s_thresh}_cytoplasm'],qcdir)
    df_mi_filter = process.filter_dapi_cellpose(df_pos_auto,ls_color,df_mi,ls_filter,df_img_all,qcdir)
    df_mi_filter.to_csv(f'{segdir}/features_{s_sample_name}_FilteredMeanIntensity_{"_".join([item.split("_")[0] for item in ls_filter])}_good.csv')
'''
#### 10 generate multicolor pngs and ome-tiff overlays (cropped) #### 

#crop coordinates  x, y upper corner
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
 }

d_mod = {'JP-TMA1-1-Scene-001':(3000,3000), #edited 8/21/21
 'JP-TMA1-1-Scene-003':(1600,2000),
 'JP-TMA1-1-Scene-004':(1000,1000),
 'JP-TMA1-1-Scene-005':(1600,2600),
 'JP-TMA1-1-Scene-006':(3000,3000),
 'JP-TMA1-1-Scene-008':(4000,2000),
 'JP-TMA1-1-Scene-009':(3200,1500),
 'JP-TMA1-1-Scene-015':(2000,500),
 'JP-TMA1-1-Scene-016':(1000,1000),
 'JP-TMA1-1-Scene-017':(2200,2200),
 'JP-TMA1-1-Scene-019':(1200,2200),
 'JP-TMA1-1-Scene-023':(500,2700),
 'JP-TMA1-1-Scene-024':(1500,1000),
 'JP-TMA1-1-Scene-025':(500,1000),
 'JP-TMA1-1-Scene-026':(500,1000),
 'JP-TMA1-1-Scene-028':(1800,3200),
 'JP-TMA1-1-Scene-029':(1000,1000),
 'JP-TMA1-1-Scene-030':(2000,1500),
 'JP-TMA1-1-Scene-031':(2500,2500),
 'JP-TMA1-1-Scene-033':(2000,3000),
 'JP-TMA1-1-Scene-037':(2000,2500),
 'JP-TMA1-1-Scene-039':(2000,2400),
 'JP-TMA1-1-Scene-040':(500,2500),
 'JP-TMA1-1-Scene-042':(500,2500), 
 'JP-TMA1-1-Scene-045':(500,1000),
 'JP-TMA1-1-Scene-046':(3000,3000),
 'JP-TMA1-1-Scene-046':(2000,4000),
 'JP-TMA1-1-Scene-050':(2500,2000),
 'JP-TMA1-1-Scene-052':(2000,2500),
 'JP-TMA1-1-Scene-054':(3000,5000),
 'JP-TMA1-1-Scene-056':(2200,800),
 'JP-TMA1-1-Scene-064':(3000,100),
 'JP-TMA1-1-Scene-065':(2500,1000),
 'JP-TMA1-1-Scene-066':(3000,100),
 'JP-TMA1-1-Scene-067':(500,1500),
 'JP-TMA1-1-Scene-069':(2500,1000),
 'JP-TMA1-1-Scene-072':(200,2000),
 'JP-TMA1-1-Scene-077':(1600,2400),
 'JP-TMA1-1-Scene-078':(3000,1000),
 'JP-TMA1-1-Scene-081':(2000,3000),
 'JP-TMA1-1-Scene-082':(2000,3000),
 'JP-TMA1-1-Scene-088':(200,2000),
 'JP-TMA1-1-Scene-089':(2000,500),
 'JP-TMA1-1-Scene-092':(2500,2500),
 'JP-TMA1-1-Scene-095':(2200,2500),
 'JP-TMA1-1-Scene-097':(1000,2500),
 'JP-TMA1-1-Scene-098':(2000,3000),
 'JP-TMA1-1-Scene-098':(3000,3000),
 'JP-TMA1-1-Scene-103':(1000,2000),
 'JP-TMA1-1-Scene-104':(1000,1000),
 'JP-TMA1-1-Scene-107':(1000,1000),
 'JP-TMA1-1-Scene-108':(500,3200),
 'JP-TMA1-1-Scene-110':(1500,2000),
 'JP-TMA1-1-Scene-116':(1600,2200),
 'JP-TMA1-1-Scene-121':(1000,1600),
 'JP-TMA1-1-Scene-122':(2000,1000),
 'JP-TMA1-1-Scene-124':(1000,1000),
 'JP-TMA1-1-Scene-126':(2000,1000),
 'JP-TMA1-1-Scene-127':(3000,3000),
 'JP-TMA1-1-Scene-131':(1000,3500),
 }

d_mod2 = {
  'JP-TMA2-1-Scene-02':(1000,2000),
  'JP-TMA2-1-Scene-03':(1000,2000),
  'JP-TMA2-1-Scene-04':(1500,1500),
  'JP-TMA2-1-Scene-05':(1000,2000),
  'JP-TMA2-1-Scene-06':(1300,2300),
  'JP-TMA2-1-Scene-07':(3000,2600),#  'JP-TMA2-1-Scene-07':(2600,2600),
  'JP-TMA2-1-Scene-08':(1000,1000),
  'JP-TMA2-1-Scene-09':(1000,2000),
  'JP-TMA2-1-Scene-11':(1000,2000),
  'JP-TMA2-1-Scene-12':(800,2300),
  'JP-TMA2-1-Scene-13':(1000,2000),
  'JP-TMA2-1-Scene-14':(1000,2000),
  'JP-TMA2-1-Scene-15':(1000,2000),
  'JP-TMA2-1-Scene-16':(200,2000),
  'JP-TMA2-1-Scene-18':(1000,2000),
  'JP-TMA2-1-Scene-22':(1000,2000),
  'JP-TMA2-1-Scene-25':(2000,1000),
  'JP-TMA2-1-Scene-26':(2400,1000),
  'JP-TMA2-1-Scene-27':(2000,1000),
  'JP-TMA2-1-Scene-28':(2000,1000),
  'JP-TMA2-1-Scene-29':(2000,600),
  'JP-TMA2-1-Scene-34':(1500,2000),
  'JP-TMA2-1-Scene-35':(1500,2000),
  'JP-TMA2-1-Scene-36':(1500,2000),
  'JP-TMA2-1-Scene-37':(1500,2000),
  'JP-TMA2-1-Scene-39':(1500,2000),
  'JP-TMA2-1-Scene-40':(1500,1500),
  'JP-TMA2-1-Scene-41':(1500,1500),
  }

tu_dim=(2000,3500)

#10-1 PNGs

#PNG parameters
d_overlay = {#'R1':['CD20','CD8','CD4','CK19'],
     #'R2':[ 'PCNA','HER2','ER','CD45'],
     #'R3':['pHH3', 'CK14', 'CD44', 'CK5'],
     #'R4':[ 'Vim', 'CK7', 'PD1', 'LamAC',],
     #'R5':['aSMA', 'CD68', 'Ki67', 'Ecad'],
     #'R6':['CK17','PDPN','CD31','CD3'],
     #'R7':['CK5R','CD8R','CD4R','CD20R'],
     #'R8':['LamB1','AR','ColIV','ColI'],
     #'subtype':['PCNA','HER2','ER','Ki67'],
     #'diff':['Ecad', 'CK14', 'CD44', 'CK5'],
     #'immune':['PD1','CD8R','CD4R','CD20R'],
     #'stromal':['aSMA','Vim','CD68','CD31'],
     #'subtype':['CD68','CK7','ER','HER2'],
     #'immune':['CK5','CK7','CD4','CD68'],
     #'diff':['CK5','ER','CK7','CD68'],
     'stromal':['ER','CK7','PDPN','aSMA'],
     }
es_bright = {'pHH3','CK14','CK5','CK17'} #'CD68',
high_thresh=0.998
'''
for s_sample in ls_sample:
    print(s_sample)
    if s_sample != 'JE-TMA-42':
        os.chdir(f'{subdir}/{s_sample}')
        df_img = mpimage.parse_org()
        d_crop = dict(zip(sorted(set(df_img.scene)),len(sorted(set(df_img.scene)))*[(2000,2000)]))
        if s_sample == 'JP-TMA1-1':
            for key, item in d_mod.items():
                d_crop.update({key:item})
        elif s_sample == 'JP-TMA2-1':
            for key, item in d_mod2.items():
                d_crop.update({key:item})
    os.chdir(codedir)
    for s_scene in sorted(d_crop.keys()):
        cmif.visualize_multicolor_overlay(s_scene,subdir,qcdir,d_overlay,d_crop,es_bright,high_thresh)
'''
#10 -2 ome-tiff

#ome-tiff parameters
b_resize = True
s_dapi = 'DAPI2'
d_combos = {
        'Stromal':{'PDPN','CD31','PDGFRa','aSMA','ColI','ColIV','BMP2'}, 
        'Tumor':{'HER2','ER','PgR','AR','EGFR','Ecad','Ki67'},
        'Immune':{'CD45','CD20','CD68','PD1', 'CD8', 'CD4','FoxP3','CD3','GRNZB'},
        'Differentiation':{'CK8','CK7','CK19','CK14','CK17','CK5','CD44','Vim'},
        'Growth': {'pAKT', 'pS6RP', 'CoxIV', 'pERK', 'Glut1','pHH3','pRB','PCNA'},
        'Other': {'H3K4','gH2AX', 'H3K27', 'HIF1a', 'cPARP','LamB2', 'LamB1', 'LamAC'},
        'DAPI':{'DAPI1','DAPI10','DAPI12'}
    }
'''
for s_sample in ls_sample:
    if s_sample != 'JE-TMA-42':
        os.chdir(f'{subdir}/{s_sample}')
        df_img = mpimage.parse_org()
        d_crop = dict(zip(sorted(set(df_img.scene)),len(sorted(set(df_img.scene)))*[(2000,2000)]))
        if s_sample == 'JP-TMA1-1':
            for key, item in d_mod.items():
                d_crop.update({key:item})
            d_crop =  {'JP-TMA1-1-Scene-028':(1800,800)} # (2300,800)(1800,3200)
            df = pd.read_csv(f'{segdir}/features_JP-TMA1-1_BboxCoords_JE.csv',index_col=0)
            for s_index in df.index: #[59::]:
                s_scene = s_index.replace('_scene','-Scene-')
                d_crop = {s_scene:(df.loc[s_index,'minc'],df.loc[s_index,'minr'])}
                tu_dim = (df.loc[s_index,'maxc'] - df.loc[s_index,'minc'],df.loc[s_index,'maxr'] - df.loc[s_index,'minr'])
                cmif.cropped_ometiff(s_scene,subdir,cropdir,d_crop,d_combos,s_dapi,tu_dim,b_8bit=True,b_resize=True)
                #cmif.load_crop_labels(d_crop,tu_dim,segdir,cropdir,s_find='exp5_CellSegmentationBasins',b_resize=True)
                cmif.load_crop_labels(d_crop,tu_dim,segdir,cropdir,s_find='Nuclei Segmentation Basins',b_resize=True)

        elif s_sample == 'JP-TMA2-1':
            for key, item in d_mod2.items():
                d_crop.update({key:item})
    #for s_scene in sorted(d_crop.keys()):
    #    cmif.cropped_ometiff(s_scene,subdir,cropdir,d_crop,d_combos,s_dapi,tu_dim)

    #10-3 crop basins to match cropped overlays
    #cmif.load_crop_labels(d_crop,tu_dim,segdir,cropdir,s_find='exp5_CellSegmentationBasins',b_resize=True)
    #cmif.load_crop_labels(d_crop,tu_dim,segdir,cropdir,s_find='Nuclei Segmentation Basins',b_resize=True)
'''

#### 11 Tissue edge detection ####
'''
from mplex_image import features
nuc_diam = 30
i_pixel = 153
for s_sample in ls_sample:
    features.edge_mask(s_sample,segdir,subdir,i_pixel=i_pixel, dapi_thresh=600,i_fill=250000) 
    df_sample = features.edge_cells(s_sample,segdir,nuc_diam,i_pixel=i_pixel)
    df_sample.to_csv(f'{segdir}/features_{s_sample}_EdgeCells{i_pixel}pixels_CentroidXY.csv')
'''
### 12 tissue bbox ###
'''
from mplex_image import features
nuc_diam = 30
i_pixel = 153
for s_sample in ls_sample:
    df_sample = features.edge_bbox(s_sample,segdir,i_pixel=i_pixel)
    df_sample.to_csv(f'{segdir}/features_{s_sample}_BboxCoords.csv')
'''

#Co-localization analysis

#functions
def pixel_pearson(x, y):
    try:
        r, p = scipy.stats.pearsonr(x.ravel(),y.ravel())
    except ValueError:
        #print(x)
        print(y)
        r=0
    return r

def manders_cc(a_R, a_G):
    '''
    m_one: fraction of first entry colocalized
    m_two: fraction of second entry colocalized
    '''
    m_one = (a_R & a_G).sum().sum()/a_R.sum()
    m_two = (a_R & a_G).sum().sum()/a_G.sum()
    return(m_one, m_two)

def thresh_img(x, i_thresh):
    diff = x - i_thresh
    return diff

def label_difference(labels,cell_labels):
    '''
    given matched nuclear and cell label IDs,return cell_labels minus labels
    '''
    overlap = cell_labels==labels
    ring_rep = cell_labels.copy()
    ring_rep[overlap] = 0
    return(ring_rep)

def extract_feat(labels,intensity_image, properties=('centroid','mean_intensity','area','eccentricity')):
    ''' 
    given labels and intensity image, extract features to dataframe
    '''
    props = measure.regionprops_table(labels,intensity_image, properties=properties)
    df_prop = pd.DataFrame(props)
    return(df_prop)

def costes_thresh(a_img,a_target):
    '''
    Costes et al. (14) developed a unique approach for automatically
    identifying the threshold value to be used to identify
    background based on an analysis that determines the range of
    pixel values for which a positive PCC is obtained. In this
    approach, PCC is measured for all pixels in the image and then
    again for pixels for the next lower red and green intensity
    values on the regression line. This process is repeated until
    pixel values are reached for which PCC drops to or below zero.
    '''
    p = np.polyfit(a_img.ravel(), a_target.ravel(), 1)
    #y = p[0]+p[1]x


#Co-localization analysis

d_coloc = {'Vim':['CK19','CK7','CK8','CK5','Ecad'],#
           'CD44':['CK19','CK7','CK8','CK5','Ecad'],
           'CK5':['CK19','CK7','CK8','Ecad'],
           'CK14':['CK19','CK7','CK8','Ecad'],
           'EGFR':['CK19','CK7','CK8','CK5','Ecad'],
    }

b_pearson = False

for s_sample in ls_sample:
    df_xy = pd.read_csv(f'{segdir}/features_{s_sample}_CentroidXY.csv',index_col=0)
    df_thresh = pd.read_csv(f'/home/groups/graylab_share/OMERO.rdsStore/engje/Data/20200000/20200406_JP-TMAs/data/thresh_JE_{s_sample}.csv',index_col=0)
    os.chdir(f'{subdir}/{s_sample}')
    df_img = mpimage.parse_org()
    for s_key, ls_item in d_coloc.items():
        print(s_key)
        df_marker = df_img[(df_img.marker==s_key)]
        df_coloc = pd.DataFrame()
        for s_scene in sorted(df_marker.scene.unique()):
            print(s_scene)
            #load segmentation
            cell_labels = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{s_scene}_CK7-cell30_exp5_CellSegmentationBasins.tif')
            labels = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{s_scene} nuclei30 - Nuclei Segmentation Basins.tif')
            ring_labels = label_difference(labels,cell_labels)
            #extract single cell intensity
            a_img = io.imread(df_marker[df_marker.scene==s_scene].index[0])
            df_prop = extract_feat(ring_labels,a_img, properties=('label','intensity_image'))
            #threshold
            if df_thresh.index.isin([s_scene.replace('-Scene-','_scene')]).any():
                    i_thresh = df_thresh.loc[s_scene.replace('-Scene-','_scene'),s_key]
            else:
                continue
            if math.isnan(i_thresh):
                i_thresh = df_thresh.loc['global',s_key]
            df_prop[f'{s_key}_thresh'] = df_prop.intensity_image.apply(lambda x: np.clip(a=(x - i_thresh*256), a_min=0,a_max=None))
            for s_target in ls_item:
                print(s_target)
                #extract single cell intensity
                a_target = io.imread(df_img[(df_img.marker==s_target) & (df_img.scene==s_scene)].index[0])
                df_prop_target = extract_feat(ring_labels,a_target, properties=('label','intensity_image'))
                #threshold
                i_thresh_tar = df_thresh.loc[s_scene.replace('-Scene-','_scene'),s_target]

                if math.isnan(i_thresh_tar):
                    i_thresh_tar = df_thresh.loc['global',s_target]
                df_prop[f'{s_target}_thresh'] = df_prop_target.intensity_image.apply(lambda x: np.clip(a=(x - i_thresh_tar*256), a_min=0,a_max=None))    
                #pearson
                if b_pearson:
                    df_prop[f'{s_key}-{s_target}_pearson'] = df_prop.intensity_image.combine(df_prop_target.intensity_image, pixel_pearson)
                #pearson w/ threshold
                df_prop[f'{s_key}-{s_target}_pearsont'] = df_prop.loc[:,f'{s_key}_thresh'].combine(df_prop.loc[:,f'{s_target}_thresh'], pixel_pearson).fillna(0)                        
                #manders
                # m_one: fraction of first entry colocalized
                # m_two: fraction of second entry colocalized
                se_key = df_prop.loc[:,f'{s_key}_thresh'].apply(lambda x: x > 0)
                se_target = df_prop.loc[:,f'{s_target}_thresh'].apply(lambda x: x > 0)
                se_tuple = se_key.combine(se_target, manders_cc)   
                df_prop[f'{s_key}-{s_target}_M1'] =  pd.Series([item[0] for item in se_tuple]).fillna(0)
                df_prop[f'{s_key}-{s_target}_M2'] =  pd.Series([item[1] for item in se_tuple]).fillna(0)
            df_prop.index = [f'{s_sample}_scene{s_scene.split("-Scene-")[1].split("_")[0]}_cell{item}' for item in df_prop.label]
            df_coloc = df_coloc.append(df_prop.loc[:,df_prop.dtypes=='float64'])
        df_coloc.to_csv(f'{segdir}/features_{s_sample}_Colocalization_{s_key}.csv')
        df_xy = df_xy.merge(df_coloc,left_index=True, right_index=True, how='left')
    df_xy.to_csv(f'{segdir}/features_{s_sample}_Colocalization.csv')


os.chdir(codedir)    

