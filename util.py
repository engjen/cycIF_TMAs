#load libraries

import os
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import math
import warnings


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
mpl.rc('figure', max_open_warning = 0)

import sklearn
from sklearn.preprocessing import minmax_scale, scale, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import exceptions
warnings.filterwarnings("ignore",category = exceptions.ApproximationWarning)

import scipy
from scipy import stats
from scipy.stats import entropy, norm
from scipy.spatial import cKDTree

import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import matplotlib

import anndata
from anndata import AnnData


codedir = os.getcwd()


#functions
def df_from_mcomp(m_comp):
    df_test = pd.DataFrame.from_records(m_comp.summary().data,coerce_float=True)
    df_test.columns=df_test.loc[0].astype('str')
    df_test.drop(0,inplace=True)
    df_test =df_test.apply(pd.to_numeric, errors='ignore')
    ls_order = pd.concat([df_test.group1,df_test.group2]).unique()
    return(df_test, ls_order)
def plt_sig(df_test,ax,ax_factor=5):
    #ls_order = df_test.group1.append(df_test.group2).unique()
    ls_order = pd.concat([df_test.group1,df_test.group2]).unique()
    props = {'connectionstyle':matplotlib.patches.ConnectionStyle.Bar(armA=0.0, armB=0.0, fraction=0.0, angle=None),
             'arrowstyle':'-','linewidth':.5}
    #draw on axes
    y_lim = ax.get_ylim()[1]
    y_lim_min = ax.get_ylim()[0]
    y_diff = y_lim-y_lim_min
    for count, s_index in enumerate(df_test[df_test.reject].index):
        text =f"p = {df_test.loc[s_index,'p-adj']:.1}"
        #text = "*"
        one = df_test.loc[s_index,'group1']
        two = df_test.loc[s_index,'group2']
        x_one = np.argwhere(ls_order == one)[0][0]
        x_two = np.argwhere(ls_order == two)[0][0]
        ax.annotate(text, xy=(np.mean([x_one,x_two]),y_lim - (y_diff+count)/ax_factor),fontsize=6)
        ax.annotate('', xy=(x_one,y_lim - (y_diff+count)/ax_factor), xytext=(x_two,y_lim - (y_diff+count)/ax_factor), arrowprops=props)
        #break
    return(ax)
def post_hoc(confusion_matrix):
    chi2, pvalue, dof, expected  = stats.chi2_contingency(confusion_matrix)
    observed_vals = confusion_matrix
    expected_vals = pd.DataFrame(expected,index=confusion_matrix.index,columns=confusion_matrix.columns)
    result_val = pd.DataFrame(data='',index=confusion_matrix.index,columns=confusion_matrix.columns)
    col_sum = observed_vals.sum(axis=1)
    row_sum = observed_vals.sum(axis=0)

    for indx in confusion_matrix.index:
        for cols in confusion_matrix.columns:
            observed = float(observed_vals.loc[indx,cols])
            expected = float(expected_vals.loc[indx,cols])
            col_total = float(col_sum[indx])
            row_total = float(row_sum[cols])
            expected_row_prop = expected/row_total
            expected_col_prop = expected/col_total
            std_resid = (observed - expected) / (math.sqrt(expected * (1-expected_row_prop) * (1-expected_col_prop)))
            p_val = norm.sf(abs(std_resid))
            if p_val < 0.05/(len(confusion_matrix.index)*len(confusion_matrix.columns)):
                print(indx,cols, "***", p_val)
                result_val.loc[indx,cols] = '***'
            elif p_val < 0.05:
                print (indx,cols, '*', p_val)
                result_val.loc[indx,cols] = '*'
            else:
                print (indx,cols, 'not sig', p_val)
    print('cutoff')
    print(0.05/(len(confusion_matrix.index)*len(confusion_matrix.columns)))
    return(result_val)

def single_var_km_cph(df_all,df_surv,s_subtype,s_platform,s_cell,alpha=0.05,min_cutoff=0.003,savedir=f"/home/groups/graylab_share/OMERO.rdsStore/engje/Data/20200000/20200406_JP-TMAs/20220408/Survival_Plots"):
    df_all.index = df_all.index.astype('str')
    df_surv.index = df_surv.index.astype('str')
    df_all = df_all.merge(df_surv.loc[:,['Survival','Survival_time','subtype','Platform']],left_index=True,right_index=True)
    if s_platform == 'IMC':
        df = df_all[(df_all.Platform==s_platform) & (~df_all.index.str.contains('Z')) & (df_all.subtype==s_subtype)].copy()
    elif s_platform == 'cycIF':
        df = df_all[(df_all.Platform==s_platform) & (~df_all.index.str.contains('JP-TMA2')) & (df_all.subtype==s_subtype)].copy()
    else:
        df = df_all[(df_all.Platform==s_platform) & (df_all.subtype==s_subtype)].copy()
    df = df.dropna() #df.dropna(axis=1).dropna()
    #KM
    for s_col in df.columns.drop(['Survival','Survival_time','subtype','Platform']):
        b_low = df.loc[:,s_col] <= df.loc[:,s_col].median()
        s_title1 = f'{s_subtype} {s_platform}'
        s_title2 = f'{s_cell} {s_col.replace(".","")}'
        if df.loc[:,s_col].median() < min_cutoff:
            continue
        elif len(df) < 1:
            continue
        df.loc[b_low,'abundance'] = 'low'
        df.loc[~b_low,'abundance'] = 'high'
        #log rank
        results = multivariate_logrank_test(event_durations=df.Survival_time,
                                            groups=df.abundance, event_observed=df.Survival)
        if results.summary.p[0] < alpha:
            print(s_col)
            #kaplan meier plotting
            kmf = KaplanMeierFitter()
            fig, ax = plt.subplots(figsize=(3,3),dpi=300)
            for s_group in ['high','low']:
                df_abun = df[df.abundance==s_group]
                durations = df_abun.Survival_time
                event_observed = df_abun.Survival
                try:
                    kmf.fit(durations, event_observed,label=s_group)
                    kmf.plot(ax=ax,ci_show=False,show_censors=True)
                except:
                    print('.')
            ax.set_title(f'{s_title1}\n{s_title2}\np={results.summary.p[0]:.2} (n={len(df)})',fontsize=10)
            ax.legend(loc='upper right',title=f'{df.loc[:,s_col].median():.2}')
            plt.tight_layout()
            fig.savefig(f"{savedir}/KM_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}.png",dpi=300)
        #CPH
        cph2 = CoxPHFitter(penalizer=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                cph2.fit(df.loc[:,[s_col,'Survival_time','Survival']], duration_col='Survival_time', event_col='Survival')
                if cph2.summary.p[0] < alpha:
                    print(s_col)
                    fig, ax = plt.subplots(figsize=(2.5,2),dpi=300)
                    cph2.plot(ax=ax)
                    ax.set_title(f'{s_title1} (n={len(df)})\n{s_title2}\np={cph2.summary.p[0]:.2} ({df.loc[:,s_col].median():.2})',fontsize=10)
                    ax.set_ylabel(f'{s_col}')
                    ax.set_yticklabels([])
                    plt.tight_layout()
                    fig.savefig(f"{savedir}/CPH_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}.png",dpi=300)
            except:
                print(f'skipped {s_col}')   
    return(df)

def cluster_leiden(adata, resolution,n_neighbors, s_subtype, s_type, s_partition, s_cell):
    sc.tl.leiden(adata,resolution=resolution)
    fig,ax = plt.subplots(figsize=(2.5,2),dpi=300)
    figname=f'both_{s_subtype}_{s_partition}_{s_cell}_{n_neighbors}_{resolution}.png'
    sc.pl.umap(adata, color='leiden',ax=ax,title=figname.split('.png')[0].replace('_',' '),wspace=.25,save=figname,size=40)
    return(adata)
def km_cph(adata,df_surv,s_subtype,s_plat,s_type,s_partition,s_cell,savedir=f'{codedir}/20220222/Survival_Plots_Both'):
    if type(adata) == anndata._core.anndata.AnnData:
        df_p = pd.DataFrame(data=adata.raw.X, index=adata.obs.index, columns=adata.var.index) #adata.to_df()
        df_p['Subtype'] = adata.obs.subtype
        df_p['leiden'] = adata.obs.leiden
        df_p['Platform'] = adata.obs.Platform
    else:
        df_p = adata
    df_p.index = df_p.index.astype('str')
    df_p['Survival'] = df_p.index.map(dict(zip(df_surv.index,df_surv.Survival)))
    df_p['Survival_time'] = df_p.index.map(dict(zip(df_surv.index,df_surv.Survival_time)))
    df_st = df_p[(df_p.Subtype==s_subtype)].dropna()
    if s_plat != 'Both':
        df_st = df_p[(df_p.Platform==s_plat) & (df_p.Subtype==s_subtype)].dropna()
    if not len(df_st) < 1:
        
        print(len(df_st)) 
        T = df_st['Survival_time']     ## time to event
        E = df_st['Survival']      ## event occurred or censored
        groups = df_st.loc[:,'leiden'] 
        kmf1 = KaplanMeierFitter() ## instantiate the class to create an object
        fig, ax = plt.subplots(figsize=(3,3),dpi=200)
        for idx, s_group in enumerate(sorted(df_p.leiden.unique())):
            i1 = (groups == s_group)
            if sum(i1) > 0:
                kmf1.fit(T[i1], E[i1], label=s_group)    ## fit thedata
                kmf1.plot(ax=ax,ci_show=False,color=f'C{idx}',show_censors=True)
                print(f'{s_group}: {kmf1.median_survival_time_}, {kmf1.percentile(.75)} ({i1.sum()})')
        results = multivariate_logrank_test(event_durations=T, groups=groups, event_observed=E)
        ax.set_title(f'{s_subtype} {s_plat} {s_cell} \n  p={results.summary.p[0]:.1} n={len(df_st)}') #res={resolution}
        ax.legend(loc='upper right')
        ax.set_ylim(-0.05,1.05)
        plt.tight_layout()
        #CPH
        df_dummy = pd.get_dummies(df_st.loc[:,['Survival_time','Survival','leiden']])
        df_dummy = df_dummy.loc[:,df_dummy.sum() != 0]
        cph = CoxPHFitter(penalizer=0.1)  ## Instantiate the class to create a cph object
        cph.fit(df_dummy, 'Survival_time', event_col='Survival')
        fig2, ax2 = plt.subplots(figsize=(2.5,3),dpi=200)
        cph.plot(ax=ax2)
        pvalue = cph.summary.loc[:,'p'].min()
        ax2.set_title(f'CPH: {s_subtype} {s_plat} {s_cell}\np={pvalue:.2}')
        plt.tight_layout()
    else:
        fig = None
        fig2 = None
    return(df_p, fig,fig2)

def km_cph_entropy(df_p,df,ls_col,s_subtype,s_plat,s_cell,savedir=f'{codedir}/20220222/Survival_Plots_Both'):
    df_p['entropy'] = entropy(df_p.loc[:,df_p.columns[df_p.dtypes=='float32']].fillna(0),axis=1,base=2)
    df_st = df_p[(df_p.Subtype==s_subtype)].dropna()
    if s_plat != 'Both':
        df_st = df_p[(df_p.Platform==s_plat) & (df_p.Subtype==s_subtype)].dropna()
    #######3 Entropy
    s_col = 'entropy'
    # no df and ls_col variable
    df_st = df.loc[:,ls_col].merge(df_st.loc[:,['Subtype','Platform','Survival','Survival_time','entropy']],left_index=True,right_index=True)
    if not len(df_st) < 1:
        b_low = df_st.loc[:,s_col] <= df_st.loc[:,s_col].median()
        if df_st.loc[:,s_col].median() == 0:
            b_low = df.loc[:,s_col] <= 0
        df_st.loc[b_low,'abundance'] = 'low'
        df_st.loc[~b_low,'abundance'] = 'high'
        kmf = KaplanMeierFitter()
        results = multivariate_logrank_test(event_durations=df_st.Survival_time, groups=df_st.abundance, event_observed=df_st.Survival)
        print(f'entropy {results.summary.p[0]}')
        if results.summary.p[0] < 0.2:
            fig, ax = plt.subplots(figsize=(3,3),dpi=200)
            for s_group in ['high','low']:
                    df_abun = df_st[df_st.abundance==s_group]
                    durations = df_abun.Survival_time
                    event_observed = df_abun.Survival
                    kmf.fit(durations, event_observed,label=s_group)
                    kmf.plot(ax=ax,ci_show=False,show_censors=True)
            s_title1 = f'{s_subtype} {s_plat}'
            s_title2 = f'{s_cell} {s_col}'
            ax.set_title(f'{s_title1}\n{s_title2}\np={results.summary.p[0]:.2}',fontsize=10)
            ax.legend(loc='upper right')
            plt.tight_layout()
            fig.savefig(f"{savedir}/KM_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}.png",dpi=300)
            cph = CoxPHFitter(penalizer=0.1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    cph.fit(df_st.loc[:,[s_col,'Survival','Survival_time']], duration_col='Survival_time', event_col='Survival')
                    if cph.summary.p[0] < 0.1:
                        print(s_col)
                        fig, ax = plt.subplots(figsize=(2.5,2),dpi=200)
                        cph.plot(ax=ax)
                        s_title1 = f'{s_subtype} {s_plat}'
                        s_title2 = f'{s_cell} {s_col}'
                        ax.set_title(f'{s_title1}\n{s_title2}\np={cph.summary.p[0]:.2}',fontsize=10)
                        plt.tight_layout()
                        fig.savefig(f"{savedir}/CPH_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}.png",dpi=300)
                except:
                    print(f'skipped {s_col}')   


def group_median_diff(df_marker,s_group,s_marker):
    lls_result = []
    for s_test in df_marker.loc[:,s_group].dropna().unique():
        ls_result = df_marker.loc[df_marker.loc[:,s_group] == s_test,s_marker].values
        lls_result.append(ls_result)
    if len(lls_result)==2:
        try:
            statistic, pvalue = stats.mannwhitneyu(lls_result[0],lls_result[1])
        except:
            print('error in group median diff mannwhitney')
            pvalue = 1.00
            statistic = None
    elif len(lls_result) > 2:
        try:
            statistic, pvalue = stats.kruskal(*lls_result,nan_policy='omit')
        except:
            print('error in group median diff kruskal')
            pvalue = 1.00
            statistic = None
    else:
        #print('no groups found')
        pvalue = None
        statistic = None
    #print(pvalue)
    return(statistic,pvalue)


#functions
def silheatmap(adata,clust,marker_list,sil_key):
    cluster_list = [str(item) for item in adata.uns[f'dendrogram_{clust}']['categories_ordered']]
    #dataframe
    df = adata.to_df()
    df[clust] = adata.obs[clust]
    #sort by sil
    df[sil_key] = adata.obs[sil_key]
    df = df.sort_values(by=sil_key)
    #sort by cluster, markers
    df['old_index'] = df.index
    obs_tidy = df.set_index(clust)
    obs_tidy.index = obs_tidy.index.astype('str')
    obs_tidy = obs_tidy.loc[cluster_list,:]
    df = df.loc[obs_tidy.old_index]
    obs_tidy = obs_tidy.loc[:,marker_list]
    #scale
    obs_tidy = pd.DataFrame(data=minmax_scale(obs_tidy),index=obs_tidy.index,columns=obs_tidy.columns)
    # define a layout of 3 rows x 3 columns
    # The first row is for the dendrogram (if not dendrogram height is zero)
    # second row is for main content. This col is divided into three axes:
    #   first ax is for the heatmap
    #   second ax is for 'brackets' if any (othwerise width is zero)
    #   third ax is for colorbar
    colorbar_width = 0.2
    var_names = marker_list
    width = 10
    dendro_height = 0.8 #if dendrogram else 0
    groupby_height = 0.13 #if categorical else 0
    heatmap_height = len(var_names) * 0.18 + 1.5
    height = heatmap_height + dendro_height + groupby_height + groupby_height
    height_ratios = [dendro_height, heatmap_height, groupby_height,groupby_height]
    width_ratios = [width, 0, colorbar_width, colorbar_width]
    fig = plt.figure(figsize=(width, height),dpi=200)
    axs = gridspec.GridSpec(
        nrows=4,
        ncols=4,
        wspace=1 / width,
        hspace=0.3 / height,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False)
    norm2 = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)

    # plot heatmap
    heatmap_ax = fig.add_subplot(axs[1, 0])
    im = heatmap_ax.imshow(obs_tidy.T.values, aspect='auto',norm=norm,interpolation='nearest') # ,interpolation='nearest'
    heatmap_ax.set_xlim(0 - 0.5, obs_tidy.shape[0] - 0.5)
    heatmap_ax.set_ylim(obs_tidy.shape[1] - 0.5, -0.5)
    heatmap_ax.tick_params(axis='x', bottom=False, labelbottom=False)
    heatmap_ax.set_xlabel('')
    heatmap_ax.grid(False)
    heatmap_ax.tick_params(axis='y', labelsize='small', length=1)
    heatmap_ax.set_yticks(np.arange(len(var_names)))
    heatmap_ax.set_yticklabels(var_names, rotation=0)

    #colors
    value_sum = 0
    ticks = []  # list of centered position of the labels
    labels = []
    label2code = {}  # dictionary of numerical values asigned to each label
    for code, (label, value) in enumerate(
            obs_tidy.index.value_counts().loc[cluster_list].iteritems()
        ):
            ticks.append(value_sum + (value / 2))
            labels.append(label)
            value_sum += value
            label2code[label] = code

    groupby_cmap = mpl.colors.ListedColormap(adata.uns[f'{clust}_colors'])
    groupby_ax = fig.add_subplot(axs[3, 0])
    groupby_ax.imshow(
                np.array([[label2code[lab] for lab in obs_tidy.index]]),
                aspect='auto',
                cmap=groupby_cmap,
            )
    groupby_ax.grid(False)
    groupby_ax.yaxis.set_ticks([])
    groupby_ax.set_xticks(ticks,labels,fontsize='xx-small',rotation=90)
    groupby_ax.set_ylabel('Cluster',fontsize='x-small',rotation=0,ha='right',va='center')


    #sil
    sil_ax = fig.add_subplot(axs[2, 0])
    #max_index = df[sil_key].idxmax()    #df.loc[max_index,sil_key] = 1    #min_index = df[sil_key].idxmin()    #df.loc[min_index,sil_key] = -1 #not needed
    a=np.array([df[sil_key]]) #f'{clust}_silhuette'
    a_tile = np.tile(a,(int(len(df)/80),1))
    sil_ax.imshow(a_tile,cmap='bwr',norm=norm2)
    sil_ax.xaxis.set_ticks([])
    sil_ax.yaxis.set_ticks([])
    sil_ax.set_ylabel('Silhouette',fontsize='x-small',rotation=0,ha='right',va='center')
    sil_ax.grid(False)

    #dendrogram
    dendro_ax = fig.add_subplot(axs[0, 0], sharex=heatmap_ax)
    #_plot_dendrogram(dendro_ax, adata, groupby, dendrogram_key=dendrogram,ticks=ticks, orientation='top', )
    dendro_info = adata.uns[f'dendrogram_{clust}']['dendrogram_info']
    leaves = dendro_info["ivl"]
    icoord = np.array(dendro_info['icoord'])
    dcoord = np.array(dendro_info['dcoord'])
    orig_ticks = np.arange(5, len(leaves) * 10 + 5, 10).astype(float)
    for xs, ys in zip(icoord, dcoord):
        if ticks is not None:
            xs = translate_pos(xs, ticks, orig_ticks)
        dendro_ax.plot(xs, ys, color='#555555')
    dendro_ax.tick_params(bottom=False, top=False, left=False, right=False)
    ticks = ticks if ticks is not None else orig_ticks
    dendro_ax.set_xticks(ticks)
    #dendro_ax.set_xticklabels(leaves, fontsize='small', rotation=90)
    dendro_ax.set_xticklabels([])
    dendro_ax.tick_params(labelleft=False, labelright=False)
    dendro_ax.grid(False)
    dendro_ax.spines['right'].set_visible(False)
    dendro_ax.spines['top'].set_visible(False)
    dendro_ax.spines['left'].set_visible(False)
    dendro_ax.spines['bottom'].set_visible(False)

    # plot colorbar
    cbar_ax = fig.add_subplot(axs[1, 2])
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = plt.colorbar(mappable=mappable, cax=cbar_ax)
    cbar_ax.tick_params(axis='both', which='major', labelsize='xx-small',rotation=90,length=.1)
    cbar_ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(locs=[0,1]))
    cbar.set_label('Expression', fontsize='xx-small',labelpad=-5)

    # plot colorbar2
    cbar_ax = fig.add_subplot(axs[1, 3])
    mappable = mpl.cm.ScalarMappable(norm=norm2, cmap='bwr')
    cbar = plt.colorbar(mappable=mappable, cax=cbar_ax)
    cbar_ax.tick_params(axis='both', which='major', labelsize='xx-small',rotation=90,length=.1)
    cbar_ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(locs=[-1,0,1]))
    cbar.set_label('Silhouette Score', fontsize='xx-small',labelpad=0)

    #return dict
    return_ax_dict = {'heatmap_ax': heatmap_ax}
    return_ax_dict['groupby_ax'] = groupby_ax
    return_ax_dict['dendrogram_ax'] = dendro_ax
    return(fig)

def translate_pos(pos_list, new_ticks, old_ticks):
    """
    transforms the dendrogram coordinates to a given new position.
    """
    # of given coordinates.

    if not isinstance(old_ticks, list):
        # assume that the list is a numpy array
        old_ticks = old_ticks.tolist()
    new_xs = []
    for x_val in pos_list:
        if x_val in old_ticks:
            new_x_val = new_ticks[old_ticks.index(x_val)]
        else:
            # find smaller and bigger indices
            idx_next = np.searchsorted(old_ticks, x_val, side="left")
            idx_prev = idx_next - 1
            old_min = old_ticks[idx_prev]
            old_max = old_ticks[idx_next]
            new_min = new_ticks[idx_prev]
            new_max = new_ticks[idx_next]
            new_x_val = ((x_val - old_min) / (old_max - old_min)) * (
                new_max - new_min
            ) + new_min
        new_xs.append(new_x_val)
    return new_xs

#functions

# count the neighbors.
class NeighborsCounter:

    def __init__(self, rad, xy=['CentroidX', 'CentroidY']):
        self.rad = rad
        self.xy = xy

    def query_balltree_vanilla(self, coords_np):
        """
        input coords_np:
            these are coordinates. possible shape: (N,2)

        output neighbor_indices:
            this is a list of lists.
            there is one list per row in coords_np (i.e. there are N)
            the i'th list contains the indices of the neighbors of i,
            not including itself.
        """
        n_points = coords_np.shape[0]
        print(f'Counting neighbors for {n_points} points.')

        tree = cKDTree(coords_np)
        neighbor_indices = tree.query_ball_tree(tree, self.rad)
        for i in range(n_points):
            neighbor_indices[i].remove(i)
        return neighbor_indices

    def run(self, dataframe):
        """
        Splits the input dataframe into cell types and coordinates
        Runs query_balltree_vanilla on the coordinates
        Uses the neighbor indices to get cell type neighbor counts.

        Input:
            a dataframe with boolean cell type columns and coordinate columns
            the coordinate columns by default are named ['CentroidX', 'CentroidY']
            (coordinate column names are stored in attribute self.xy)

        Output:
            a dataframe with the same shape and index as the input dataframe.
        """

        types = [c for c in dataframe.columns if c not in self.xy]
        #why do we have to do this?
        types.remove('slide')
        g = self.query_balltree_vanilla(dataframe[self.xy].to_numpy())
        counts = np.zeros((len(g), len(types)))
        df_arra = dataframe[types].to_numpy()

        #return(counts)
        
        for n in range(dataframe.shape[0]):
            idx = np.array(g[n])
            if idx.size:
                counts[n, :] = df_arra[idx, :].sum(axis=0)

        return pd.DataFrame(counts, index=dataframe.index, columns=types)
        

        

def km_cph_all(df_both,df_clin,s_title1,s_title2,s_marker,alpha=0.05,s_time='Survival_time', s_censor='Survival',
               s_groups='abundance',s_cph_model='high',ls_clin=['age','tumor_size','Stage'],p_correct=None):
    '''
    df_both must have s_time, s_censor, s_groups
    s_marker: rename anudance_high into somthing more meaningful for CPH plots
    df_clin: clinical covariates data frame
    ls_clin = clinical covariates columns
    '''
    ### log rank ###
    #print(len(df_both))
    if len(df_both) > 0:
        results = multivariate_logrank_test(event_durations=df_both.loc[:,s_time],
            groups=df_both.loc[:,s_groups], event_observed=df_both.loc[:,s_censor])
        pvalue_km = results.summary.p[0]
    else: 
        pvalue_km = 1
    #kaplan meier plotting
    if pvalue_km < alpha:
        kmf = KaplanMeierFitter()
        fig1, ax = plt.subplots(figsize=(3,3),dpi=300)
        for s_group in sorted(df_both.loc[:,s_groups].unique()):
            df_abun = df_both[df_both.loc[:,s_groups]==s_group]
            durations = df_abun.loc[:,s_time]
            event_observed = df_abun.loc[:,s_censor]
            kmf.fit(durations, event_observed,label=s_group) #try:#except:#results.summary.p[0] = 1
            kmf.plot(ax=ax,ci_show=False,show_censors=True)
        ax.set_title(f'{s_title1}\n{s_title2}\n p={pvalue_km:.2} n={len(df_both)}')
        if not p_correct is None:
            ax.set_title(f'{s_title1}\n{s_title2}\n p_corrected={p_correct:.2} n={len(df_both)}')
        ax.legend(loc='upper right',title=f'{s_groups}')
        ax.set_xlabel(s_time)
        plt.tight_layout()  
    else:
        fig1 = None
    ##### CPH ######
    cph = CoxPHFitter(penalizer=0.1)
    try:
        df_dummy = pd.get_dummies(df_both).loc[:,[s_time,s_censor,f'{s_groups}_{s_cph_model}']]
        df_dummy = df_dummy.rename({f'{s_groups}_{s_cph_model}':s_marker},axis=1)
        df_dummy.index = df_dummy.index.astype('str')
        df_marker = df_dummy.merge(df_clin,left_index=True,right_index=True).loc[:,[s_time,s_censor,s_marker] + ls_clin]
        df_marker = df_marker.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #multi
            cph.fit(df_marker, s_time, event_col=s_censor) 
            pvalue = cph.summary.loc[s_marker,'p']
    except:
        pvalue = 1
    if pvalue < alpha:
        fig2, ax = plt.subplots(figsize=(3.2,2),dpi=200)
        cph.plot(ax=ax)
        ax.set_title(f'{s_title1}\n{s_title2}\n{s_censor} p={pvalue:.2} n={len(df_marker)}')
        plt.tight_layout()
    else:
        fig2 = None
    return(fig1, fig2, pvalue,pvalue_km)


def plt_sig2(df_test,ax):
    ls_order = pd.concat([df_test.group1,df_test.group2]).unique()
    props = {'connectionstyle':matplotlib.patches.ConnectionStyle.Bar(armA=0.0, armB=0.0, fraction=0.0, angle=None),
             'arrowstyle':'-','linewidth':.5}
    #draw on axes
    y_lim = ax.get_ylim()[1]
    y_lim_min = ax.get_ylim()[0]
    y_diff = (y_lim-y_lim_min)/10
    for count, s_index in enumerate(df_test[df_test.reject].index):
        y_test = (y_diff+count*y_diff)
        text =f"p = {df_test.loc[s_index,'p-adj']:.1}"
        one = df_test.loc[s_index,'group1']
        two = df_test.loc[s_index,'group2']
        x_one = np.argwhere(ls_order == one)[0][0]
        x_two = np.argwhere(ls_order == two)[0][0]
        ax.annotate(text, xy=(np.mean([x_one,x_two]),y_lim - y_test),fontsize=6)
        ax.annotate('', xy=(x_one,y_lim - y_test), xytext=(x_two,y_lim - y_test), arrowprops=props)
        #break
    return(ax)

def more_plots(adata,df_p,s_subtype,s_type,s_partition,s_cell,n_neighbors,resolution,z_score,linkage,
               s_color_p='Platform',d_color_p = {'cycIF':'gold','IMC':'darkblue'},savedir=f'{codedir}/20220222/Survival_Plots_Both'):
    #more plots
    #color by platform/leiden
    from matplotlib.pyplot import gcf
    d_color = dict(zip(sorted(adata.obs.leiden.unique()),sns.color_palette()[0:len(adata.obs.leiden.unique())]))
    
    network_colors = df_p.leiden.astype('str').map(d_color)#
    network_colors.name = 'cluster'
    node_colors  = df_p.loc[:,s_color_p].astype('str').map(d_color_p)
    network_node_colors = pd.DataFrame(node_colors).join(pd.DataFrame(network_colors))
    ls_col = df_p.drop(['Subtype', 'leiden', 'Platform','Survival', 'Survival_time'],axis=1).columns.tolist()
    g = sns.clustermap(df_p.loc[:,ls_col].dropna(),figsize=(7,6),cmap='viridis',z_score=z_score,
            row_colors=network_node_colors,method=linkage,dendrogram_ratio=0.16)
    for label,color in d_color_p.items():
        g.ax_col_dendrogram.bar(0, 0, color=color,label=label, linewidth=0)
    l1 = g.ax_col_dendrogram.legend(loc="right", ncol=1,bbox_to_anchor=(-0.1, 0.72),bbox_transform=gcf().transFigure)
    for label,color in d_color.items():
        g.ax_row_dendrogram.bar(0, 0, color=color,label=label, linewidth=0)
    l2 = g.ax_row_dendrogram.legend(loc="right", ncol=1,bbox_to_anchor=(-0.1, 0.5),bbox_transform=gcf().transFigure)
    g.savefig(f'{savedir}/clustermap_PlatformandSubtype_{s_type}_{s_partition}_{s_cell}_{s_type}_{n_neighbors}_{resolution}.png',dpi=200)

    #subtypes' mean
    d_replace = {}
    df_plot = df_p.loc[:,ls_col+['leiden']].dropna().groupby('leiden').mean()
    df_plot.index.name = f'leiden {resolution}'
    #fig,ax=plt.subplots(dpi=300,figsize=(4,len(ls_col)*.25+1))
    g = sns.clustermap(df_plot.dropna().T,z_score=z_score,cmap='RdBu_r',vmin=-2,vmax=2,method='ward',figsize=(4,len(ls_col)*.25+1),dendrogram_ratio=0.1,cbar_kws={"orientation": "horizontal"})
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 0.99, g.ax_row_dendrogram.get_position().width *2.5, 0.02])
    #fig.suptitle(f'leiden {resolution}',x=.9) 
    g.savefig(f'{savedir}/clustermap_mean_subtypes_{s_type}_{s_partition}_{s_cell}_{s_type}_{n_neighbors}_{resolution}.png',dpi=300)
    marker_genes = df_plot.dropna().T.iloc[:,g.dendrogram_col.reordered_ind].columns.tolist()
    categories_order = df_plot.dropna().T.iloc[g.dendrogram_row.reordered_ind,:].index.tolist()
    #barplot
    fig,ax=plt.subplots(figsize=(2.5,2.5),dpi=300)
    df_p.groupby(['leiden','Platform','Subtype']).count().iloc[:,0].unstack().loc[marker_genes].plot(kind='barh',title='Patient Count',ax=ax)
    plt.tight_layout()
    fig.savefig(f'{savedir}/barplot_subtyping_{s_type}_{s_partition}_{s_cell}_{s_type}_{n_neighbors}_{resolution}.png')


## find best cutpoint 
def low_high_abun(df_all,s_subtype,s_plat,s_col):
    df_all.index = df_all.index.astype('str')
    df = df_all[(df_all.Platform==s_plat) & (df_all.subtype==s_subtype)].copy()
    if len(df) > 0:
        #KM
        i_cut = np.quantile(df.loc[:,s_col],cutp)
        b_low = df.loc[:,s_col] <= i_cut
        if i_cut == 0:
            b_low = df.loc[:,s_col] <= 0
        df.loc[b_low,'abundance'] = 'low'
        df.loc[~b_low,'abundance'] = 'high'
    return(df)
    

# def single_km(df_all,s_cell,s_subtype,s_plat,s_col,savedir,alpha=0.05,cutp=0.5,s_time='Survival_time',
#               s_censor='Survival',s_propo='in'):
#     df_all.index = df_all.index.astype('str')
#     df = df_all[(df_all.Platform==s_plat) & (df_all.subtype==s_subtype)].copy()
#     df = df.loc[:,[s_col,s_time,s_censor]].dropna()
#     if len(df) > 0:
#         #KM
#         i_cut = np.quantile(df.loc[:,s_col],cutp)
#         b_low = df.loc[:,s_col] <= i_cut
#         s_title1 = f'{s_subtype} {s_plat}'
#         s_title2 = f'{s_col} {s_propo} {s_cell}'
#         if i_cut == 0:
#             b_low = df.loc[:,s_col] <= 0
#         df.loc[b_low,'abundance'] = 'low'
#         df.loc[~b_low,'abundance'] = 'high'
#         #log rank
#         results = multivariate_logrank_test(event_durations=df.loc[:,s_time],
#                                             groups=df.abundance, event_observed=df.loc[:,s_censor])
#         #kaplan meier plotting
#         if results.summary.p[0] < alpha:
#             kmf = KaplanMeierFitter()
#             fig, ax = plt.subplots(figsize=(3,3),dpi=300)
#             for s_group in ['high','low']:
#                 df_abun = df[df.abundance==s_group]
#                 durations = df_abun.loc[:,s_time]
#                 event_observed = df_abun.loc[:,s_censor]
#                 try:
#                     kmf.fit(durations, event_observed,label=s_group)
#                     kmf.plot(ax=ax,ci_show=False,show_censors=True)
#                 except:
#                     results.summary.p[0] = 1
#             ax.set_title(f'{s_title1}\n{s_title2}\nn={len(df)} p={results.summary.p[0]:.2}',fontsize=10)
#             ax.set_xlabel(s_censor)
#             ax.legend(loc='upper right',title=f'{cutp}({i_cut:.2})')
#             plt.tight_layout()
#             fig.savefig(f"{savedir}/Survival_Plots/KM_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}_{cutp}_{s_censor}.png",dpi=300)
#         else:
#             print('no survival data to fit')
#         return(df)

def single_km(df_all,s_cell,s_subtype,s_plat,s_col,savedir,alpha=0.05,cutp=0.5,s_time='Survival_time',
              s_censor='Survival',s_propo='in'):
    df_all.index = df_all.index.astype('str')
    df = df_all[(df_all.Platform==s_plat) & (df_all.subtype==s_subtype)].copy()
    df = df.loc[:,[s_col,s_time,s_censor]].dropna()
    if len(df) > 0:
        #KM
        i_cut = np.quantile(df.loc[:,s_col],cutp)
        b_low = df.loc[:,s_col] <= i_cut
        s_title1 = f'{s_subtype} {s_plat}'
        s_title2 = f'{s_col} {s_propo} {s_cell}'
        if i_cut == 0:
            b_low = df.loc[:,s_col] <= 0
        df.loc[b_low,'abundance'] = 'low'
        df.loc[~b_low,'abundance'] = 'high'
        #log rank
        results = multivariate_logrank_test(event_durations=df.loc[:,s_time],
                                            groups=df.abundance, event_observed=df.loc[:,s_censor])
        #kaplan meier plotting
        if results.summary.p[0] < alpha:
            kmf = KaplanMeierFitter()
            fig, ax = plt.subplots(figsize=(3,3),dpi=300)
            for s_group in ['high','low']:
                df_abun = df[df.abundance==s_group]
                durations = df_abun.loc[:,s_time]
                event_observed = df_abun.loc[:,s_censor]
                try:
                    kmf.fit(durations, event_observed,label=s_group)
                    kmf.plot(ax=ax,ci_show=False,show_censors=True)
                except:
                    results.summary.p[0] = 1
            ax.set_title(f'{s_title1}\n{s_title2}\nn={len(df)} p={results.summary.p[0]:.2}',fontsize=10)
            ax.set_xlabel(s_censor)
            ax.legend(loc='upper right',title=f'{cutp}({i_cut:.2})')
            plt.tight_layout()
            fig.savefig(f"{savedir}/Survival_Plots/KM_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}_{cutp}_{s_censor}.pdf")
        return(df)


warnings.filterwarnings("default",category = exceptions.ApproximationWarning)


def make_adata(df, ls_col,df_surv, n_neighbors, s_subtype, s_type, s_partition, s_cell,ncols=4):#b_norm=True,
    print('making adata')
    adata = sc.AnnData(df.loc[:,ls_col].fillna(0))
    adata.raw = adata
    #platform
    adata.obs['Platform'] = adata.obs.index.astype('str').map(dict(zip(df_surv.index.astype('str'),df_surv.Platform)))
    adata.obs['Platform'] = adata.obs.Platform.fillna('cycIF')
    #subtype
    adata.obs['subtype'] = adata.obs.index.astype('str').map(dict(zip(df_surv.index.astype('str'),df_surv.subtype)))
    #CAREFUL
    adata.obs['subtype'] = adata.obs['subtype'].fillna('TNBC')
    #subtype
    #reduce dimensionality
    sc.tl.pca(adata, svd_solver='auto')
    print('scaling')
    #scale
    sc.pp.scale(adata, zero_center=False, max_value=20)
    print('calc umap')
    # calculate neighbors 
    sc.pp.neighbors(adata, n_neighbors=n_neighbors) 
    sc.tl.umap(adata)
    #platform
    fig,ax=plt.subplots(dpi=300,figsize=(2.5,2))
    figname = f"Umapboth_Platform_{s_subtype}_{s_type}_{s_partition}_{s_cell}_{n_neighbors}neigh.png"
    title=figname.split('.png')[0].replace('_',' ')
    sc.pl.umap(adata, color='Platform',save=figname,size=40,ax=ax)
    #color by markers   
    figname = f"Umapboth_markers_{s_subtype}_{s_type}_{s_partition}_{s_cell}_{n_neighbors}neigh.png"
    title=figname.split('.png')[0].replace('_',' ')
    sc.pl.umap(adata, color=ls_col,vmin='p1.5',vmax='p99.5',ncols=ncols,save=figname,size=250)
    #color by subtype
    fig,ax=plt.subplots(dpi=300,figsize=(2.5,2))
    figname = f"Umapboth_subtype_{s_subtype}_{s_type}_{s_partition}_{s_cell}_{n_neighbors}neigh.png"
    title=figname.split('.png')[0].replace('_',' ')
    sc.pl.umap(adata, color='subtype',save=figname,size=40,ax=ax)
    return(adata) #adata_norm

def patient_heatmap(df_p,ls_col,ls_annot,z_score=0,figsize=(7,6),linkage='complete',
                    ls_color=[mpl.cm.tab10.colors,mpl.cm.Set2.colors,mpl.cm.Set1.colors[::-1],
                              mpl.cm.Paired.colors,mpl.cm.Pastel1.colors,mpl.cm.Set3.colors]):
    #more plots
    #color by platform/leiden
    from matplotlib.pyplot import gcf
    
    #
    df_annot = pd.DataFrame()
    dd_color = {}
    for idx, s_annot in enumerate(ls_annot):
        color_palette = ls_color[idx]
        d_color = dict(zip(sorted(df_p.loc[:,s_annot].dropna().unique()),color_palette)) #[0:len(df_p.loc[:,s_annot].dropna().unique())]))
        network_colors = df_p.loc[:,s_annot].astype('str').map(d_color) 
        df_annot[s_annot] = pd.DataFrame(network_colors)
        dd_color.update({s_annot:d_color})
    try:
        g = sns.clustermap(df_p.loc[:,ls_col],figsize=figsize,cmap='RdBu_r',z_score=z_score,
            row_colors=df_annot,method=linkage,dendrogram_ratio=0.1,xticklabels=1,vmin=-4,vmax=4,#yticklabels=None,
                          cbar_kws={"orientation": "horizontal"})
        x0, _y0, _w, _h = g.cbar_pos
        g.ax_cbar.set_position([x0, 0.95, g.ax_row_dendrogram.get_position().width *2.5, 0.02])
        for idx, (s_annot, d_color) in enumerate(dd_color.items()):
            g.ax_col_dendrogram.bar(0, 0, color='w',label=' ', linewidth=0)
            for label,color in d_color.items():
                g.ax_col_dendrogram.bar(0, 0, color=color,label=label, linewidth=0)
        
        l1 = g.ax_col_dendrogram.legend(loc="right", ncol=1,bbox_to_anchor=(0, 0.6),bbox_transform=gcf().transFigure)
    except:
        print('clustermap error')
        g= df_p.loc[:,ls_col].dropna(how='any')
    return(g,df_annot)        
        
def plot_sil(d_sil,s_name='Tumor'):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(dpi=200)
    pd.Series(d_sil).plot(ax=ax)
    ax.set_title(f'{s_name}: Mean Silhoutte Scores')
    ax.set_xlabel('k')
    plt.tight_layout()
    fig.savefig(f'{s_name}_Silhouette.png')
    
def km_pvalue(df,s_col,s_time,s_censor,cutp=0.5):
    i_cut = np.quantile(df.loc[:,s_col],cutp)
    b_low = df.loc[:,s_col] <= i_cut
    if i_cut == 0:
        b_low = df.loc[:,s_col] <= 0
    df.loc[b_low,'abundance'] = 'low'
    df.loc[~b_low,'abundance'] = 'high'
    #log rank
    results = multivariate_logrank_test(event_durations=df.loc[:,s_time],
                                        groups=df.abundance, event_observed=df.loc[:,s_censor])
    pvalue = results.summary.p[0]
    d_result = {}
    for s_group in ['high','low']:
        kmf = KaplanMeierFitter()
        df_abun = df[df.abundance==s_group]
        durations = df_abun.loc[:,s_time]
        event_observed = df_abun.loc[:,s_censor]
        try:
            kmf.fit(durations, event_observed,label=s_group)
            d_result.update({s_group:kmf.median_survival_time_})
            if math.isinf(kmf.median_survival_time_):
                d_result.update({s_group:kmf.percentile(.8)})
        except:
            d_result.update({s_group:np.nan})
    #if d_result['high']!=d_result['low']:
    try:
        median_diff = d_result['high']-d_result['low']
    except:
        median_diff = np.nan
    return(pvalue, median_diff,d_result)

# func
def make_mean(df,s_plat,s_center,s_subtype,s_col,s_center_column='leiden'): #leidencelltype5
    if s_center_column is None:
        df_mean = df.loc[((df.subtype==s_subtype) & (df.Platform==s_plat)),[s_col,'Patient']].groupby('Patient').mean()
    else:
        df_mean = df.loc[((df.loc[:,s_center_column]==s_center) & (df.Platform==s_plat) & (df.subtype==s_subtype) ),[s_col,'Patient']].groupby('Patient').mean()
    df_mean.index = df_mean.index.astype('str')
    #d_mean.update({s_plat:df_mean})
    return(df_mean) 

def run_multi_test(d_data,df_clin,ls_pval,s_discovery,s_subtype,s_censor,s_time,alpha,s_propo='neighbors of',
                   s_center_column='leiden',savedir=f'Survival_Plots'):
    #run multiple test correction
    reject, corrected, __, __ = statsmodels.stats.multitest.multipletests(ls_pval,method='fdr_bh')# #'fdr_bh'
    d_correct = dict(zip(d_data.keys(),corrected))
    d_orig = dict(zip(d_data.keys(),ls_pval))
    d_result = {}
    for s_col_center, p_correct in d_correct.items():
        pvalue = d_orig[s_col_center]
        if s_discovery.find('Discovery') > -1:
            p_correct_used=None
        else:
            p_correct_used=p_correct
        if pvalue < alpha:
            df_both_surv = d_data[s_col_center]
            s_col = s_col_center.split('_')[0]
            if s_center_column is None:
                s_center = ''
                cut_p = s_col_center.split('_')[1]
            else:
                s_center = s_col_center.split('_')[1]
                cut_p = s_col_center.split('_')[2]
            #cool plotting function for all platforms
            s_title1 = f'{s_subtype} {s_censor} {s_discovery}'
            s_title2 = f'{s_col} {s_propo} {s_center}'
            fig1, fig2, pval_cph, pval_km = km_cph_all(df_both_surv,df_clin,s_title1,s_title2,s_col,alpha=alpha,s_time=s_time, s_censor=s_censor,
                   s_groups='abundance',s_cph_model='high',ls_clin=['age','tumor_size','Stage'],p_correct=p_correct_used)
            if not fig1 is None:
                fig1.savefig(f"{savedir}/KM_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}_{cut_p}.png",dpi=300)
            if not fig2 is None:
                fig2.savefig(f"{savedir}/CPH_{s_title1.replace(' ','_')}_{s_title2.replace(' ','_')}_{cut_p}.png")
            d_result.update({s_col_center:[pval_cph,pval_km]})
    return(d_orig,d_correct,d_result)

#try:
#reject2, corrected2, __, __ = statsmodels.stats.multitest.multipletests(ls_pval_cph,alpha=alpha,method='fdr_bh')
#    print(f'{s_discovery} {s_subtype}')
#    [print(f'{ls_cph_markers[idx]} {corrected2[idx]}') for idx,item in enumerate(reject2) if item]
#except:
#    print('')