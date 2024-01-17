import random
import numpy as np
import pandas as pd
import re
from collections import Counter
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from natsort import natsorted
from adjustText import adjust_text
import seaborn as sns
import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
from datetime import datetime



def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    for word in text.split(' '):
        if word in words_to_index:
            result_vector[words_to_index[word]] +=1
    return result_vector


def word_dict_maker(ndict,df):
    words_counts = {}
    for comments in df['text']:
        for word in comments.split():
            if word not in words_counts:
                words_counts[word] = 1
            words_counts[word] += 1

    DICT_SIZE = ndict
    POPULAR_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {key: rank for rank, key in enumerate(POPULAR_WORDS, 0)}
    INDEX_TO_WORDS = {index:word for word, index in WORDS_TO_INDEX.items()}
    ALL_WORDS = WORDS_TO_INDEX.keys()
    return(WORDS_TO_INDEX, DICT_SIZE)



def analysis_performance(model_save_path,
                        study_name,
                        used_model):

    with open(model_save_path + '/' + study_name + '.csv', "r") as trial_file:
        trial_metrics = pd.read_csv(trial_file)

        ## step 1: define best trial on fold averages df
        average_performance = trial_metrics.drop('Fold', axis=1)
        average_performance = average_performance.groupby(['Trial','Attribute']).mean()
        average_performance = average_performance.reset_index()
        average_performance = average_performance.set_index('Attribute')
        average_trial_micro = average_performance.loc[["micro avg"]]

        top5_f1 = average_trial_micro.nlargest(5,'F1')
        top_precision = top5_f1.nlargest(1,'Precision')
 
        best_trial_number = top_precision.loc['micro avg']['Trial'].astype(int)
        micro_Precision = top_precision.loc['micro avg']['Precision']
        print('The Precision of the best trial is: ', micro_Precision)
        micro_F1 = top_precision.loc['micro avg']['F1']
        print('The F1 score of the best trial is: ', micro_F1)
        print('The best trial is Trial ',best_trial_number) 

        ## step 2: save the best trial averages
        best_trial =average_performance.loc[average_performance['Trial'] == best_trial_number]
        best_trial['pass_fail'] = np.where(((best_trial.F1 >= 0.8) | (best_trial.Precision >= 0.8) ),'pass', 'fail')
        best_trial.rename(index={'Fatique': 'Fatigue'}, inplace=True)

        ## save the best trial performance
        best_trial.to_csv(model_save_path +'/{}_{}_best_trial_performance.csv'.format(used_model,study_name))

        ## create a 'long' dataframe of all the folds for trial improvement barplot
        average_trial_metrics = trial_metrics.set_index('Attribute')
        average_trial_metrics = average_trial_metrics.loc[["micro avg", "macro avg"]]
        average_trial_metrics = average_trial_metrics.reset_index()
        f1_precision_long = pd.melt(average_trial_metrics,
                                    id_vars=['Trial','Attribute','Fold'],
                                    value_vars=['F1','Precision'],
                                    var_name='metric', value_name='value')
        f1_precision_long['metric2'] = f1_precision_long['Attribute'] + f1_precision_long['metric']
        f1_precision_long['Trialname'] = 'Trial_' + f1_precision_long['Trial'].astype(str)
        f1_precision_long = f1_precision_long.sort_values(by=['Attribute','Trial'])

    return best_trial_number,best_trial,f1_precision_long,micro_F1,micro_Precision

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.replace('\n', ' ').lower()# lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([w for w in text.split() if not w in STOPWORDS])# delete stopwords from text
    return text


def split_vis(x_train,x_test,y_train, y_test,train_val_size,test_size,attributes):
    counts = {}
    counts["train_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
        y_train, order=1) for combination in row)
    counts["test_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
        y_test, order=1) for combination in row)    

    # view distributions
    multi_split_dist = pd.DataFrame({"train_and_val": counts["train_counts"],
                                     "test": counts["test_counts"]
                                    }).T.fillna(0)
    multi_split_dist = multi_split_dist.reindex(natsorted(multi_split_dist.columns), axis=1)
    multi_split_dist.columns = attributes
    
    for k in counts["test_counts"].keys():
        counts["test_counts"][k] = int(counts["test_counts"][k] * (train_val_size/test_size))
        
    # View size corrected distributions
    multi_split_dist_corr = pd.DataFrame({"train_and_val": counts["train_counts"],
                                          "test": counts["test_counts"]
                                         }).T.fillna(0)
    multi_split_dist_corr =multi_split_dist_corr.reindex(natsorted(multi_split_dist_corr.columns), axis=1)
    multi_split_dist_corr.columns = attributes
    
    # correlation value between test and trainval
    dist_split = np.mean(np.std(multi_split_dist_corr.to_numpy(), axis=0))
    
    return dist_split,multi_split_dist,multi_split_dist_corr


sns.set(style="ticks", font_scale=2.5)

def plot_trials(df,
                save_path_figures,
                used_model,
                study_name,
                best_trial_number,
                metric=None,
                pal=None):
    sns.set(style="ticks", font_scale=2.5)
    fig, ax = plt.subplots(figsize=(25,5))
    sns.violinplot(x="Trialname", y="value", hue="Attribute", 
                data=df.loc[df['metric']==metric],
                ax=ax,palette = pal)
    plt.legend(bbox_to_anchor=(0.4, 1.4), loc="upper center", ncol=4, borderaxespad=0)
    for index,i in enumerate(ax.get_xticklabels()):
        if i.get_text() == 'Trial_{}'.format(best_trial_number):
            colorindex = index
    ax.set_xlabel(xlabel=None,labelpad=10)
    ax.set_ylabel(ylabel='F1-score',labelpad=10)
    sns.despine(offset=10, trim=False)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.get_xticklabels()[colorindex].set_color("red")
    plt.xticks(rotation=90)
    ax.set(ylim=(0, 1.01))
    ax.set(yticks=np.arange(0,1.1,0.2))

    plt.savefig(save_path_figures+ "/{}_{}_{}_all_trials.png".format(used_model,study_name,metric),bbox_inches="tight",dpi=600)
    plt.savefig(save_path_figures + "/{}_{}_{}_all_trials.pdf".format(used_model,study_name,metric),bbox_inches="tight",dpi=600) 
    plt.show()

def scatter_plot(df,used_model,study_name,
                 fancy_good_attributes,
                 fancy_bad_attributes,
                 metrics=None, printf1=None,
                 printprec=None,trialname=None,
                 wheretosave=None,
                 val_or_test=None):
    BG_WHITE = "#FFFFFF"#"#fbf9f4"
    COLORS = [ "#7fc97f","#fdb462","#386cb0"]
    COLORS = [ "steelblue","orange"]
    GREY30 = "#4d4d4d"
    GREY50 = "#7F7F7F"
    YTICKS = [0,0.2,0.4,0.6,0.8,1.0]
    XTICKS = [0,0.2,0.4,0.6,0.8,1.0]
    HLINES = [0.8]
    VLINES = [0.8]
    fig, ax = plt.subplots(figsize= (12, 10))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    if metrics == 'Precision_F1':
        colorcolumn = "pass_fail"
        x_data = "F1" 
        y_data = "Precision"
        
        suptitletext = '{} attribute performance'.format(val_or_test)
        namepdf = "/{}_{}_{}_Precision_F1.pdf".format(used_model,val_or_test,study_name)
        namepng = "/{}_{}_{}_Precision_F1.png".format(used_model,val_or_test,study_name)
        title_y = 'Precision'
        title_x = 'F1-score'
        

    PASS_FAIL = sorted(df[colorcolumn].unique())
    for h in HLINES:
        ax.axhline(h, color=GREY50, ls=(0, (5, 5)), alpha=0.8, zorder=0,xmin=0, xmax=1/1.1)
    for v in VLINES:
        ax.axvline(v, color=GREY50, ls=(0, (5, 5)), alpha=0.8, zorder=0,ymin=0, ymax=1/1.1)
    
    for p_f, color  in zip(PASS_FAIL, COLORS):
        data = df[df[colorcolumn] == p_f]
        ax.scatter(x_data, y_data, s=40, color=color, alpha=0.9, data=data)
    
    ax.set(ylim=(-0.1, 1.1))

    TEXTS = []
    for j,i in df.iterrows():
        if j in fancy_bad_attributes or j in fancy_good_attributes: 
            x = i[x_data]
            y = i[y_data]
            text = j
            TEXTS.append(ax.text(x, y, text, color=GREY30, fontsize=22))

    adjust_text(
        TEXTS, 
#         force_points=0.2,
        force_objects=1.9,
#         force_text=1.8,
        expand_points=(1.7, 1.4), 
        arrowprops=dict(
            arrowstyle="-", 
            color=GREY50, 
            lw=1.4
        ),
        ax=fig.axes[0]
    )

    fig.suptitle(
        "{} {}".format(used_model,suptitletext), 
        x = 0.122,
        y = 0.95,
        ha="left",
        fontsize=26,
        weight="bold",    
    )
    ax.set_title(
        (str(trialname)+" F1 score: "+ str(printf1) + ' , Precision: '+ str(printprec)), #!
        loc="left",
        ha="left",
        fontsize=24,
        weight="bold",
        pad=10
    )


    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
#     ax.spines.left.set_bounds((0, 1))
#     ax.spines.bottom.set_bounds((0,1))
    ax.set(xlim=(-0.1, 1.1))
    ax.set_yticks(YTICKS)
    ax.set_xticks(XTICKS)

    ax.set_ylabel(title_y,labelpad=10)
    ax.set_xlabel(title_x,labelpad=10)
    print(wheretosave,namepng)
    label_column = ['High performance','Low performance']
    fig.savefig(wheretosave + namepdf,bbox_inches="tight",dpi=600)
    fig.savefig(wheretosave + namepng,bbox_inches="tight",dpi=600)
    fig.show()





def permutation_of_individual_test(selected_diagnosis,
                                   flattened,
                                   mean_df,
                                   nr_donors_with_d,
                                   perms,
                                   flattened_t1,
                                   grouper,
                                   attribute_number):#,donor_diagnosis_list):
    

    # flattened = dataframe, occurence of attribute per donor, also contains main diganosis
    # general_mean_multi = mean occurence per attribute per diagnosis
    # nr_donors_with_d = number, how many donors to randomly pick.
    # flattened_t1 = table 1 occurence
    
    ##Function that does permutation testing of individual attribute in a single diagnosis and returns a p-value
    ##Set Nr of permutations with p. p=10 is for testing, while p=100000 should be used for definitive number. 
    p=perms
    attribute_name = flattened.columns[attribute_number]
    nr_donors_with_d = int(nr_donors_with_d)
    pR = []
    
    ## remove donors with disease x from background dataframe
    donorIDS_with_disease = flattened[flattened[grouper] == selected_diagnosis ]
    donorIDS_with_disease = list(donorIDS_with_disease.DonorID)
    table1_without_dd = flattened_t1[~flattened_t1.DonorID.isin(donorIDS_with_disease)]
    table1_without_dd = table1_without_dd[['DonorID',grouper,attribute_name]]
    
    ## loop over p
    for i in range(0,p):
        # pick n random rows --> df
        occurence_without_d_random = table1_without_dd.sample(n=nr_donors_with_d,replace=True)
        # calculate the mean (or median) occurence of that attribute
        attribute_average = occurence_without_d_random[attribute_name].mean()
        pR.append(attribute_average)
        
    ## compare with mean (or median) of our actual attribute
    compare_to = mean_df.loc[selected_diagnosis][attribute_name]
    p_val = len(np.where(pR>=compare_to)[0])/p
    message=False
    if message:
        ##Message 
        message = 'running permutation test for {selected_diagnosis}, which affects {nrofdonors} donors.\
                   \n attribute-nr: {attribute_number},attribute-name: {attribute_name}, which has mean/median value {mvalue}.\
                   \n nr of permutations {nrofpermutations}. \
                   \n There are {lens} permutations that exceed the mean/median, leading to a pvalue of {pval} \n'\
                   .format(attribute_number=attribute_number-2,
                           attribute_name= attribute_name,
                           mvalue=compare_to,nrofpermutations=p,
                           lens=len(np.where(pR>=compare_to)[0]),
                           pval=p_val,
                           selected_diagnosis= selected_diagnosis,
                           nrofdonors=nr_donors_with_d)
        print(message)

    return p_val


def table_selector(table_of_choice, df):
    
#     #checked
#     table1_dict = {
#                     'CON': 'CON',
#                     'TRANS,CON':'CON',
#                     'CON,ED':'CON',
#                     'SSA,CON':'CON',
#                     'AD': 'AD',
#                     'PD': 'PD',
#                     'PD,DEM':'PDD',
#                     'LBV,DEM':'DLB',
#                     'VD' : 'VD',

#                     'FTD,FTD-TDP':'FTD','FTD,FTD-TDP-A,PROG':'FTD','FTD,FTD-TDP-B,C9ORF72':'FTD','FTD,FTD-TDP-C':'FTD', #change 21 nov 2022
#                     'FTD,FTD-TAU,TAU':'FTD',
#                     'FTD,FTD-FUS':'FTD',
#                     'FTD,FTD-UPS':'FTD',               
#                     'FTD,PID':'FTD','PID,PIDC1,FTD':'FTD','PID,PIDC2,FTD':'FTD','PIA':'FTD', #change 21 nov 2022
#                     'FTD':'FTD', 

#                     'ALS,MND':'MND',
#                     'MND':'MND',

#                     'PSP' : 'PSP',

#                     'SCA,ATAXIA':'ATAXIA',
#                     'FRAGX,ATAXIA':'ATAXIA',
#                     'ADCA,ATAXIA':'ATAXIA',
#                     'FA,ATAXIA':'ATAXIA',

#                     'MS,MS_PP':'MS',
#                     'MS,MS_SP':'MS',
#                     'MS,MS-UN':'MS',
#                     'MS,MS_RR':'MS',
#                     'MS':'MS',

#                     'MSA' : 'MSA',
#                     'DEPRI,PSYCH':'MDD',
#                     'PSYCH,DEPMA':'BP',
#                     'SCHIZ,PSYCH':'SCZ'
#                                     }

    #checked
    table1_dict_paper = {
                    'CON': 'CON',
                    'CON,TRANS':'CON',
                    'CON,ED':'CON',
                    'AD': 'AD',
                    'PD': 'PD',
                    'PDD':'PDD',
                    'DLB':'DLB',
                    'VD' : 'VD',

                    # 'FTD,FTD-TDP':'FTD',
                    'FTD,FTD-TDP-A,PROG':'FTD',
                    'FTD,FTD-TDP-B,C9ORF72':'FTD',
                    'FTD,FTD-TDP-C':'FTD', 
                    'FTD,FTD-TAU,TAU':'FTD',
                    'FTD,FTD-TDP_undefined':'FTD',
                    'FTD,FTD-FUS':'FTD',
                    'FTD,FTD-UPS':'FTD',               
                    'FTD,PID':'FTD',
                    'FTD_undefined':'FTD', 

                    'MND,ALS':'MND',
                    'MND_other':'MND',

                    'PSP' : 'PSP',

                    'ATAXIA,SCA':'ATAXIA',
                    # 'ATAXIA,FRAGX':'ATAXIA',
                    'ATAXIA,ADCA':'ATAXIA',
                    'ATAXIA,FA':'ATAXIA',
                    'ATAXIA,FXTAS': 'ATAXIA',

                    'MS,MS-PP':'MS',
                    'MS,MS-SP':'MS',
                    # 'MS,MS-UN':'MS',
                    'MS,MS-RR':'MS',
                    'MS_undefined':'MS',

                    'MSA' : 'MSA',
                    'PSYCH,MDD':'MDD',
                    'PSYCH,BP':'BP',
                    'PSYCH,SCZ':'SCZ'
                                    }
    
    table1_dict_ms = {
                    'CON': 'CON',
                    'CON,TRANS':'CON',
                    'CON,ED':'CON',
                    'AD': 'AD',
                    'PD': 'PD',
                    'PDD':'PDD',
                    'DLB':'DLB',
                    'VD' : 'VD',

                    'FTD,FTD-TDP':'FTD','FTD,FTD-TDP-A,PROG':'FTD','FTD,FTD-TDP-B,C9ORF72':'FTD','FTD,FTD-TDP-C':'FTD', 
                    'FTD,FTD-TAU,TAU':'FTD',
                    'FTD,FTD-FUS':'FTD',
                    'FTD,FTD-UPS':'FTD',               
                    'FTD,PID':'FTD',
                    'FTD':'FTD', 

                    'MND,ALS':'MND',
                    'MND':'MND',

                    'PSP' : 'PSP',

                    'ATAXIA,SCA':'ATAXIA',
                    'ATAXIA,FRAGX':'ATAXIA',
                    'ATAXIA,ADCA':'ATAXIA',
                    'ATAXIA,FA':'ATAXIA',

                    'MSA' : 'MSA',
                    'PSYCH,MDD':'MDD',
                    'PSYCH,BP':'BP',
                    'PSYCH,SCZ':'SCZ'
                                    }


    table2_dict = {#'ADHD,PSYCH': 'ADHD',
                   'ASD,PSYCH': 'ASD',
                   'PSYCH,DEPMA':'BP',              
                   'DEPRI,PSYCH': 'MDD',
                   #'':'EADIS',
                  # '':'GLT',
                   'OCD,PSYCH':'OCD',
                   'PTSD,PSYCH':'PTSD',
                   'SCHIZ,PSYCH': 'SCZ',
                   'CON': 'CON',
                   'TRANS,CON':'CON',
                   'CON,ED':'CON',
                   'SSA,CON':'CON',
                   #'WH':'WH',
                   }
    table2_dict_paper = {
                   'PSYCH,ASD': 'ASD',
                   'PSYCH,BP':'BP',              
                   'PSYCH,MDD': 'MDD',
                   'PSYCH,OCD':'OCD',
                   'PSYCH,PTSD':'PTSD',
                   'PSYCH,SCZ': 'SCZ',
                   'CON': 'CON',
                   'CON,TRANS':'CON',
                   'CON,ED':'CON',
                   }


#     table3_dict = {'PSP':'PSP',
#                    'CBD':'CBD',
#                    'PID,PIDC2,FTD':'PID',
#                    'PID,PIDC1,FTD':'PID',
#                    'FTD,PID':'PID',
#                    'PIA':'PID',
#                    'ALS,MND': 'ALS',
#                    'FTD,FTD-TDP,MND':'FTD_MND',
#                    'FTD,FTD-FUS': 'FTD-FUS',
#                    'FTD,FTD-TDP': 'FTD-TDP',
#                    'FTD,FTD-TAU,TAU':'FTD-TAU',
#                    'FTD,FTD-TDP-A,PROG':'FTD-TDP-A',
#                    'FTD,FTD-TDP-B,C9ORF72': 'FTD-TDP-B',
#                    'FTD,FTD-TDP-C':'FTD-TDP-C'
#                    }

    table3_dict_paper = {
                    'PSP':'PSP',
                    'CBD':'CBD',
                    'FTD,PID':'PID',
                    'MND,ALS': 'ALS',
                    'FTD,FTD-TDP,MND':'FTD_MND',
                    'FTD,FTD-FUS': 'FTD-FUS',
                    'FTD,FTD-TDP_undefined': 'FTD-TDP',
                    'FTD,FTD-TAU,TAU':'FTD-TAU',
                    'FTD,FTD-TDP-A,PROG':'FTD-TDP-A',
                    'FTD,FTD-TDP-B,C9ORF72': 'FTD-TDP-B',
                    'FTD,FTD-TDP-C':'FTD-TDP-C',
                    'FTD_undefined':'FTD'
                       }
#     table3_con_dict = {'PSP':'PSP',
#                        'CBD':'CBD',
#                        'PID,PIDC2,FTD':'PID',
#                        'PID,PIDC1,FTD':'PID',
#                        'FTD,PID':'PID',
#                        'PIA':'PID',
#                        'ALS,MND': 'ALS',
#                        'FTD,FTD-TDP,MND':'FTD_MND',
#                        'FTD,FTD-FUS': 'FTD-FUS',
#                        'FTD,FTD-TDP': 'FTD-TDP',
#                        'FTD,FTD-TAU,TAU':'FTD-TAU',
#                        'FTD,FTD-TDP-A,PROG':'FTD-TDP-A',
#                        'FTD,FTD-TDP-B,C9ORF72': 'FTD-TDP-B',
#                        'FTD,FTD-TDP-C':'FTD-TDP-C',
#                        'CON': 'CON',
#                        'TRANS,CON':'CON',
#                        'CON,ED':'CON',
#                        'SSA,CON':'CON',
#                                            }
    
    table3_con_dict_paper = {
                                'PSP':'PSP',
                                'CBD':'CBD',
                                'FTD,PID':'PID',
                                'MND,ALS': 'ALS',
                                'FTD,FTD-TDP,MND':'FTD_MND',
                                'FTD,FTD-FUS': 'FTD-FUS',
                                'FTD,FTD-TDP_undefined': 'FTD-TDP',
                                'FTD,FTD-TAU,TAU':'FTD-TAU',
                                'FTD,FTD-TDP-A,PROG':'FTD-TDP-A',
                                'FTD,FTD-TDP-B,C9ORF72': 'FTD-TDP-B',
                                'FTD,FTD-TDP-C':'FTD-TDP-C',
                                'CON': 'CON',
                                # 'CON,TRANS':'CON',
                                # 'CON,ED':'CON',
                                       }


    table4_dict = {'MS,MS_PP': 'MS_PP',
                   'MS,MS_SP': 'MS_SP',
                   'MS,MS_RR':'MS_RR',
                   'MS,MS-UN': 'MS_UN',
                   'MS,AD':'MS,AD',
                   'MS':'MS_unclassified',
                    'CON': 'CON',
                   'TRANS,CON':'CON',
                   'CON,ED':'CON',
                   'SSA,CON':'CON',
                   }

    #checked
#     table5_dict = {'VD' : 'VD',
                   
#                    'FTD,FTD-TDP':'FTD','FTD,FTD-TDP-A,PROG':'FTD','FTD,FTD-TDP-B,C9ORF72':'FTD',
#                    'FTD,FTD-TDP-C':'FTD',
#                    'FTD,FTD-TAU,TAU':'FTD',
#                    'FTD,FTD-FUS':'FTD',
#                    'FTD,FTD-UPS':'FTD',               
#                    'FTD,PID':'FTD','PID,PIDC1,FTD':'FTD','PID,PIDC2,FTD':'FTD','PIA':'FTD', 
#                    'FTD':'FTD', 
                   
#                    'PD': 'PD',
#                    'PD,DEM':'PDD',   
#                    'LBV,DEM':'DLB',
#                    'LB': 'ILBD',
#                    'PD,AD':'PD_AD',
#                    'AD,LB,LBV':'AD_DLB',
#                    'AD': 'AD',
#                    'AD,CA':'AD_CA',
#                    'AD,VE,ENCEPHA':'AD_VE',
#                    'VE,ENCEPHA': 'VE',
#                    'DEM,VE,ENCEPHA':'DEM_VE',
#                    'DEM,SICC':'DEM_SICC',
#                    'DEM,SICC,ARG': 'DEM_SICC_AGD',
                   
#                    'DEM,SICC,LB,LBV': 'DLB_SICC',
#                    'AD,LB':'AD_ILBD'
#                    }
    
    table5_dict_paper = {
                        'VD' : 'VD',

                        'FTD,FTD-TDP_undefined':'FTD',
                        'FTD,FTD-TDP-A,PROG':'FTD',
                        'FTD,FTD-TDP-B,C9ORF72':'FTD',
                        'FTD,FTD-TDP-C':'FTD',
                        'FTD,FTD-TAU,TAU':'FTD',
                        'FTD,FTD-FUS':'FTD',
                        'FTD,FTD-UPS':'FTD',               
                        'FTD,PID':'FTD',
                        'FTD_undefined':'FTD', 

                        'PD': 'PD',
                        'PDD':'PDD',   
                        'DLB':'DLB',
                        'ILBD': 'ILBD',
                        'PD,AD':'PD_AD',
                        'AD,DLB':'AD_DLB',
                        'AD': 'AD',
                        'AD,CA':'AD_CA',
                        'AD,ENCEPHA,VE':'AD_VE',
                        'ENCEPHA,VE': 'VE',
                        'DEM,ENCEPHA,VE':'DEM_VE',
                        'DEM,SICC':'DEM_SICC',
                        'DEM,SICC,AGD': 'DEM_SICC_AGD',

                        'DLB,SICC': 'DLB_SICC',
                        'AD,ILBD':'AD_ILBD'
                   }

    table6_dict = {'NARCO,PSYCH':'NARCO',
                   'CJD':'CJD',
                   'WER,KOR':'KOR',
                   'EPI':'EPI',
                   'HD':'HD',
                   'HD,HD44':'HD',
                   'HD,HD46':'HD',
                   'CBD':'CBD',
                   'SCA,ATAXIA':'SCA',
                   'FA,ATAXIA':'FA',
                   'FRAGX,ATAXIA':'FRAGX_ATAXIA',
                   'FRAGX':'FRAGX',
                   'PML,ENCEPHA':'PML',
                   'PWS':'PWS',
                   'LDA':'LDA',
                   'LD':'LD',
                   'DOWN':'DOWN',
                   'DEV':'DEV',
                   'CMT':'CMT'
                   }



    table7_dict = {'TUM': 'TUM',
                   'CVA': 'CVA',
                   'DEM,CVA': 'CVA',
                   'TAU':'TAU',
                   'IS': 'IS',
                   'HIP':'HIP',
                   }
    
    
    
#     table_all_dict = {'AD': 'AD',
#                        'PD': 'PD',
#                        'PD,DEM':'PDD',

#                        'CON': 'CONTROL',
#                        'TRANS,CON':'CONTROL',
#                        'CON,ED':'CONTROL',
#                        'SSA,CON':'CONTROL',

#                        'LBV,DEM':'DLB',
#                        'DEM,SICC,LB,LBV': 'DLB_SICC',
                       

#                        'VD' : 'VD',

#                        'FTD,FTD-TDP':'FTD',
#                        'FTD,FTD-TDP-A,PROG':'FTD',
#                        'FTD,FTD-TAU,TAU':'FTD',
#                        'FTD,FTD-FUS':'FTD',
#                        'FTD,FTD-TDP-B,C9ORF72':'FTD',
#                        'FTD,FTD-UPS':'FTD',
#                        'FTD,FTD-TDP-C':'FTD',
#                        'FTD,PID':'FTD',
#                        'PID,PIDC1,FTD':'FTD',
#                        'FTD':'FTD',
#                        'PID,PIDC2,FTD':'FTD',
#                        'PIA':'FTD',
                      
#                        'ALS,MND':'MND',
#                        'MND':'MND',

#                        'PSP' : 'PSP',

#                        'SCA,ATAXIA':'ATAXIA',
#                        'FRAGX,ATAXIA':'ATAXIA',
#                        'ADCA,ATAXIA':'ATAXIA',
#                        'FA,ATAXIA':'ATAXIA',

#                        'MS,MS_PP':'MS',
#                        'MS,MS_SP':'MS',
#                        'MS,MS-UN':'MS',
#                        'MS,MS_RR':'MS',
#                        'MS':'MS',

#                        'MSA' : 'MSA',
#                        'DEPRI,PSYCH':'MDD',
#                        'PSYCH,DEPMA':'BP',
#                        'SCHIZ,PSYCH':'SCZ',
#                        'ADHD,PSYCH': 'Other_PSYCH',
#                        'ASD,PSYCH': 'Other_PSYCH',
#                        'OCD,PSYCH':'Other_PSYCH',
#                        'PTSD,PSYCH':'Other_PSYCH',
#                       'NARCO,PSYCH':'Other_PSYCH',
#                       'PSYCH':'Other_PSYCH',
                      
#                        'DEM,SICC,ARG': 'DEM_SICC_AGD',
#                        'VE,ENCEPHA': 'VE',
#                        'DEM,VE,ENCEPHA':'DEM_VE',
# #                        'NAD,DEM': 'NAD',
#                        'DEM,SICC':'DEM_SICC',
                      
#                        'AD,CA':'AD_CA',
#                        'AD,VE,ENCEPHA':'AD_VE',
#                        'PD,AD':'PD_AD',
#                        'AD,LB,LBV':'AD_DLB',
#                        'AD,LB':'AD_ILBD',
#                        'LB': 'ILBD',
                      
                      
#                        'ALEX':'Other',
#                       'AR615':'Other',
#                         'BINSW':'Other',
#                        'CA':'Other',
#                         'CBD':'Other',
#                        'CJD':'Other',
#                        'CMT':'Other',
#                        'COHA':'Other',
#                       'CVA':'Other',
#                        'DAI':'Other',
#                        'DEM,CVA':'Other',
#                        'DEM,SICC,CA':'Other',
#                       'DEM,SICC,LB':'Other', # change nov 22
#                        'DEV':'Other',
#                       'DOWN':'Other',
#                       'DYSTO':'Other',
#                       'ENCE':'Other',
#                       'EPI':'Other',
#                       'FAHR':'Other',
#                       'FRAGX':'Other',
#                       'FTD,FTD-TDP,MND':'Other',
# #                        'FTD,FTD-TAU,TAU,PSP':'Other',
#                        'GUIL':'Other',
#                        'HD':'Other',
#                        'HD,HD44':'Other',
#                        'HD,HD46':'Other',
#                         'HIP':'Other',
#                          'HIV':'Other',
#                         'HSP':'Other',
#                       'HMSN':'Other',
#                       'IS':'Other',
#                        'KLIN':'Other',
#                        'LDA':'Other',
#                        'LD':'Other',
#                        'MEDIS':'Other',
#                        'MEN':'Other',
#                       'MS,AD':'Other',
#                       'NCSD':'Other',
#                        'NHL':'Other',
#                       'NIG':'Other',
#                        'NMO,MS':'Other',
#                        'NU':'Undefined',
#                         'OTHER':'Undefined',
#                        'PAL':'Other',
#                          'PCAD':'Other',
#                       'PD,ATPD':'Other',
#                         'PML,ENCEPHA':'Other',
#                        'POLY':'Other',
#                        'PWS':'Other',
#                        'SICC,LB':'Other',
#                          'SCHIZ,PSYCH,AD':'Other',
#                       'SEP':'Other',
#                       'TAU':'Other',
#                        'TUM':'Other',
#                       'Unknown':'Undefined',
#                        'VDCAD':'Other',
#                            'VD,LB':'Other',
#                        'WER,KOR':'Other',
#                          'WH':'Other',}
    
    table_all_dict_paper = {
                        'AD': 'AD',
                        'PD': 'PD',
                        'PDD':'PDD', ##

                        'CON': 'CONTROL',
                        # 'CON,TRANS':'CONTROL',
                        # 'CON,ED':'CONTROL',

                        'DLB':'DLB',
                        'DLB,SICC': 'DLB_SICC',


                        'VD' : 'VD',

                        'FTD,FTD-TDP':'FTD',
                        'FTD,FTD-TDP-A,PROG':'FTD',
                        'FTD,FTD-TAU,TAU':'FTD',
                        'FTD,FTD-FUS':'FTD',
                        'FTD,FTD-TDP-B,C9ORF72':'FTD',
                        # 'FTD,FTD-UPS':'FTD',
                        'FTD,FTD-TDP-C':'FTD',
                        'FTD,PID':'FTD',
                        'FTD':'FTD',
                      
                        'MND,ALS':'MND',
                        'MND':'MND',

                        'PSP' : 'PSP',

                        'ATAXIA,SCA':'ATAXIA',##
                        'ATAXIA,FRAGX':'ATAXIA',##
                        'ATAXIA,ADCA':'ATAXIA',##
                        'ATAXIA,FA':'ATAXIA',##

                        'MS,MS-PP':'MS',
                        'MS,MS-SP':'MS',
                        # 'MS,MS-UN':'MS',
                        'MS,MS-RR':'MS',
                        'MS':'MS',

                        'MSA' : 'MSA',
                        'PSYCH,MDD':'MDD',
                        'PSYCH,BP':'BP',
                        'PSYCH,SCZ':'SCZ',
                        # 'PSYCH,ADHD': 'Other_PSYCH',
                        'PSYCH,ASD': 'Other_PSYCH',
                        'PSYCH,OCD':'Other_PSYCH',
                        'PSYCH,PTSD':'Other_PSYCH',
                        # 'PSYCH,NARCO':'Other_PSYCH',
                        # 'PSYCH':'Other_PSYCH',

                        'DEM,SICC,AGD': 'DEM_SICC_AGD',
                        'ENCEPHA,VE': 'VE',
                        'DEM,ENCEPHA,VE':'DEM_VE',
                        #                        'NAD,DEM': 'NAD',
                        'DEM,SICC':'DEM_SICC',

                        'AD,CA':'AD_CA',
                        'AD,ENCEPHA,VE':'AD_VE', ##
                        'PD,AD':'PD_AD',
                        'AD,DLB':'AD_DLB',##
                        # 'AD,ILBD':'AD_ILBD',##
                        # 'ILBD': 'ILBD',
        
                        'FTD,FTD-UPS':'Other',     ## 26 july 2023    
                        'PSYCH,NARCO':'Other',     ## 26 july 2023 
                        'PSYCH,ADHD': 'Other',     ## 26 july 2023 
                        'AD,ILBD':'Other',##        ## 26 july 2023 
                        'ILBD': 'Other',             ## 26 july 2023 
                        'ALEX':'Other',
                        'AR615':'Other',
                        'BINSW':'Other',
                        'CA':'Other',
                        'CBD':'Other',
                        'CJD':'Other',
                        'CMT':'Other',
                        'COHA':'Other',
                        'CVA':'Other',
                        'DAI':'Other',
                        'DEM,CVA':'Other',
                        'DEM,SICC,CA':'Other',
                        'DEM,SICC,ILBD':'Other', 
                        'DEV':'Other',
                        'DOWN':'Other',
                        'DYSTO':'Other',
                        'ENCE':'Other',
                        'ENCEPHA,PML':'Other',
                        'EPI':'Other',
                        'FAHR':'Other',
                        'FRAGX':'Other',
                        'FTD,FTD-TDP,MND':'Other',
                        'GUIL':'Other',
                        'HD':'Other',
                        'HIP':'Other',
                        'HIV':'Other',
                        'HSP':'Other',
                        'HMSN':'Other',
                        'IS':'Other',
                        'KLIN':'Other',
                        'LDA':'Other',
                        'LD':'Other',
                        'MEDIS':'Other',
                        'MEN':'Other',
                        'MS,AD':'Other',
                        'NCSD':'Other',
                        'NHL':'Other',
                        'NIG':'Other',
                        'MS,NMO':'Other',
                        'PAL':'Other',
                        'PCAD':'Other',
                        'PD,ATPD':'Other',

                        'POLY':'Other',
                        'PWS':'Other',
                        'SICC,ILBD':'Other',
                        'PSYCH,SCZ,AD':'Other',
                        'SEP':'Other',
                        'TAU':'Other',
                        'TUM':'Other',

                        'TRANS,AD':'Other',
                        'TRANS,PSYCH':'Other',
                        'TRANS,PSYCH,MDD':'Other',
                        'TRANS,PSYCH,MDD,ILBD':'Other',
                        'TRANS,TUM':'Other',

                        'UNDEFINED':'Undefined',
                        'VDCAD':'Other',
                        'VD,ILBD':'Other',
                        'WER,KOR':'Other',
                        'PSYCH,WH':'Other',}
    
    

    table_all_diagnoses = ['CONTROL','AD','PD','PDD','DLB','VD','DEM_VE','DEM_SICC',  'AD_VE', 
                           'PD_AD', 'AD_CA', 'VE', 'AD_DLB','AD_ILBD', 'DLB_SICC', 'DEM_SICC_AGD', 'ILBD', 
                           'FTD','MND','PSP','ATAXIA','MS','MSA','Other','MDD','BP','SCZ', 'Other_PSYCH','Undefined']


    table1_diagnoses = ['CON','AD','PD','PDD','DLB','VD','FTD','MND','PSP','ATAXIA','MS','MSA','MDD','BP','SCZ'] 
    table1_ms = ['CON','AD','PD','PDD','DLB','VD','FTD','MND','PSP','ATAXIA','MSA','MDD','BP','SCZ'] 
    table2_diagnoses = ['CON','SCZ','PTSD','OCD','MDD','BP','ASD']

    table3_diagnoses = ['PSP','CBD','PID','ALS','FTD_MND','FTD-FUS','FTD-TDP',
                        'FTD-TAU','FTD-TDP-A','FTD-TDP-B','FTD-TDP-C'] 
    table3_con_diagnoses = ['CON','PID','PSP','CBD','ALS','FTD_MND','FTD-FUS','FTD-TAU',
                            'FTD-TDP','FTD-TDP-A','FTD-TDP-B','FTD-TDP-C'] 
    table4_diagnoses = ['MS_unclassified','MS,AD','MS_UN','MS_RR','MS_SP','MS_PP','CON']
    table5_diagnoses = ['VD','FTD','PD','PDD','DLB','ILBD','PD_AD','AD_DLB','AD',
                        'AD_CA','AD_VE','VE','DEM_VE','DEM_SICC','DEM_SICC_AGD','DLB_SICC','AD_ILBD'] 
    
    
    table6_diagnoses = ['NARCO','CJD','KOR','EPI','HD','CBD','SCA','FA',
                        'FRAGX_ATAXIA','FRAGX','PML','PWS','LDA','LD','DOWN','DEV','CMT']
    table7_diagnoses = ['TUM','CVA','TAU','IS','HIP']

#     if table_of_choice == 'table1':
#         dict_table=table1_dict
#         ordered_diagnoses = table1_diagnoses
    if table_of_choice == 'table1_p':
        dict_table=table1_dict_paper
        ordered_diagnoses = table1_diagnoses
    elif table_of_choice == 'table1_ms':
        dict_table=table1_dict_ms
        ordered_diagnoses = table1_ms
    elif table_of_choice == 'table2':
        dict_table=table2_dict
        ordered_diagnoses = table2_diagnoses
    elif table_of_choice == 'table2_p':
        dict_table=table2_dict_paper
        ordered_diagnoses = table2_diagnoses
#     elif table_of_choice == 'table3':
#         dict_table=table3_dict
#         ordered_diagnoses = table3_diagnoses
    elif table_of_choice == 'table3_p':
        dict_table=table3_dict_paper
        ordered_diagnoses = table3_diagnoses
#     elif table_of_choice == 'table3_with_con':
#         dict_table=table3_con_dict
#         ordered_diagnoses = table3_con_diagnoses
    elif table_of_choice == 'table3_with_con_p':
        dict_table = table3_con_dict_paper
        ordered_diagnoses = table3_con_diagnoses
        
       
    elif table_of_choice == 'table4':
        dict_table=table4_dict
        ordered_diagnoses = table4_diagnoses
#     elif table_of_choice == 'table5':
#         dict_table=table5_dict
#         ordered_diagnoses = table5_diagnoses
    elif table_of_choice == 'table5_p':
        dict_table=table5_dict_paper
        ordered_diagnoses = table5_diagnoses
    elif table_of_choice == 'table6':
        dict_table=table6_dict
        ordered_diagnoses = table6_diagnoses
    elif table_of_choice == 'table7':
        dict_table=table7_dict
        ordered_diagnoses = table7_diagnoses
#     elif table_of_choice == 'tableall':
#         dict_table=table_all_dict
#         ordered_diagnoses = table_all_diagnoses
    elif table_of_choice == 'tableall_p':
        dict_table=table_all_dict_paper
        ordered_diagnoses = table_all_diagnoses
        
        
        
    
    keysList = list(dict_table.keys())
    df_out = df.loc[df['neuropathological_diagnosis'].isin(keysList)]
    df_out = df_out.replace({"neuropathological_diagnosis": dict_table})
    
    return df_out, ordered_diagnoses










