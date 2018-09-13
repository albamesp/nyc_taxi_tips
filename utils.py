import os
import datetime as dt
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from scipy.stats import skew
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error



import warnings
warnings.filterwarnings('ignore')

def reg_error(mydata, real, pred, bins=20):
    myerrordata = mydata[pred] - mydata[real]
    # Error distribution
    plt.hist(myerrordata, bins=bins)
    # Error
    print("AVG_SC: {}".format(mydata[real].mean()))
    print("MAE   : {}".format(mean_absolute_error(mydata[real], mydata[pred])))
    print("MAEN  : {}".format(mean_absolute_error(mydata[real], mydata[pred]) / (0.0 + mydata[real].mean())))    

def plot_continuous(df,label,method={'type':'histogram','bins':20},outlier='on'):
    """
    function to quickly visualize continous variables
    df: pandas.dataFrame 
    label: str, name of the variable to be plotted. It should be present in df.columns
    method: dict, contains info of the type of plot to generate. It can be histogram or boxplot [-Not yet developped]
    outlier: {'on','off'}, Set it to off if you need to cut off outliers. Outliers are all those points
    located at 3 standard deviations further from the mean
    """
    # create vector of the variable of interest
    v = df[label]
    # define mean and standard deviation
    m = v.mean()
    s = v.std()
    # prep the figure
    fig,ax = plt.subplots(1,2,figsize=(14,4))
    ax[0].set_title('Distribution of '+label)
    ax[1].set_title('Tip % by '+label)
    if outlier=='off': # remove outliers accordingly and update titles
        v = v[(v-m)<=3*s]
        ax[0].set_title('Distribution of '+label+'(no outliers)')
        ax[1].set_title('Tip % by '+label+'(no outliers)')
    if method['type'] == 'histogram': # plot the histogram
        v.hist(bins = method['bins'],ax=ax[0])
    if method['type'] == 'boxplot': # plot the box plot
        df.loc[v.index].boxplot(label,ax=ax[0])
    ax[1].plot(v,df.loc[v.index].Tip_percentage,'.',alpha=0.4)
    ax[0].set_xlabel(label)
    ax[1].set_xlabel(label)
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Tip') 
                     
    
def plot_categories(df,catName,chart_type='boxplot',ylimit=[None,None]):
    """
    This functions helps to quickly visualize categorical variables. 
    This functions calls other functions generate_boxplot and generate_histogram
    df: pandas.Dataframe
    catName: str, variable name, it must be present in df
    chart_type: {histogram,boxplot}, choose which type of chart to plot
    ylim: tuple, list. Valid if chart_type is histogram
    """
    print( catName)
    cats = sorted(pd.unique(df[catName]))
    if chart_type == 'boxplot': #generate boxplot
        generate_boxplot(df,catName,ylimit)
    elif chart_type == 'histogram': # generate histogram
        generate_histogram(df,catName)
    else:
        pass
    
    #=> calculate test statistics
    groups = df[[catName,'Tip_percentage']].groupby(catName).groups #create groups
    tips = df.Tip_percentage
    if len(cats)<=2: # if there are only two groups use t-test
        print(ttest_ind(tips[groups[cats[0]]],tips[groups[cats[1]]]))
    else: # otherwise, use one_way anova test
        # prepare the command to be evaluated
        cmd = "f_oneway("
        for cat in cats:
            cmd+="tips[groups["+str(cat)+"]],"
        cmd=cmd[:-1]+")"
        print("one way anova test:", eval(cmd) )#evaluate the command and print
    print("Frequency of categories (%):\n",df[catName].value_counts(normalize=True)*100)

def generate_histogram(df,catName):
    """
    generate histogram of tip percentage by variable "catName" with ylim set to ylimit
    df: pandas.Dataframe
    catName: str
    ylimit: tuple, list
    """
    cats = sorted(pd.unique(df[catName]))
    colors = plt.cm.jet(np.linspace(0,1,len(cats)))
    hx = np.array(map(lambda x:round(x,1),np.histogram(df.Tip_percentage,bins=20)[1]))
    fig,ax = plt.subplots(1,1,figsize = (15,4))
    for i,cat in enumerate(cats):
        vals = df[df[catName] == cat].Tip_percentage
        h = np.histogram(vals,bins=hx)
        w = 0.9*(hx[1]-hx[0])/float(len(cats))
        plt.bar(hx[:-1]+w*i,h[0],color=colors[i],width=w)
    plt.legend(cats)
    plt.yscale('log')
    plt.title('Distribution of Tip by '+catName)
    plt.xlabel('Tip (%)')
    
def generate_boxplot(df,catName,ylimit):
    """
    generate boxplot of tip percentage by variable "catName" with ylim set to ylimit
    df: pandas.Dataframe
    catName: str
    ylimit: tuple, list
    """
    df.boxplot('Tip_percentage',by=catName)
    #plt.title('Tip % by '+catName)
    plt.title('')
    plt.ylabel('Tip (%)')
    if ylimit != [None,None]:
        plt.ylim(ylimit)
    plt.show()

def print_full(x):
    """Print a full pandas object"""
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def delifexists(df,f):
    """Delete if exists of field from a dataframe. If not, fail silently
        df: dataframe
        f:  field
    """
    try:
        del df[f]
    except:
        pass


def cumulative_gain(y_true, y_score):
    """Generate the cumulative gain chart
        y_true:  true values
        y_score: predicted values    
    """
    pos_label = 1
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    total_p = sum(y_true)
    
    weight = 1

    # cg: Cumulative gain
    cg_pop_perc = np.linspace(0.0, 1.0, num=len(y_true))
    cg_cap_perc = (y_true * weight).cumsum() / (total_p + 0.0)
    cg_wizard_perc = np.array([min(i + 1, total_p) for i in range(0, len(y_score))]) / (total_p + 0.0)
    return cg_pop_perc, cg_cap_perc, cg_wizard_perc

def modelfit(alg,dtrain,predictors,target,scoring_method,performCV=True,printFeatureImportance=True,cv_folds=5):
    """
    This functions train the model given as 'alg' by performing cross-validation. It works on both regression and classification
    alg: sklearn model
    dtrain: pandas.DataFrame, training set
    predictors: list, labels to be used in the model training process. They should be in the column names of dtrain
    target: str, target variable
    scoring_method: str, method to be used by the cross-validation to valuate the model
    performCV: bool, perform Cv or not
    printFeatureImportance: bool, plot histogram of features importance or not
    cv_folds: int, degree of cross-validation
    """
    # train the algorithm on data
    alg.fit(dtrain[predictors],dtrain[target])
    #predict on train set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    if scoring_method == 'roc_auc':
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #perform cross-validation
    if performCV:
        cv_score = cross_validation.cross_val_score(alg,dtrain[predictors],dtrain[target],cv=cv_folds,scoring=scoring_method)
        #print model report
        print("\nModel report:")
        if scoring_method == 'roc_auc':
            print("Accuracy:", metrics.accuracy_score(dtrain[target].values,dtrain_predictions))
            print("AUC Score (Train):", metrics.roc_auc_score(dtrain[target], dtrain_predprob))
        if (scoring_method == 'mean_squared_error'):
            print("Accuracy:",metrics.mean_squared_error(dtrain[target].values,dtrain_predictions))
    if performCV:
        print("CV Score - Mean : %.7g | Std : %.7g | Min : %.7g | Max : %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    #print feature importance
    if printFeatureImportance:
        if dir(alg)[0] == '_Booster': #runs only if alg is xgboost
            feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        else:
            feat_imp = pd.Series(alg.feature_importances_,predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar',title='Feature Importances')
        plt.ylabel('Feature Importe Score')
        plt.show()
        
def lift_charts(y_true, y_score, num_buckets=10):    
    """Generates the lift charts with the given number of buckets.
        y_true:  true values
        y_score: predicted values
        num_buckets: number of buckets to perform the analysis
    """
    # results = pd.DataFrame(data={'expected': y_score, 'true': y_true}).sort(columns='expected', ascending=False)
    # (Angel) cambio esta linea de codigo porque en mi version falla, parece que la funcion sort no funciona en mi version python. Usamos sort_values en su lugar
    results = pd.DataFrame(data={'expected': y_score, 'true': y_true}).sort_values('expected', ascending=False)
    results['order'] = np.arange(len(y_score))
    results['bucket'] = results['order'] * num_buckets / (0.0 + len(y_score))
    results['bucket'] = results['bucket'].apply(lambda x: int(x) + 1)    

    #results = pd.DataFrame({'expected': y_score, 'true': y_true})
    
    avg_expected = y_score.mean()
    avg_true = y_true.mean()
    plt.figure()

    # Probability chart
    ax = plt.subplot(131)
    
    #results['bucket'] = num_buckets - pd.qcut(results.expected, num_buckets, labels=False)   
    by_decile = results.groupby('bucket')    
    max_fact = by_decile.true.mean().iloc[-1]    
    diffs = by_decile.expected.mean() - by_decile.true.mean()
    pdf = pd.DataFrame({'expected': by_decile.expected.mean(), 'true': by_decile.true.mean(), 'diff': diffs})
    plt.plot(by_decile.bucket.mean(),pdf)
    plt.legend(['diff','exp','true'])
    plt.title('Probability chart')
    
    # Lift chart
    ax = plt.subplot(132)      
        
    #results['bucket'] = num_buckets - pd.qcut(results.expected, num_buckets, labels=False)   
    by_decile = results.groupby('bucket')    
    max_fact = by_decile.true.mean().iloc[-1]    
    diffs = by_decile.expected.mean() - by_decile.true.mean()
    pdf = pd.DataFrame({'expected': by_decile.expected.mean() / avg_expected, 
                  'true': by_decile.true.mean() / avg_true })
    plt.plot(by_decile.bucket.mean(),pdf)
    plt.legend(['exp','true'])
    plt.title('Lift chart')

    
    # Cumulative lift chart normalized   
    ax = plt.subplot(133)      

    #results['bucket'] = num_buckets - pd.qcut(results.expected, num_buckets, labels=False)   
    by_decile = results.groupby('bucket')    
    max_fact = by_decile.true.mean().iloc[-1]    
    diffs = by_decile.expected.mean() - by_decile.true.mean()
    pdf = pd.DataFrame({'expected': (by_decile.expected.mean() / avg_expected).cumsum() / by_decile.bucket.mean(), 
                  'true': (by_decile.true.mean() / avg_true).cumsum() / by_decile.bucket.mean() })
    plt.plot(by_decile.bucket.mean(),pdf)
    plt.legend(['exp','true'])
    plt.title('Cumulative Lift')
    
    plt.subplots_adjust(left=.02, right=.98)
    plt.show()


   
def generate_tramified_var(df, var, breaks):
    """Generates a tramified variable according to the specified breaks.
        df:     dataframe
        var:    variable to be tramified
        breaks: breaks
    """
    g = df.groupby(var)
    tramified = pd.Series(index=df.index)
    for i, break_ in enumerate(breaks):
        ix = g['exit'].transform(lambda x: True if (x.mean() >= break_) else False).astype(bool)
        tramified[ix] = i
    return tramified.astype(int)

    
def univariate_analysis(df, variable, target):
    by_var = df.groupby(variable)
    sums = pd.DataFrame({'conv': by_var[target].mean(), 
                         'vol': 100*by_var[target].count()/len(df) })
    sums['cumvol'] = sums['vol'].cumsum()
    return sums
    

def make_univariate_plot(a, title):
    plt.figure()

    line1 = a['vol'].plot(kind='bar', color='blue', label='Vol')
    line2 = a['conv'].plot(secondary_y=True, kind='line', color='orange', marker='o', label='Conv', mark_right=False)
    line1.set_ylim(ymin=0)
    line2.set_ylim(ymin=0, ymax=max(a['conv'])* 1.1)
    line1.set_ylabel('Volume')
    line1.right_ax.set_ylabel('Conversion')
    line1.legend(loc=3)
    line2.legend(loc=4)
    plt.title('Univariate analysis{}'.format(title))    
    
    
def univariate_with_plot(table, var, target='exit', temporal=None):
    """Univariate analysis. Parameters:
        table:    dataframe to study
        var:      variable to study
        target:   target variable (default 'exit')
        temporal: variable to perform time analysis (default None)
    """
    lframes = ['',]
    if temporal is not None:
        lframes = table[temporal].unique()
        lframes.sort()
    for frame in lframes:
        if frame != '':
            tabletemp = group_low_freq(table[table[temporal] == frame], var, suffix='_' + str(frame))
        else:
            tabletemp = group_low_freq(table, var, suffix='_' + str(frame))
        a = univariate_analysis(tabletemp, var + '_' + str(frame), target=target)
        print(a)
        make_univariate_plot(a, ': ' + var + ' ' + str(frame))
        
      
     
def encode_vars(mydata, lc):
    le = LabelEncoder()

    for myc in lc:
        print(mydata[myc].unique())
        le.fit(mydata[myc].unique())
        print(le.classes_)
        mydata[myc + '_NUM'] = le.transform(mydata[myc])

    for myc in lc:
        print(mydata[myc + '_NUM'].head())        



def encode_vars_ratio(mydatadict, mydataset_orig, mydataset, catvars, target_var, train=True, min_obs=100):
    """
    mydatadict     -> Dict where lookup tables will be stored
    mydataset_orig -> Original dataset to extract ratios for missing categories
    mydataset      -> Dataset to be encoded
    catvars        -> Categorical varables to be encoded
    target_var     -> Target variable to extract ratios from
    train          -> If true, encodings will be calculated. If false, only will be read
    min_obs        -> Minimum obs to calculate the ratio. Otherwise the mean will be considered
    """
    mydataset_res = mydataset.copy()
    if train:
        mydataset_res[target_var + '_AUX'] = mydataset_res[target_var]
    for k in catvars:
        #print(k)
        if train:
            aux = mydataset_res.groupby([k]).agg({target_var:'mean', target_var + '_AUX':'count'}).reset_index()        
            # Avoid overfitting
            aux.loc[(aux[target_var + '_AUX'] < min_obs), target_var] = mydataset_orig[target_var].mean()
            aux = aux[[k, target_var]]
            mydatadict['ENC_{}'.format(k)] = aux            
            mydatadict['ENC_{}'.format(k)].columns = [mydatadict['ENC_{}'.format(k)].columns[0], '{}_RATIO'.format(k)]            
            
        mydataset_res = pd.merge(
            left = mydataset_res,
            right = mydatadict['ENC_{}'.format(k)],
            left_on = [k],
            right_on = [k],
            how = 'left',
            suffixes = ['','_AUX']
            )
        
    for k in catvars:
        # Fill na
        mydataset_res['{}_RATIO'.format(k)].fillna(mydataset_orig[target_var].mean(), inplace=True)

    return mydataset_res

  
def pickle_save(object_to_save, file_path, file_name):
    """
    Save an object using pickle module
    object_to_save: Python object to be saved
    file_path: Path to folder where the file is written
    file_name: Name of the saved file
    """
    pickle.dump(object_to_save, open(file_path + file_name + 'pkl', 'wb'))
    return None


def mysplit(mydata, prop_train=0.6, myseed=12345):
    """Split a dataset according to a proportion
 
    mydata     -> DataFrame to be splitted
    prop_train -> Proportion used to split the DataFrame
    myseed     -> Seed for random state.
    """
    np.random.seed(myseed)
    tablon_temp = mydata.copy()
    tablon_temp['rand'] = np.random.rand(len(tablon_temp))
    tablon_train = tablon_temp[tablon_temp['rand'] < prop_train]
    tablon_test = tablon_temp[tablon_temp['rand'] >= prop_train]    
    return tablon_train, tablon_test


def lift_var_plots(data_set,var, target, kpi = 'churn_rate', with_warnings = False, cortes = 5, pct_pobl_min = .05 ,min_proportion = 0.05):
    from IPython.display import display
    numeric_columns = data_set._get_numeric_data().columns.tolist()
    pobl_min = int(pct_pobl_min * len(data_set))
    if var in numeric_columns:
        data_set2 = segment_continuous_variable_df(data_set,var,target,
                                                   tree_props={'criterion':'gini','max_leaf_nodes': cortes,
                                                               'min_samples_leaf':pobl_min})
    else:
        data_set2 = data_set
        data_set2['segment'] = data_set2[var]
    agg_data_set2 = data_set2[['segment', target]].groupby(['segment']).agg(['mean', 'count']).reset_index()
    agg_data_set2['mean_target'] = agg_data_set2[target]['mean']
    agg_data_set2['density'] = agg_data_set2[target]['count']
    agg_data_set2['proportion'] = agg_data_set2['density']/agg_data_set2.density.sum()
    agg_data_set2['lift'] = agg_data_set2['mean_target'] / data_set2[target].mean()
    agg_data_set2['churn_rate'] = agg_data_set2['mean_target'] 
    agg_data_set2['index'] = range(len(agg_data_set2))
    agg_data_set2['segment'] = [str(x) for x in agg_data_set2['segment']]
    remove_rows = []
    for i in range(len(agg_data_set2)):
        if (agg_data_set2['proportion'][i] < min_proportion):
            if with_warnings:
                print('WARNING: Less than ' + str(100 * min_proportion) + '% of sample in segment '+ str(agg_data_set2['segment'][i]))
            if(var not in numeric_columns):
                remove_rows = remove_rows + [i]
    agg_data_set2.reset_index(inplace = True)
    display(agg_data_set2[['segment', 'density','lift', 'churn_rate']])
    agg_data_set2.drop(agg_data_set2.index[remove_rows], inplace = True)
    agg_data_set2[' '] = range(len(agg_data_set2))
    
    plot_title = 'Volume & '+ kpi + '\n' + 'by ' + var.lower() + '; target = ' + target.lower()
    
    fig = plt.figure()
    
    
    ax = agg_data_set2[[' ', kpi]].plot(
    x=' ', linestyle='-', marker='o', color = 'black',secondary_y = True ,legend=None)


    agg_data_set2[[' ',  'density']].plot(x=' ', kind='bar', color = 'orange',legend=None,
                                        ax=ax)
    ax.set_xticklabels(agg_data_set2['segment'])
       
    ax.set_ylim(ymin=0)
    plt.title(plot_title)   
    plt.xticks(rotation='vertical')
    plt.show()
    plt.close()

