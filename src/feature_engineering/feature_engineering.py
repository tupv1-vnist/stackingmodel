import pandas as pd
import numpy as np

def lag_feature(df, lags, col):
    tmp = df[['date_block_num','campaign_id','product_id',col]]
    
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','campaign_id','product_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','campaign_id','product_id'], how='left')
    
    return df

def add_group_stats(matrix_, groupby_feats, target, enc_feat, last_periods):
    if not 'date_block_num' in groupby_feats:
        print ('date_block_num must in groupby_feats')
        return matrix_
    
    group = matrix_.groupby(groupby_feats)[target].sum().reset_index()
    max_lags = np.max(last_periods)
    
    for i in range(1,max_lags+1):
        shifted = group[groupby_feats+[target]].copy(deep=True)
        shifted['date_block_num'] += i
        shifted.rename({target:target+'_lag_'+str(i)},axis=1,inplace=True)
        group = group.merge(shifted, on=groupby_feats, how='left')
    
    group.fillna(0,inplace=True)
    
    for period in last_periods:
        lag_feats = [target+'_lag_'+str(lag) for lag in np.arange(1,period+1)]
        mean = group[lag_feats].sum(axis=1)/float(period)
        mean2 = (group[lag_feats]**2).sum(axis=1)/float(period)
        group[enc_feat+'_avg_sale_last_'+str(period)] = mean
        group[enc_feat+'_std_sale_last_'+str(period)] = (mean2 - mean**2).apply(np.sqrt)
        group[enc_feat+'_std_sale_last_'+str(period)].replace(np.inf,0,inplace=True)
        group[enc_feat+'_avg_sale_last_'+str(period)] /= group[enc_feat+'_avg_sale_last_'+str(period)].mean()
        group[enc_feat+'_std_sale_last_'+str(period)] /= group[enc_feat+'_std_sale_last_'+str(period)].mean()
    
    cols = groupby_feats + [f_ for f_ in group.columns.values if f_.find('_sale_last_')>=0]
    matrix = matrix_.merge(group[cols], on=groupby_feats, how='left')
    
    return matrix

def target_encoding(matrix_, groupby_feats, target, enc_feat, lags):
    print ('target encoding for',groupby_feats)
    group = matrix_.groupby(groupby_feats).agg({target:'mean'})
    group.columns = [enc_feat]
    group.reset_index(inplace=True)
    matrix = matrix_.merge(group, on=groupby_feats, how='left')
    matrix[enc_feat] = matrix[enc_feat].astype(np.float16)
    matrix = lag_feature(matrix, lags, enc_feat)
    matrix.drop(enc_feat, axis=1, inplace=True)
    return matrix
