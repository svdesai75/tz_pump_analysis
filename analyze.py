#!/usr/bin/env python
import pandas as pd
import numpy as np
from load_data import load_data
from collections import OrderedDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

status_dict=OrderedDict([('f', 'functional'), ('fnr','functional_needs_repair'), ('nf','non_functional')])
statuses=status_dict.keys()

feature_cols=["approx_latitude","approx_longitude", "population","gps_height", "installer_count",
              "qty_frac_f", "ward_frac_f", "st_frac_f",'installer_frac_f',"wq_frac_f",
              "qty_frac_fnr", "ward_frac_fnr", "st_frac_fnr",'installer_frac_fnr',"wq_frac_fnr",
              "qty_frac_nf", "ward_frac_nf", "st_frac_nf",'installer_frac_nf',"wq_frac_nf"]

quantity_transform = MultiLabelBinarizer()

def status_transform(d,value):
    return (d.status_group==value).astype(int).values.reshape(-1,1)


def summarize_categories(d,catgeories,abbrev):
    summary=d.pivot_table(index=catgeories, columns="status_group",aggfunc="count",values="id").\
              reset_index().\
              fillna(0).\
              assign(total = lambda x:x.eval("functional + functional_needs_repair + non_functional"))
    
    summary["{}_frac_f".format(abbrev)]=summary.eval("functional/total")
    summary["{}_frac_fnr".format(abbrev)]=summary.eval("functional_needs_repair/total")
    summary["{}_frac_nf".format(abbrev)]=summary.eval("non_functional/total")
    summary.columns.name=None
    return summary.drop(["functional","functional_needs_repair","non_functional","total"],axis=1)


def shape_data(d, no_labels=False):
    x = d[feature_cols].values

    if not no_labels:
        functional=status_transform(df,'functional')
        functional_needs_repair=status_transform(df,'functional_needs_repair')
        non_functional=status_transform(df,'non_functional')

        
    i = d.index.values
    if no_labels:
        print ("shape_data x size",x.shape)
        print ("shape_data i size",i.shape)
        return [x,i]
    else:
        return [x,functional,functional_needs_repair,non_functional,i]


def label_vector_to_name(label_vector):
    if label_vector.sum() != 1:
        return "unknown"        
    elif label_vector[0]==1:
        return "functional"
    elif label_vector[1]==1:
        return "functional_needs_repair"
    elif label_vector[2]==1:
        return "non_functional"
    else:
        return "unknown"


def get_final_prediction(row):
    if row.combo_pred != "unknown":
        return row.combo_pred
    elif row.nf_prob > row.f_prob:
        return 'non_functional'
    elif row.fnr_prob > row.f_prob:
        return "functional_needs_repair"
    else:
        return "functional"
    
def concat_probs(probs):
    return np.concatenate([np.array(probs[s]).reshape(-1,1) for s in statuses],1)

def mk_comparison_frame(d, probs, combo_pred, index):
    records={}
    for s in statuses:
        name = "{}_prob".format(s)
        records[name]=probs[s]
    records['combo_pred'] = [label_vector_to_name(lv) for lv in combo_pred]
    
    comparison=pd.DataFrame(records, index=index.reshape(1,-1)[0])
    comparison=d.merge(comparison,left_index=True,right_index=True)
    comparison["final_pred"]= comparison.apply(get_final_prediction,axis=1)
    return comparison

####################################################################
df=load_data('training')

qty_rates=summarize_categories(df,"quantity","qty")
ward_rates=summarize_categories(df,"ward","ward")
st_rates=summarize_categories(df,"source_type","st")
installer_rates=summarize_categories(df,"installer","installer")
wq_rates=summarize_categories(df,"water_quality","wq")

df=df.merge(qty_rates,on="quantity").\
      merge(ward_rates,on="ward").\
      merge(st_rates, on="source_type").\
      merge(installer_rates, on="installer").\
      merge(wq_rates, on="water_quality")




df.status_group=pd.Categorical(df.status_group,
                               categories=["functional","functional_needs_repair","non_functional"])

quantity_transform = quantity_transform.fit([df.quantity.unique()])

########################################
print("Shaping analysis data")

#input_samples=shape_data(df)
#analysis_samples=train_test_split(*input_samples)
analysis_samples=train_test_split(*shape_data(df))

train_features=analysis_samples[0]
train_functional=analysis_samples[2]
train_functional_needs_repair=analysis_samples[4]
train_non_functional=analysis_samples[6]
train_index=analysis_samples[8]

test_features=analysis_samples[1]
test_functional=analysis_samples[3]
test_functional_needs_repair=analysis_samples[5]
test_non_functional=analysis_samples[7]
test_index=analysis_samples[9]

train_labels = {"f":train_functional, "fnr":train_functional_needs_repair, "nf":train_non_functional}
train_labels["comb"]=np.concatenate([train_labels[s] for s in statuses],1)

test_labels  = { "f":test_functional,  "fnr":test_functional_needs_repair,  "nf":test_non_functional}
test_labels["comb"]=np.concatenate([test_labels[s] for s in statuses],1)

########################################
models={}
train_probs={}
test_probs={}
for s in statuses:
    print("=========")
    print("Training classifier for {}".format(s))
    rf=RandomForestClassifier(n_estimators=50,criterion='entropy',class_weight="balanced", min_samples_split=3)
    rf = rf.fit(train_features,
                train_labels[s].reshape(1,-1)[0])

    train_probs[s] = [p[1] for p in rf.predict_proba(train_features)]
    test_probs [s] = [p[1] for p in rf.predict_proba(test_features)]

    train_prediction = rf.predict(train_features)
    print ("Train accuracy", accuracy_score(train_labels[s],train_prediction))
    print ("Train precision", precision_score(train_labels[s],train_prediction))
    print ("Train recall", recall_score(train_labels[s],train_prediction))

    
    test_prediction = rf.predict(test_features)
    print ("Test accuracy", accuracy_score(test_labels[s],test_prediction))
    print ("Test precision", precision_score(test_labels[s],test_prediction))
    print ("Test recall", recall_score(test_labels[s],test_prediction))
    
    models[s]=rf

train_probs['comb'] = concat_probs(train_probs)
test_probs['comb'] = concat_probs(test_probs)

print("==============================")
print("Training global classifier")
rfcombo = RandomForestClassifier(n_estimators=50, criterion='entropy',class_weight="balanced")
rfcombo = rfcombo.fit(train_probs['comb'], train_labels['comb'])

train_combo_predict = rfcombo.predict(train_probs['comb'])
test_combo_predict  = rfcombo.predict(test_probs['comb'])
print("Train combo accuracy", accuracy_score(train_labels['comb'],train_combo_predict))
print("Test combo accuracy", accuracy_score(test_labels['comb'],test_combo_predict))


train_comparison=mk_comparison_frame(df, train_probs, train_combo_predict, train_index)
test_comparison=mk_comparison_frame(df, test_probs, test_combo_predict, test_index)
print("Final accuracy",test_comparison.eval("status_group==final_pred").mean())


print("==============================")
valid_df=load_data('testing')
print( valid_df.shape)
valid_df=valid_df.merge(qty_rates,on="quantity").\
                  merge(ward_rates,on="ward",how="left").\
                  merge(st_rates, on="source_type").\
                  merge(installer_rates, on="installer", how="left").\
                  merge(wq_rates, on="water_quality")
print( valid_df.shape)

valid_df.installer_frac_f.fillna(0.543,inplace=True)
valid_df.installer_frac_fnr.fillna(0.07267676767676767,inplace=True)
valid_df.installer_frac_nf.fillna(0.3842424242424242,inplace=True)

valid_df.ward_frac_f.fillna(0.543,inplace=True)
valid_df.ward_frac_fnr.fillna(0.07267676767676767,inplace=True)
valid_df.ward_frac_nf.fillna(0.3842424242424242,inplace=True)

print ("valid df shape", valid_df.shape)
print("Shaping validation data")
valid_samples=shape_data(valid_df, no_labels=True)
valid_features=valid_samples[0]
valid_index=valid_samples[1]

print("Applying classifiers")
valid_probs={}

for s in statuses:
    rf=models[s]
    valid_probs[s]=[p[1] for p in rf.predict_proba(valid_features)]
    print (len(valid_probs[s]))
valid_probs['comb'] = concat_probs(valid_probs)
valid_combo_predict=rfcombo.predict(valid_probs['comb'])
print (valid_combo_predict.shape)
valid_final_df = mk_comparison_frame(valid_df, valid_probs, valid_combo_predict, valid_index)

print("Writing results")
valid_final_df[['id','final_pred']].\
              rename(columns={'final_pred':'status_group'}).to_csv('submission.csv',index=['False'])
