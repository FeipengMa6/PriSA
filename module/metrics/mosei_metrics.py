import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score
def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
def urfunny_metrics(y_true, y_pred, target_names,key_head='EVAL'):
    metrics_dict = {}
    acc = accuracy_score(y_true, y_pred)
    metrics_dict[key_head+'/mae'] = acc 
    metrics_dict[key_head+'/corr'] = acc 
    metrics_dict[key_head+'/acc_5'] = acc
    metrics_dict[key_head+'/acc_7'] = acc
    metrics_dict[key_head+'/non0_f1-score'] = acc
    metrics_dict[key_head+'/has0_f1-score'] = acc
    return metrics_dict
def mosei_metrics(y_true, y_pred, target_names,key_head='EVAL'):
    metrics_dict = {}
    test_preds = y_pred
    test_truth = y_true
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    metrics_dict[key_head+'/non0_acc2'] = accuracy_score(binary_truth, binary_preds)
    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)
    metrics_dict[key_head+'/has0_acc2'] = accuracy_score(binary_truth, binary_preds)
    has0_f_score = f1_score(binary_truth, binary_preds, average='weighted')
    test_preds_a7 = np.round(np.clip(test_preds, a_min=-3., a_max=3.))
    test_truth_a7 = np.round(np.clip(test_truth, a_min=-3., a_max=3.))
    test_preds_a5 = np.round(np.clip(test_preds, a_min=-2., a_max=2.))
    test_truth_a5 = np.round(np.clip(test_truth, a_min=-2., a_max=2.))
    mae = np.mean(np.absolute(test_preds - test_truth))   
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    non0_f_score = f1_score((test_truth[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
    target_names = ['strong neg','neg','weakly neg','netural','weakly pos','pos','strong pos']
    metrics_dict[key_head+'/mae'] = mae 
    metrics_dict[key_head+'/corr'] = corr 
    metrics_dict[key_head+'/acc_5'] = mult_a5
    metrics_dict[key_head+'/acc_7'] = mult_a7
    metrics_dict[key_head+'/non0_f1-score'] = non0_f_score
    metrics_dict[key_head+'/has0_f1-score'] = has0_f_score
    return metrics_dict
def sims_metrics(y_true, y_pred,target_names,key_head='EVAL'):
    metrics_dict = {}
    test_preds = np.clip(y_pred, a_min=-1., a_max=1.)
    test_truth = np.clip(y_true, a_min=-1., a_max=1.)
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    ms_2 = [-1.01, 0.0, 1.01]
    test_preds_a2 = test_preds.copy()
    test_truth_a2 = test_truth.copy()
    for i in range(2):
        test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
    for i in range(2):
        test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i
    ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
    test_preds_a5 = test_preds.copy()
    test_truth_a5 = test_truth.copy()
    for i in range(5):
        test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
    for i in range(5):
        test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
    mae = np.mean(np.absolute(test_preds - test_truth))   
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a2 = multiclass_acc(test_preds_a2, test_truth_a2)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')
    metrics_dict[key_head+'/mae'] = mae 
    metrics_dict[key_head+'/corr'] = corr 
    metrics_dict[key_head+'/non0_acc2'] = mult_a2
    metrics_dict[key_head+'/has0_acc2'] = mult_a2
    metrics_dict[key_head+'/acc_5'] = mult_a5
    metrics_dict[key_head+'/acc_7'] = mult_a5
    metrics_dict[key_head+'/non0_f1-score'] = f_score
    metrics_dict[key_head+'/has0_f1-score'] = f_score
    return metrics_dict
