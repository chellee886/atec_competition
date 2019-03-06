import pandas as pd
import math
from sklearn.metrics import log_loss
test= pd.read_csv("../data/test", sep='\t', header=None)
# test = pd.read_csv("data/test.csv")
test.columns = ["id", "question1", "question2", "is_duplicate"]
# test = test.iloc[:5000, :]
preds = pd.read_csv('pred30.csv', sep='\t', header=None, encoding='utf-8')

pred_list = preds.iloc[:, 1].tolist()
test_list = test['is_duplicate'].tolist()

print(log_loss(test_list, pred_list))
def readResult(y_test,results):
    index=0
    p=n=tp=tn=fp=fn=0.0
    for predLabel in results:
        if y_test[index]>0:
            p+=1
            if predLabel>0:
                tp+=1
            else:
                fn+=1
        else:
            n+=1
            if predLabel==0:
                tn+=1
            else:
                fp+=1
        index+=1

    acc=(tp+tn)/(p+n)
    precisionP=tp/(tp+fp)
    precisionN=tn/(tn+fn)
    recallP=tp/(tp+fn)
    recallN=tn/(tn+fp)
    gmean=math.sqrt(recallP*recallN)
    f_p=2*precisionP*recallP/(precisionP+recallP)
    f_n=2*precisionN*recallN/(precisionN+recallN)
    print ('{gmean:%s recallP:%s recallN:%s} {precP:%s precN:%s fP:%s fN:%s} acc:%s' %(gmean,recallP,recallN,precisionP,precisionN,f_p,f_n,acc))
    # print('AUC %s' %average_precision_score(y_test,results))

    output=open('result.output','w')
    output.write('\n'.join(['%s' %r for r in results]))

readResult(test_list, pred_list)
