import csv
import argparse
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

#provide inputs
parser = argparse.ArgumentParser()
parser.add_argument('--det',type=str) #give path to detection files
parser.add_argument('--gt',type=str) #give path to ground truth files
args = parser.parse_args()

#graph style
sns.set_style('whitegrid')

image_names= []
driver_ground_truth= []
front_ground_truth= []
back_ground_truth= []

# retrieve ground truth
for root, dirs, files in os.walk(args.gt):
    if not files:
        continue
    prefix = os.path.basename(root)
    files.sort()
    for f in files:
        if f.endswith('.txt'):
            image_names.append(f)
            with open(os.path.join(root, f)) as txt_file:
                driver_found= False
                front_found= False
                back_found= False
                for line in txt_file:
                    type= line.split()[0]
                    if type=='0':
                        driver_found = True
                    elif type=='1':
                        front_found = True
                    elif type=='2':
                        back_found = True
                if driver_found==False:
                    driver_ground_truth.append(0)
                else:
                    driver_ground_truth.append(1)
                if front_found==False:
                    front_ground_truth.append(0)
                else:
                    front_ground_truth.append(1)
                if back_found==False:
                    back_ground_truth.append(0)
                else:
                    back_ground_truth.append(1)

#prediction of driver in images
driver_pred_res= []
vertical_limit= 240
horizontal_limit= 360
bbox_area_limit= 30000
i= 0
for root, dirs, files in os.walk(args.det):
    if not files:
        continue
    prefix = os.path.basename(root)
    files.sort()
    for f in files:
        if f.endswith('.txt'):
            if os.path.splitext(f)[0]==os.path.splitext(image_names[i])[0]:
                with open(os.path.join(root, f)) as txt_file:
                    prob= []
                    for line in txt_file:
                        type= line.split()[0]
                        xmin = float(line.split()[2])
                        ymin = float(line.split()[3])
                        xmax = float(line.split()[4])
                        ymax = float(line.split()[5])
                        conf = float(line.split()[1])
                        bbox_area= (xmax-xmin)*(ymax-ymin)
                        x= (xmin+xmax)/2
                        y= (ymin+ymax)/2
                        if bbox_area>bbox_area_limit and x>horizontal_limit and y>vertical_limit:
                            prob.append(conf)
                temp = 1
                for j in prob:
                    temp *= (1-j)
                final_prob= (1-temp)
                driver_pred_res.append(final_prob)
                i+=1

#prediction of front passenger in images
front_pred_res= []
vertical_limit= 240
horizontal_limit= 360
bbox_area_limit= 30000
i= 0
for root, dirs, files in os.walk(args.det):
    if not files:
        continue
    prefix = os.path.basename(root)
    files.sort()
    for f in files:
        if f.endswith('.txt'):
            if os.path.splitext(f)[0]==os.path.splitext(image_names[i])[0]:
                with open(os.path.join(root, f)) as txt_file:
                    prob= []
                    for line in txt_file:
                        type= line.split()[0]
                        xmin = float(line.split()[2])
                        ymin = float(line.split()[3])
                        xmax = float(line.split()[4])
                        ymax = float(line.split()[5])
                        conf = float(line.split()[1])
                        bbox_area= (xmax-xmin)*(ymax-ymin)
                        x= (xmin+xmax)/2
                        y= (ymin+ymax)/2
                        if bbox_area>bbox_area_limit and x<horizontal_limit and y>vertical_limit:
                            prob.append(conf)
                temp = 1
                for j in prob:
                    temp *= (1-j)
                final_prob= (1-temp)
                front_pred_res.append(final_prob)
                i+=1

#prediction of back passenger in images
back_pred_res= []
horizontal_limit1= 250
horizontal_limit2= 720-250
bbox_area_limit= 0
i= 0
for root, dirs, files in os.walk(args.det):
    if not files:
        continue
    prefix = os.path.basename(root)
    files.sort()
    for f in files:
        if f.endswith('.txt'):
            if os.path.splitext(f)[0]==os.path.splitext(image_names[i])[0]:
                with open(os.path.join(root, f)) as txt_file:
                    prob= []
                    for line in txt_file:
                        type= line.split()[0]
                        xmin = float(line.split()[2])
                        ymin = float(line.split()[3])
                        xmax = float(line.split()[4])
                        ymax = float(line.split()[5])
                        conf = float(line.split()[1])
                        bbox_area= (xmax-xmin)*(ymax-ymin)
                        x= (xmin+xmax)/2
                        y= (ymin+ymax)/2
                        if bbox_area>bbox_area_limit and x>horizontal_limit1 and x<horizontal_limit2:
                            prob.append(conf)
                temp = 1
                for j in prob:
                    temp *= (1-j)
                final_prob= (1-temp)
                back_pred_res.append(final_prob)
                i+=1

# confusion matrix for driver 
print('driver')
limit= 0.87
yconf= []
for i in driver_pred_res:
    if i<limit:
        yconf.append(0)
    else:
        yconf.append(1)
a= confusion_matrix(driver_ground_truth, yconf).ravel()
if len(a)==1:
    print("all true positive")
else:
    tn, fp, fn, tp = a
    acc= (tp+tn)/(tn+ fp+ fn+ tp)
    print("True Positives: "+str(tp))
    print("True Negatives: "+str(tn))
    print("False Positives: "+str(fp))
    print("False Negatives: "+str(fn))
    print("Accuracy: "+str(acc))

# confusion matrix for front seat passenger
print("front seat")
limit= 0.89
yconf= []
for i in front_pred_res:
    if i<limit:
        yconf.append(0)
    else:
        yconf.append(1)
tn, fp, fn, tp = confusion_matrix(front_ground_truth, yconf).ravel()
acc= (tp+tn)/(tn+ fp+ fn+ tp)
print("True Positives: "+str(tp))
print("True Negatives: "+str(tn))
print("False Positives: "+str(fp))
print("False Negatives: "+str(fn))
print("Accuracy: "+str(acc))

# confusion matrix for back seat passenger
print("back seat")
limit= 0.11
yconf= []
for i in back_pred_res:
    if i<limit:
        yconf.append(0)
    else:
        yconf.append(1)
tn, fp, fn, tp = confusion_matrix(back_ground_truth, yconf).ravel()
acc= (tp+tn)/(tn+ fp+ fn+ tp)
print("True Positives: "+str(tp))
print("True Negatives: "+str(tn))
print("False Positives: "+str(fp))
print("False Negatives: "+str(fn))
print("Accuracy: "+str(acc))

# # Optional
# # ROC curve
# fpr, tpr, thresholds = metrics.roc_curve(front_ground_truth, front_pred_res, pos_label=1)
# pyplot.plot(fpr, tpr, marker='.', label='Front Seat Passenger')
# fpr, tpr, thresholds = metrics.roc_curve(back_ground_truth, back_pred_res, pos_label=1)
# pyplot.plot(fpr, tpr, marker='.', label='Back Seat Passenger')
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.legend()
# pyplot.savefig('ROC.png')
# pyplot.close()

# # Optional
# # calculate AUC
# # example
# roc_auc = roc_auc_score(front_ground_truth, front_pred_res)
# print('ROC curve AUC: %.3f' % roc_auc)

# # Optional
# # Find for which confidence there is the maximum F1 score
# def frange(start, stop, step):
#     i = start
#     while i < stop:
#         yield i
#         i += step
# # print(ground_truth)
# max= 0
# yhat= []
# for limit in frange(0.5,1.0,0.001):
#     yhat= []
#     for i in front_pred_res:
#         if i<limit:
#             yhat.append(0)
#         else:
#             yhat.append(1)
#     pr_f1 = f1_score(front_ground_truth, yhat)
#     if pr_f1>max:
#         max= pr_f1
#         index= limit
# print('PR curve max F1: %.3f at confidence threshold %.3f' %(max,index))