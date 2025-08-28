import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils import *


def post_process_pred(pred):
    #Convert the predictions to a list
    if 'no misleader' in pred.lower():
        pred = []
    elif ',' in pred:
        pred = pred.lower().replace('\n','').split(',')
    else:
        pred = [pred.lower().replace('\n','')]
    return pred

def compute_metrics(data, subset=True):
    if subset:
        #Restrict to instances that have only pie, bar, or line charts
        data = [d for d in data if not 'map' in d['chart_type'] and not 'scatter plot' in d['chart_type'] and not 'other' in d['chart_type']]
    #binary metrics
    binary_predictions = ['misleading' if len(post_process_pred(d['predicted_misleader']))>=1 else 'non-misleading' for d in data]
    binary_ground_truths = ['misleading' if len(d['true_misleader'])>=1 else 'non-misleading' for d in data]
    acc = round(100*accuracy_score(binary_ground_truths, binary_predictions),2)
    f1 = round(100*f1_score(binary_ground_truths, binary_predictions, average='macro'),2)
    precision = round(100*precision_score(binary_ground_truths, binary_predictions,  pos_label='misleading'),2)
    recall = round(100*recall_score(binary_ground_truths, binary_predictions, pos_label='misleading'),2)
    #multiclass metrics
    PM = []
    EM = []
    for d in data:
        pred = set(post_process_pred(d['predicted_misleader']))
        gt = set(d['true_misleader'])
        #Only compute this for the instances with misleaders
        if len(gt) >=1:
            if len(pred)==0:
                EM.append(0)
                PM.append(0)
            elif pred==gt:
                EM.append(1)
                PM.append(1)
            elif pred.issubset(gt): 
                EM.append(0)
                PM.append(1)
            else:
                EM.append(0)
                PM.append(0)


    EM = round(100*sum(EM)/len(EM), 2)
    PM = round(100*sum(PM)/len(PM), 2)

    
    print(f"Accuracy: {acc} | Precision: {precision} | Recall: {recall} | F1: {f1} | Exact match misleader {EM} | Partial match misleader {PM}")
    return acc, precision, recall, f1, EM, PM



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz',  help="Dataset to use")
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--model', type=str, required=True,  help="Name of the model to run")
    parser.add_argument('--use_subset', type=int, required=True, default=1,  help="if 1, provide results only on the subset of instances with bar, line, and pie charts.")
    args = parser.parse_args()

    print('-----------------------------------------')
    metadata = load_json(f"data/{args.dataset}/{args.dataset}.json")

    print(f"{args.model} - {args.dataset}")
    if args.use_subset:
        print("Results reported on subset with bar, line, and pie charts")
    results = load_json(f"results/{args.model}/{args.dataset}_{args.split}.json")
    compute_metrics(results,subset=args.use_subset)
    print('-----------------------------------------')