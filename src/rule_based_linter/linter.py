from tqdm import tqdm
import json
import argparse
import os


def is_truncated(axis_metadata):
    '''
    Rule check for truncated axis.
    '''
    try:
        if 'y1' in axis_metadata['axis']:
            y1_axis_first_tick = sorted([axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y1'])[0]
        else:
            y1_axis_first_tick = 0
        if 'y2' in axis_metadata['axis']:
            y2_axis_first_tick = sorted([axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y2'])[0]
        else:
            y2_axis_first_tick = 0
        if type(y1_axis_first_tick) in [int, float]:
            if y1_axis_first_tick > 0: 
                return True
        if type(y2_axis_first_tick) in [int, float]:
            if y2_axis_first_tick > 0: 
                return True
        return False
    except:
        return False



def is_inverted(axis_metadata):
    '''
    Rule check for inverted axis.
    '''
    if 'y1' in axis_metadata['axis']:
        try:
            y1_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y1']
            if sorted(y1_axis)==y1_axis[::-1]:
                return True
        except:
            pass
    if 'y2' in axis_metadata['axis']:
        try:
            y2_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y2']
            if sorted(y2_axis)==y2_axis[::-1]:
                return True
        except:
            pass
    if 'x' in axis_metadata['axis']:
        try:
            #Only verify it if the x axis can be converted to float values (it is numerical values or dates)
            x_axis = [float(axis_metadata['label'][d]) for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='x']
            if sorted(x_axis)==x_axis[::-1]:
                return True
        except:
            pass
    return False


def is_inconsistent_tick(axis_metadata):
    '''
    Rule check for inconsitent tick intervals.
    '''
    if 'y1' in axis_metadata['axis']:
        y1_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y1']
        try:
            
            intervals = [round(float(y1_axis[i+1]) - float(y1_axis[i]), 2) for i in range(len(y1_axis)-1)]
            if len(list(set(intervals)))!=1:
                return True
        except:
            pass
                                                    
    if 'y2' in axis_metadata['axis']:
        y2_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y2']
        try:
            intervals = [round(float(y2_axis[i+1]) - float(y2_axis[i]), 2) for i in range(len(y2_axis)-1)]
            if len(list(set(intervals)))!=1:
                return True
        except:
            pass
    if 'x' in axis_metadata['axis']:
        x_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='x']
        #They need to be in increasing or decreasing order but not random (otherwise it is inappropriate item order)
        try:
            if sorted(x_axis) == x_axis or sorted(x_axis)==x_axis[::-1]:
                    intervals = [round(float(x_axis[i+1]) - float(x_axis[i]),2) for i in range(len(x_axis)-1)]
                    if len(list(set(intervals)))!=1:
                        return True
        except:
            pass
    return False


def is_inconsistent_binning(axis_metadata):
    '''
    Rule check for inconsistent binning size.
    '''
    if 'x' in axis_metadata['axis']:
        try:
            x_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='x']
            if '-' in x_axis[0]:
                x_axis = [x.split('-') for x in x_axis] 
                intervals = [int(x[1])-int(x[0]) for x in x_axis]
            elif 'to' in x_axis[0].lower():
                x_axis = [x.lower().split('to') for x in x_axis] 
                intervals = [int(x[1])-int(x[0]) for x in x_axis]
            else:
                return False
            
            if len(list(set(intervals)))!=1:
                return True
        except:
            pass
    return False


def is_inappropriate_order(axis_metadata):
    '''
    Rule check for inappropriate item order (one specific case: dates are not chronological on the x-axis).
    '''
    if 'x' in axis_metadata['axis']:
        try:
            x_axis = [int(axis_metadata['label'][d]) for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='x']
            if not sorted(x_axis) == x_axis and not sorted(x_axis)==x_axis[::-1]:
                return True
        except:
            pass
    return False
    

def is_dual(axis_metadata):
    '''
    Rule check for dual axis.
    '''
    if 'y1' in axis_metadata['axis']:
        y1_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y1']
    else:
        return False
    if 'y2' in axis_metadata['axis']:
        y2_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y2']
    else:
        return False
    if y1_axis != y2_axis:
        return True
    else:
        return False
    

def coerce_number(value):
    try:
        return int(value)        
    except ValueError:
        try:
            return float(value)    
        except ValueError:
            return value          

def process_axis_data(axis_str):
    #Convert the axis extracted by the fine-tuned DePlot model to a dictionary format that can be used by the linter    
    if '\n' in axis_str:
        axis = axis_str.split('\n')[1:]
    else:
        axis = axis_str.split('<0x0A>')[1:]
    axis_dict = {'axis': [], 'label': [], 'Relative position': []}
    for item in axis:
        try:
            _, ax, la, rel = item.split(' | ')
            axis_dict['axis'].append(ax.strip())
            axis_dict['label'].append(coerce_number(la.strip().replace('%','')))
            axis_dict['Relative position'].append(rel.strip())
        except:
            pass

    return axis_dict


def get_linter_predictions(dataset='misviz', split = 'test', predicted=False):
    
    axis_path = f"src/output/predicted_axis_misviz/merged_axis_data_{dataset}.json"
    metadata = [m for m in json.load(open(f"data/{args.dataset}/{args.dataset}.json", encoding="utf-8")) if m['split']==split]
    axis_data = json.load(open(axis_path, encoding="utf-8"))[split]
   
    if dataset=='misviz_synth' and not predicted:
        axis_metadata = [process_axis_data(a['true_axis']) for a in axis_data]
    else:
        axis_metadata = [process_axis_data(a['axis_generation']) for a in axis_data]
    predictions = []

    for m in tqdm(range(len(axis_metadata))):
        pred = []
        if is_inverted(axis_metadata[m]):
            pred.append('inverted axis')

        if is_truncated(axis_metadata[m]):
            pred.append('truncated axis')

        if is_inconsistent_tick(axis_metadata[m]):
            pred.append('inconsistent tick intervals')

        if is_inconsistent_binning(axis_metadata[m]):  
            pred.append('inconsistent binning size')

        if is_dual(axis_metadata[m]):
            pred.append('dual axis')

        if is_inappropriate_order(axis_metadata[m]):
            pred.append('inappropriate item order')

        if len(pred)==0:
            #No misleader check was passed
            pred = ['no misleader']
        predictions.append(pred)

    results = []
    for p in range(len(predictions)):
        results.append(
            {
            'image_path': metadata[p]['image_path'],
            'predicted_misleader': ','.join(predictions[p]),
            'true_misleader': metadata[p]['misleader']
            }
        )
    return results

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz',  help="Dataset to use")
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--use_predicted_axis',type=int, default=0, help='If 1, use the predicted axis. Otherwise, use the ground truth one.')
    args = parser.parse_args()


    results = get_linter_predictions(args.dataset, split=args.split, predicted=args.use_predicted_axis)
    os.makedirs('results', exist_ok=True)
    if args.use_predicted_axis:
        os.makedirs('results/linter', exist_ok=True)
        output_file = f'results/linter/{args.dataset}_{args.split}.json'
    else:
        os.makedirs('results/linter_gt', exist_ok=True)
        output_file = f'results/linter_gt/{args.dataset}_{args.split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)