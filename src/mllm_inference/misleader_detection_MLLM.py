from tqdm import tqdm
from utils import *
from llm_inference import *
from loaders import *
import transformers
import argparse
import os


transformers.set_seed(42)



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz',  help="Dataset to use")
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--model', type=str, required=True,  help="Name of the model to run")
    parser.add_argument('--max_tokens', type=int, default=200,  help="Max number of generated tokens")



    args = parser.parse_args()
    m = args.model
    max_tokens = args.max_tokens
    print(f'Generating detection answers for model {m}')
    #Load model
    if m in ['GPT41', 'o3', 'gemini-1.5-pro', 'gemini-1.5-flash']:
        model, tokenizer = m, ''
    else:
        # pass
        model, tokenizer = load_model(m)
    print('Model loaded')



    os.makedirs('results',exist_ok=True)
    root = f'results/{"-".join(args.model.split("/"))[:-1]}/'
    os.makedirs(root,exist_ok=True)

    prompt_real = """
    You are an expert in data visualization analysis. Your task is to identify misleaders present in the given visualization.

    Please carefully examine the visualization and detect its misleaders. Provide all relevant misleaders, up to three, as a comma separated list.
    In most cases only one misleader is relevant.
    If you detect none of the above types of misleaders in the visualization, respond with "no misleader".

    The available misleaders to select are, by alphabetical order:
    - discretized continuous variable: a map displays a continuous variable is transformed into a categorical variable by cutting it into discrete categories, thus exaggerating the difference between boundary cases.
    - dual axis: there are two independent y-axis, one on the left and one on the right, with different scales.
    - inappropriate axis range: the axis range is too broad or too narrow.
    - inappropriate item order: instances of a variable along an axis are in an unconventional, non-linear or non-chronological order.
    - inappropriate use of line chart: a line chart is used in inappropriate or unconventional ways, e.g., using a line chart with categorical variables, or encoding the time dimension on the y-axis.   
    - inappropriate use of pie chart: a pie chart does not display data in a part-to-whole relationship, e.g., its shares do not sum to 100%.
    - inconsistent binning size: a variable, such as years or ages, is grouped in unevenly sized bins.
    - inconsistent tick intervals: the tick values in one axis are not evenly spaced, e.g., the tick value sequence is 10, 20, 40, 45.
    - inverted axis: an axis is displayed in a direction opposite to conventions, e.g., the y-axis displays values increasing from top to bottom or the x-axis displays values increasing from right to left.
    - misrepresentation:  the value labels displayed do not match the size of their visual encodings, e.g., bars may be drawn disproportionate to the corresponding numerical value.
    - truncated axis: an axis does not start from zero, resulting in a visual exaggeration of changes in the dependent variable with respect to the independent variable. 
    - 3d: the visualization includes three-dimensional effects.

    Provide only the final answer, without additional explanation.
    """

    prompt_synth="""
    You are an expert in data visualization analysis. Your task is to identify misleaders present in the given visualization.

    Please carefully examine the visualization and detect its misleaders. Provide all relevant misleaders, up to three, as a comma separated list.
    In most cases only one misleader is relevant.
    If you detect none of the above types of misleaders in the visualization, respond with "no misleader".

    The available misleaders to select are, by alphabetical order:
    - dual axis: there are two independent y-axis, one on the left and one on the right, with different scales.
    - inappropriate axis range: the axis range is too broad or too narrow.
    - inappropriate item order: instances of a variable along an axis are in an unconventional, non-linear or non-chronological order.
    - inappropriate use of line chart: a line chart is used in inappropriate or unconventional ways, e.g., using a line chart with categorical variables, or encoding the time dimension on the y-axis.   
    - inappropriate use of pie chart: a pie chart does not display data in a part-to-whole relationship, e.g., its shares do not sum to 100%.
    - inconsistent binning size: a variable, such as years or ages, is grouped in unevenly sized bins.
    - inconsistent tick intervals: the tick values in one axis are not evenly spaced, e.g., the tick value sequence is 10, 20, 40, 45.
    - inverted axis: an axis is displayed in a direction opposite to conventions, e.g., the y-axis displays values increasing from top to bottom or the x-axis displays values increasing from right to left.
    - misrepresentation:  the value labels displayed do not match the size of their visual encodings, e.g., bars may be drawn disproportionate to the corresponding numerical value.
    - truncated axis: an axis does not start from zero, resulting in a visual exaggeration of changes in the dependent variable with respect to the independent variable. 
    - 3d: the visualization includes three-dimensional effects.

    Provide only the final answer, without additional explanation.
    """


    #Misleaders set
    model_answers = []  
    dataset = [d for d in json.load(open(f"data/{args.dataset}/{args.dataset}.json", encoding="utf-8")) if d['split'] == args.split]
        
    for d in tqdm(range(len(dataset))):


        image_path = dataset[d]['image_path']
        if args.dataset=='misviz':
            image_path = f"data/misviz/{image_path}"
        else:
            image_path = f"data/misviz_synth/{image_path}"

        predicted_answer = generate_answer(image_path, 
                                        prompt_real if args.dataset=='misviz' else prompt_synth, 
                                        tokenizer, 
                                        model, 
                                        m, 
                                        max_tokens)

        if m in ['GPT41', 'o3', 'gemini-1.5-pro', 'gemini-1.5-flash']:
            usage = predicted_answer[1] 
            predicted_answer = predicted_answer[0]
        else:
            usage=''
        model_answers.append({'image_path':dataset[d]['image_path'], 
                            'predicted_misleader':predicted_answer,
                            'true_misleader': dataset[d]['misleader'],
                            'chart_type': dataset[d]['chart_type'],
                            'usage': usage})


    with open(os.path.join(root, f'{args.dataset}_{args.split}.json'), 'w') as json_file:
                json.dump(model_answers, json_file, indent=4)