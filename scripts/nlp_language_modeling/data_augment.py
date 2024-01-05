from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model, SentencePieceTokenizer
import argparse
import random
import math
import json
import os



def remove_fist_occureces(id_list, ids_to_drop):

    ids_to_drop_copy = ids_to_drop.copy()
    new_list = []
    for id_ in id_list:
        if id_ in ids_to_drop_copy:
            ids_to_drop_copy.remove(id_)
        else:
            new_list.append(id_)
    return new_list

def augment_one_sample_drop(example, percent):

    input_ids = example.copy()
    
    num_drop = math.floor(len(input_ids)*percent)

    ids_to_drop = set(random.sample(input_ids, num_drop))

    input_ids = remove_fist_occureces(input_ids, ids_to_drop)
    
    return input_ids

def generate_array(size, percentage_true):

    num_true = int(size * percentage_true)
    array = [True] * num_true + [False] * (size - num_true)
    random.shuffle(array)
    return array

def save_to_jsonl(jsons_array, save_path):
    
    with open(save_path, 'w') as file:

        for line in jsons_array:
            json_string = json.dumps(line)
            file.write(json_string+'\n')

    print(f'File of json lines is stored at {save_path}')



def augment_whole_dataset_drop(dataset_path, tokenizer, percent_perturb, save_path): # percent_perturb,
   # {"input": "Please classify if my sentence contains negative sentiment or positive sentiment Q : hide new secretions from the parental units ", "output": "False"}
    results = []

    with open(dataset_path) as file:
        which_to_perturb = generate_array(sum(1 for line in file.readlines()), percent_perturb)
    
    with open(dataset_path) as file:
        i = 0
        for line in file.readlines():
            line
            json_line = json.loads(line)

            if which_to_perturb[i]:
                input_text_tokens = tokenizer.text_to_ids(json_line['input'])
                augmented_tokens = augment_one_sample_drop(input_text_tokens, 0.1)
                new_data = {'input' : tokenizer.ids_to_text(augmented_tokens), 'output' : json_line['output']}
                results.append(new_data)
                
            results.append(json_line)
            i+=1

    save_to_jsonl(results, save_path)
                
            

def main(args):

    if args.augment_type == 'Drop_token':
        if not os.path.exists(args.tk_path):
            model_file, vocab_file = create_spt_model(
                data_file=args.data_file,
                vocab_size=args.vocab_size,
                sample_size=args.sample_size,
                do_lower_case=args.do_lower_case,
                tokenizer_type=args.tk_type,
                output_dir=args.o_dir,
                character_coverage=args.char_cov
                # Include other parameters if necessary
            )

        tknzr = SentencePieceTokenizer(args.tk_path)

        augment_whole_dataset_drop(args.data_file, tknzr, args.percent_to_perturb, args.output_path)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', default='/GPT_project/GPT_SFT_preprocessed_data/text_data_sent_anly/train.jsonl')
    parser.add_argument('--sample_size', default=100000)
    parser.add_argument('--vocab_size', default=32000)
    parser.add_argument('--do_lower_case', default=True)
    parser.add_argument('--tk_type', default='unigram')
    parser.add_argument('--o_dir', default='/.')
    parser.add_argument('--char_cov', default=1.0)
    parser.add_argument('--tk_path', default='/GPT_PROJECT/tokenizer.model')
    parser.add_argument('--percent_to_perturb', default=0.3, help='Which percent of whole dataset will be perturbed')
    parser.add_argument('--percent_of_tokens', default=0.1, help='One sequence perturbation number')
    parser.add_argument('--output_path', default='/GPT_project/GPT_SFT_preprocessed_data/train_drop_augmented.jsonl')
    parser.add_argument('--augment_type', default='Drop_token')

    args = parser.parse_args()


    main(args)

    with open(args.data_file) as file:
        print(sum(1 for i in file.readlines()))

    with open(args.output_path) as file:
        print(sum(1 for i in file.readlines()))

