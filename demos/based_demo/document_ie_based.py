import re
import time
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# from based.models.transformer.gpt import GPTLMHeadModel
from based.models.baselines.mamba_model import MambaLMHeadModel
from train.src.models.gpt import GPTLMHeadModel as BasedGPTLMHeadModel

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load pretrained models
def get_model(args, model_name="attn"): 
    print("\nLoading pretrained models...")
    if 'attn' in model_name:
        return BasedGPTLMHeadModel.from_pretrained_hf(
            "hazyresearch/attn-360m"
        ).to("cuda").to(torch.bfloat16)
    elif 'mamba' in model_name:
        return MambaLMHeadModel.from_pretrained_hf(
            "hazyresearch/mamba-360m"
        ).to("cuda").to(torch.bfloat16) 
    else:
        return BasedGPTLMHeadModel.from_pretrained_hf(
            "hazyresearch/my-awesome-model",
            device="cuda", 
            implementation='tk',           # choices are [fla_parallel, tk]
            silent=True,           
            inference_bs=args.bs,
        ).to(torch.bfloat16)


# Setup tokenizer
def get_data(task_name="hazyresearch/based-fda"):
    print(f"Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = " "
    tokenizer.pad_token_id = 220
    ds = load_dataset(task_name)
    return tokenizer, ds


# Consruct the model input
def get_inputs(tokenizer, batch, context_length, task_name):
    input_texts = [b.strip() for b in batch['text']]
    targets = batch['value']
    if any(e in task_name for e in ['squad']): 
        questions = batch['question']
    else:
        questions = batch['key']

    has_answer = True
    short_texts = []
    for context, answer, question in zip(input_texts, targets, questions):
        
        context = context.strip(".")
        if context.lower().endswith(question.lower()):
            context = context[:-len(question.lower())]
        elif context.lower().endswith(question.lower() + ":"):
            context = context[:-(len(question.lower()) + 1)]
        question = question[0].upper() + question[1:]

        doc_tokens = tokenizer.batch_encode_plus([context], return_tensors="pt", padding=True, truncation=False)['input_ids'][0]
        
        answer_pos = -1
        if type(answer) == list: answer = answer[0]
        if(answer == "" or len(answer) <= 1): return instances, new_doc_set
        answer_pattern = re.compile(re.escape(answer), re.IGNORECASE)
        if answer_match := answer_pattern.search(context):
            if answer_pos == -1 or answer_pos > answer_match.start():
                answer_pos = answer_match.start()
        if not has_answer: # e.g., summarization datasets
            answer_pos = 0

        # Convert the answer_pos to a token value
        new_text = context[:answer_pos]
        new_text_toks = tokenizer.batch_encode_plus([new_text], return_tensors="pt",)['input_ids'][0]
        answer_tok_pos = len(new_text_toks)
        if answer_pos == -1 and 'tok_pos' in doc and len(new_text_toks) > context_length:
            answer_tok_pos = doc['tok_pos']

        # pick new bounds
        half_length = context_length // 2
        start = max(0, answer_tok_pos - half_length)
        completed_length = answer_tok_pos - start
        remaining_length = context_length - completed_length
        end = min(len(doc_tokens), answer_tok_pos + remaining_length)
        subset_tokens = doc_tokens[start:end]
        short_context = tokenizer.decode(subset_tokens, skip_special_tokens=True)

        # prompt construction
        if any(e in task_name for e in ["fda", "swde"]): 
            question = question + ":"
        short_context = short_context + ". " + question 

        # print(short_context[-50:])
        short_texts.append(short_context)

    input_texts = short_texts

    inputs = tokenizer.batch_encode_plus(
        input_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048
    ).input_ids.to("cuda")
    return inputs, questions, targets


def main(args):
    task_name = args.task_name 
    model_name = args.model_name 
    bs = args.bs
    generation_length, context_length = args.generation_length, args.context_length
    model = get_model(args, model_name)
    tokenizer, ds = get_data(task_name)

    # Collect predictions
    results = []
    total_time = []
    preds_list = []
    values_list = []
    for i in tqdm(range(0, len(ds['validation']), bs)):
        if i > 50: break
        batch = ds['validation'][i:i+bs]
        inputs, questions, targets = get_inputs(tokenizer, batch, context_length, task_name)
        limit = inputs.shape[-1] + generation_length
        start = inputs.shape[-1]

        model.eval()
        fn = model.generate
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            generations = fn(
                input_ids=inputs,
                max_length=limit,
                temperature=0.1,
                top_k=1,
                top_p=1.0,
                cg=True
            )
            torch.cuda.synchronize()
            end_time = time.time()

        preds = generations[:, start:]
        preds = preds.tolist()
        preds = tokenizer.batch_decode(preds)
        seen_inputs = tokenizer.batch_decode(inputs)
        for inp, p, k, v in zip(seen_inputs, preds, questions, targets):
            correct = v.lower() in p.lower()
            results.append(correct)
            preds_list.append(p.split("\n")[0])
            values_list.append(v.split("\n")[0])
        total_time.append(end_time-start_time)


    # Results
    print(f"{model_name=}: score={sum(results)/len(results)}; total_time={sum(total_time)} seconds")
    output_file = f"{model_name}_predictions_and_values.txt"
    with open(output_file, 'w') as f:
        for i, (pred, value) in enumerate(zip(preds_list, values_list)):
            f.write(f"Example {i}:\n- Prediction: {pred}\n- Value: {value}\n")
    print(f"Predictions and values saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model and data configuration")

    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--context_length', type=int, default=1500, help='Context length')
    parser.add_argument('--generation_length', type=int, default=48, help='Generation length')
    parser.add_argument(
        '--model_name', type=str, default="based", choices= ["mamba", "attn", "based"], help='Name of the model'
    )
    parser.add_argument('--task_name', type=str, default="hazyresearch/based-fda", help='Name of the task/dataset')

    args = parser.parse_args()
    main(args)

