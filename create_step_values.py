import torch
import numpy as np
from transformers import AutoTokenizer
from accelerate import Accelerator
import classifier_lib
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, Dataset, Features, Sequence, Value
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from IPython import embed
from huggingface_hub import login
from collections import defaultdict

class dualinputdataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data[idx]['is_correct'] is not None:
            rew = float(self.data[idx]['is_correct'])
        else:
            rew = 0.0

        return {
            "prompt": self.data[idx]['messages'],
            "response": self.data[idx]['response'],
            "reward": rew,
            "id": idx
        }

def dual_input_collate(batch, tokenizer):
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    ids = [item['id'] for item in batch]
    rewards = [item['reward'] for item in batch]

    formatted_prompts = [
        tokenizer.apply_chat_template(
            prompts[i], 
            tokenize = False,
            add_generation_prompt=True
        ) for i in range(len(prompts)) 
    ]
    tokenized_prompts = tokenizer(
        formatted_prompts,
        padding = False,
    )
    prompts_len = [len(tokenized_prompts['input_ids'][i]) for i in range(len(tokenized_prompts['input_ids']))]
    combined = [
        formatted_prompts[i] + responses[i][8:] for i in range(len(prompts))
    ]
    tokenizer.padding_side = "right"
    tokenized_combined = tokenizer(
        combined,
        truncation=True,
        max_length = 4096*4 + 600,  # use 600 for prompt length
        padding = "max_length",
        return_tensors="pt",
        return_attention_mask=True
    )
    tokenized_inputs = tokenized_combined['input_ids']
    attention_masks = tokenized_combined['attention_mask']
    assert tokenized_inputs.shape == attention_masks.shape

    return {
        "prompts_len": torch.tensor(prompts_len, dtype=torch.long),
        "tokenized_inputs": tokenized_inputs, 
        "attention_masks": attention_masks,
        "rewards": torch.tensor(rewards, dtype=torch.float16),
        "id": torch.tensor(ids, dtype=torch.long)
    }


def debug_collate(batch):
    print(f"Collating batch of size: {len(batch)}")
    return torch.utils.data.default_collate(batch)


def post_processing_batch(batch, outputs, step = 4096):
    #processed_data = defaultdict(list)
    prompt_lens = batch['prompts_len'].tolist()
    #processed_data['prompts_len'].add(prompt_lens)
    bs = len(prompt_lens)

    batch_step_level_probs = [] #list of list of floats
    for i in range(bs):
        prompt_len = prompt_lens[i]
        p_gen_tokens = batch['tokenized_inputs'][i]
        valid_p_gen_len = torch.sum(batch['attention_masks'][i])
        valid_p_gen_tokens = p_gen_tokens[:valid_p_gen_len]
        valid_success_probs = outputs['success_probs'][i][:valid_p_gen_len]
        p_tokens = valid_p_gen_tokens[:prompt_len]
        gen_tokens = valid_p_gen_tokens[prompt_len:]
        max_gen_len = min(step*4, gen_tokens.size(0))
        gen_tokens = gen_tokens[:max_gen_len]

        step_level_probs = []
        for j in range(0, max_gen_len, step):
            start_id = j
            p = float(valid_success_probs[prompt_len + start_id - 1]) #start at the end token of the prompt.
            step_level_probs.append(p)
        
        batch_step_level_probs.append(step_level_probs)

    return batch_step_level_probs


def generate_values(
    data_path, value_model_path, tokenizer_name,
    batch_size,
    max_gen_len,  
    accelerator,
    device
):
    
    original_dataset = load_dataset(data_path)['train']
    original_dataset = original_dataset.shuffle(seed= 42)
    print("number of rows in the dataset: {}".format(len(original_dataset)))
    print(original_dataset[0].keys())
    
    dataset = dualinputdataset(original_dataset.select(range(1000)))

    #setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_bos_token = False #do not add bos token when call encode function
    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id
    think_token_id = tokenizer.encode('<think>')[0] 
    stop_think_token_id = tokenizer.encode('</think>')[0]


    #sampler = DistributedSampler(
    #    dataset,
    #    num_replicas=accelerator.num_processes,
    #    rank=accelerator.process_index,
    #    shuffle=False  # or False for inference
    #)

    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        #collate_fn = debug_collate,
        drop_last=False,
        #sampler = sampler, 
        collate_fn = lambda b: dual_input_collate(b, tokenizer),
    )
    dataloader = accelerator.prepare(dataloader)

    #print(f"[Rank {accelerator.process_index}] sampler length: {len(sampler)}")
    print(f"[Rank {accelerator.process_index}] expected batches: {len(dataloader)}")
    #print(f"[Rank {accelerator.process_index}] sampler length: {len(dataloader.sampler)}")

    model_loading_kwargs = dict(attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, use_cache=False)
    classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(value_model_path,
                                                                   device_map=None, 
                                                                    **model_loading_kwargs)
    classifier.to(device)
    classifier.eval()
    classifier = accelerator.prepare(classifier)

    new_data = {
        'prompt_len': [],
        'prompt_generation_tokenized': [],
        'success_probs': [],
        'rewards': [], 
    }

    i = 0
    for batch in tqdm(dataloader, desc="Loading batches"):
        i += 1
        print(i)
        batch = {k: v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = classifier(
                batch['tokenized_inputs'], 
                batch['attention_masks']
            )
        new_data['prompt_len'] += batch['prompts_len'].tolist()
        new_data['prompt_generation_tokenized'] += [row.cpu().numpy().tolist() 
                                                    for row in batch['tokenized_inputs']]
    
        batch_step_level_probs = post_processing_batch(batch, outputs)
        new_data['success_probs'] += batch_step_level_probs #list of list of floats
        new_data['rewards'] += batch['rewards'].cpu().numpy().tolist()
    
    return new_data

if __name__ == "__main__":

    #login(token="your_hf_token_here")
    
    accelerator = Accelerator()
    device = accelerator.device
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed_data = generate_values(
        data_path = "wen-sun/openr1-clean-DeepSeek-R1-Distill-Qwen-7B-generations",
        value_model_path = "VGS-AI/DeepSeek-VM-1.5B",
        tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        batch_size = 4, 
        max_gen_len = 4096*4,
        accelerator = accelerator, 
        device = device
    )
     
    
    rank = accelerator.process_index
    print(f"Rank {rank} of {accelerator.num_processes}")
    print(len(parsed_data))
    print(len(parsed_data["prompt_len"]))   


    #repo_id = "wen-sun/openr1_token_wise_values_multi_rank{}".format(rank)
    #split_name = f"train_rank{rank}"
    #HF_dataset = Dataset.from_dict(parsed_data)
    #HF_dataset.push_to_hub(repo_id)


    gathered_data = defaultdict(list)
    for key in parsed_data:
        all_values = accelerator.gather_for_metrics(parsed_data[key])
        if accelerator.is_main_process:
            gathered_data[key].extend(all_values)

    if accelerator.is_main_process:
        #features = Features({
        #    "prompt_len": Value("int32"),
        #    "prompt_generation_tokenized": Sequence(Value("int32")),
        #    "success_probs": Sequence(Value("float32")),
        #    "rewards": Value("float32")
            # Add more fields if needed
        #})
        HF_dataset = Dataset.from_dict(gathered_data)
        #print(HF_dataset.features)
        HF_dataset.push_to_hub("wen-sun/openr1_token_wise_values_test")
    
    #if accelerator.is_main_process:
    #    HF_dataset = Dataset.from_dict(parsed_data)
    #    HF_dataset.push_to_hub("wen-sun/openr1_token_wise_values")
    










