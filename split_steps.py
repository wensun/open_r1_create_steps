import torch
import numpy as np
from transformers import AutoTokenizer
from accelerate import Accelerator
import classifier_lib
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, Dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from IPython import embed
from huggingface_hub import login


class inputdataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "prompt_len": self.data[idx]['prompt_len'],
            "prompt_generation_ids": self.data[idx]['prompt_generation_tokenized'],
            "success_probs": self.data[idx]['success_probs'],
            "rewards": self.data[idx]['rewards']
        }
    
def compute_vstar(p, beta, reward = None, alpha = 0.99):
    #this function assume 0 / 1 reward, where p is the probability of getting reward 1.
    assert p >= 0 and p<=1
    if reward is None:
        vstar = beta * np.log( p*np.exp(1./ beta) + (1-p))
    else:
        p_posterior = alpha* p + (1-alpha) * reward  # logic: if reward = 1, increase p a bit, else decrease a bit
        assert p_posterior >= 0 and p_posterior <= 1
        vstar = beta * np.log( p_posterior*np.exp(1./ beta) + (1-p_posterior))
    
    return vstar

def input_collate(batch, tokenizer, beta = 0.5, step_len = 4096):
    bs = len(batch)
    eos_token_id = tokenizer.eos_token_id
    prompts_len = [item['prompt_len'] for item in batch]
    prompt_gen_ids = [item['prompt_generation_ids'] for item in batch]
    success_probs = [item['success_probs'] for item in batch]
    rewards = [item['rewards'] for item in batch]

    prompts, steps, advs, rews = [],[],[],[]

    for i in range(bs):
        p_len = prompts_len[i]
        p_gen_ids = prompt_gen_ids[i]  # list of integers
        s_probs = success_probs[i]  # list of floats
        rew = rewards[i]  # integer
        assert len(p_gen_ids) == len(s_probs)

        # exclude pad tokens
        valid_len = torch.sum(torch.tensor(p_gen_ids, dtype=int) != eos_token_id) # max length is 4096*4
        valid_p_gen_ids = p_gen_ids[:valid_len] # got rid of paddings
        valid_s_probs = s_probs[:valid_len]  # got rid of padding part, but note this still includes the prompt
        
        prompt_ids = valid_p_gen_ids[:p_len]  # just the prompt tokens
        gen_ids = valid_p_gen_ids[p_len:]
        gen_len = min(4096*4, valid_len - p_len)
        gen_ids = gen_ids[:gen_len]
        
        print("#######")
        print("p_len {}, gen_len {}".format(p_len, len(gen_ids)))
        print("#########")

        #split gen_ids into steps 
        for j in range(0, len(gen_ids), step_len):
            start_id = j
            end_id = min(j+step_len, len(gen_ids))
            
            new_prompt = prompt_ids + gen_ids[:start_id]
            step = gen_ids[start_id:end_id]
            
            # in default, we try to include the current reward in the v-star calculation. 
            # edge case: when start_id = 0, then the first prob should come from the end token of the prompt, 
            # whose index would be p_len + start_id - 1
            
            #print(p_len)
            print(p_len+start_id-1)
            print(p_len+end_id-1)
            vstar_start = compute_vstar(valid_s_probs[p_len + start_id-1], beta = beta, reward = rew) 
            vstar_end = compute_vstar(valid_s_probs[p_len + end_id-1], beta = beta, reward = rew)
            adv = vstar_end - vstar_start  # compute advantage: V(prompt+step) - V(prompt)

            prompts.append(new_prompt)
            steps.append(step)
            advs.append(adv)
            rews.append(rew)
        
    return {
        "prompts": prompts,  # list of lists
        "generations": steps,  # list of lists
        "advantages": advs, 
        "rewards": rews,
    }



def generate_step_wise_data(
        data_path, 
        tokenizer_name,
        batch_size,
        beta, # float for calculating the vstar
):
    data = load_dataset(data_path)['train']
    print(data[0].keys())
    print("number of rows is {}".format(len(data)))

    embed()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_bos_token = False #do not add bos token when call encode function


    dataset = inputdataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        collate_fn = lambda b: input_collate(b, tokenizer, beta),
    )

    parsed_data = {
        "prompt": [],
        "response": [], 
        "reward": [],
        "vstar": [],
    }

    for batch in tqdm(dataloader, desc="Loading batches"):
        parsed_data['prompt'] += batch['prompts']
        parsed_data['response'] += batch['generations']
        parsed_data['reward'] += batch['rewards']
        parsed_data['vstar'] += batch['advantages']
    
    return parsed_data



if __name__ == "__main__":

    parsed_data = generate_step_wise_data(
        data_path = "wen-sun/test_value_dataset",
        tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        batch_size= 2, 
        beta = 0.5
    )


