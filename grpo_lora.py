# CUDA_LAUNCH_BLOCKING=1 PYTHONUNBUFFERED=1 accelerate launch --num-processes=4 --gpu_ids 4,5,6,9 grpo_lora.py

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.utils import logging
import torch
logging.set_verbosity_error()
model_size = os.environ.get('MODEL_SIZE', '3')
model_name = f'Qwen2.5-VL-{model_size}B-Instruct'
model_path = os.environ.get('MODEL_PATH', f'Qwen/{model_name}')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16)
if 'CUDA_VISIBLE_DEVICES' in os.environ and ',' not in os.environ['CUDA_VISIBLE_DEVICES']:
    model.to(f'cuda:0') # 0 refers to the index in CUDA_VISIBLE_DEVICES, not in the real GPUs!!!
processor = AutoProcessor.from_pretrained(model_path,
    max_pixels=512*512,
    min_pixels=28*28)

answer_begin = '{'
answer_end = '}'

reasoning_system_prompt = (
    f'You are given an image and a question; answer the question based on the visual evidences in the image.\n'
    f'Your answer will be rated, and the answer must include multiple parts. '
    f'If any part in your answer is missing, you will lose all scores.\n'
    f'Your answer should include the following parts:\n'
    f'1. Describe the objects that should be attented to in the question.\n'
    f'2. Explicitly describe the relevant visual cues from the image in detail.\n'
    f'3. Explain how these evidences relates to the question.\n'
    f'4. Derive the answer based on the visual reasoning.\n'
    f'5. Give the final answer to the question: {answer_begin}final answer{answer_end}\n\n'
    f'Note:\n'
    f'- {answer_begin}...{answer_end} is typically short, so do not include extra text in it.\n'
    f'- Your whole response should be like:\n1. ...\n2. ...\n3. ...\n4. ...\n5. '
    f'{answer_begin}final answer{answer_end}\n'
    f'- The question only describes what the final answer should be like; it doesn\'t restrict the whole response.\n'
    f'- You must rigorously include all the 5 parts.'
)

processor.chat_template = processor.chat_template.replace('You are a helpful assistant.',
    reasoning_system_prompt)
assert reasoning_system_prompt in processor.chat_template

def load_dataset():
    from datasets import load_from_disk
    dataset = load_from_disk('./full_data_aug')
    dataset = dataset.rename_column('completion', 'answer') # because "completions" are for model output
    splitted_datasets = dataset.train_test_split(test_size=100, seed=42)
    train_dataset = splitted_datasets['train']
    eval_dataset = splitted_datasets['test']
    return train_dataset, eval_dataset

# === BEGIN REWARD ZONE ===
def return_int(func):
    def newfunc(*args, **kwargs):
        return int(func(*args, **kwargs))
    return newfunc

@return_int
def is_final_answer_correct(rawans: str, rawgt: str):
    ans = rawans.lower().strip()
    gt = rawgt.lower().strip()
    is_number_like = lambda x: x.isdigit() or x in '.-/'
    if all(is_number_like(x) for x in gt):
        # extract all digits in ans
        digitans = ''.join(x for x in ans if is_number_like(x))
        return digitans == gt
    else:
        neatans = ans.lstrip('{').rstrip('}')
        return neatans == gt
    
@return_int
def contain_string_reward(response: str, string: str):
    return string in response

def compute_reward(response: str, rawgt: str):
    rwd, maxrwd = 0, 0 # actual received reward, "100%" reward
    
    # 1. match strings
    strings_to_match = ['1. ', '2. ', '3. ', '4. ', '5. ', answer_begin, answer_end]
    string_match_max_reward = 1
    for string in strings_to_match:
        rwd += string_match_max_reward * contain_string_reward(response, string)
        maxrwd += string_match_max_reward

    # 2. extract final answer
    start_idx = response.rfind(answer_begin)
    left_idx = start_idx + len(answer_begin)
    right_idx = response.rfind(answer_end)

    rawans = ''

    # 3. match final answer
    final_answer_max_reward = 3 * maxrwd # occupies at least 75% of the reward
    if start_idx >= 0 and right_idx >= 0 and (right_idx >= left_idx):
        rawans = response[left_idx:right_idx]
        rwd += final_answer_max_reward * is_final_answer_correct(rawans, rawgt)
    maxrwd += final_answer_max_reward
    
    return rwd / maxrwd, rawans


def reward_fn(**kwargs):
    answers = kwargs['answer']
    completions = kwargs['completions']
    prompts = kwargs['prompts']
    reward_list = []
    for answer, completion, prompt in zip(answers, completions, prompts):
        rawgt = answer[0]['content']
        response = completion[0]['content']
        reward, rawans = compute_reward(response, rawgt)
        reward_list.append(reward)
    return reward_list

# === END REWARD ZONE ===


do_train = True
if __name__ == '__main__' and do_train:
    from trl import GRPOTrainer, GRPOConfig
    train_dataset, eval_dataset = load_dataset()

    num_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    num_gen = 8
    # batch_size_per_device = num_gen // num_devices # ?
    batch_size_per_device = int(os.environ.get('BATCH_SIZE_PER_DEVICE', 8))
    all_devices_batch_size = num_devices * batch_size_per_device
    global_batch_size = 64
    grad_accum_steps = global_batch_size // all_devices_batch_size

    global_eval_steps = 400 # eval per "sample" # ?
    eval_steps = global_eval_steps // all_devices_batch_size

    script_dir = os.path.dirname(os.path.abspath(__file__))

    grpoconfig = GRPOConfig(
        # TrainingArguments
        output_dir=os.path.join(script_dir, f'outputs/grpo_lora/{model_path.rstrip("/").split("/")[-1]}/try2'), # ?
        overwrite_output_dir=True, # ?
        eval_strategy='steps',
        eval_steps=eval_steps,
        per_device_train_batch_size=batch_size_per_device,
        per_device_eval_batch_size=batch_size_per_device,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=9, # original: 6
        logging_strategy='steps',
        logging_first_step=True,
        logging_steps=1, # ?
        save_strategy='steps',
        save_steps=eval_steps, # (?
        save_total_limit=15, #  (?
        seed=42,
        data_seed=42,
        load_best_model_at_end=False,
        report_to='tensorboard',
        gradient_checkpointing=False, # because we're doing LoRA
        eval_on_start=False, # !!!!!
        remove_unused_columns=False,
        # eos_token=processor.tokenizer.eos_token, # no need for rl (?)
        # pad_token=processor.tokenizer.pad_token,
        learning_rate=1e-5, # ?
        lr_scheduler_type='constant',
        warmup_steps=100, # ?

        # # SFTConfig
        # max_length=None,
        # completion_only_loss=True,
        # assistant_only_loss=False, # not supported for VLMs

        # GRPOConfig
        temperature=1.0, # original: 0.8
        top_p=0.9,
        num_generations=num_gen,
        max_prompt_length=None,
        max_completion_length=1024, # newly added

        log_completions=True,
    )

    from peft import LoraConfig, TaskType

    # this time we perform a thorough fine-tuning
    loraconfig = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules='model.language_model.*(q_proj|k_proj|v_proj|proj|qkv|gate_proj|down_proj|up_proj)',
        task_type=TaskType.CAUSAL_LM,
        bias='none',
    )

    if grpoconfig.local_rank == 0:
        os.makedirs(grpoconfig.output_dir, exist_ok=True)
        import json
        with open(os.path.join(grpoconfig.output_dir, 'config.json'), 'w') as f:
            json.dump(grpoconfig.to_dict(), f, indent=4)
        if '__file__' in globals():
            cur_file = globals()['__file__']
            # copy this file
            import shutil
            shutil.copy(cur_file, os.path.join(grpoconfig.output_dir,
                os.path.basename(cur_file)))
            filedir = os.path.dirname(cur_file)
            modelingfile = os.path.join(filedir, 'modeling_qwen2_5_vl.py')
            if os.path.exists(modelingfile):
                shutil.copy(modelingfile, os.path.join(grpoconfig.output_dir,
                    os.path.basename(modelingfile)))
            configfile = os.path.join(filedir, 'configuration_qwen2_5_vl.py')
            if os.path.exists(configfile):
                shutil.copy(configfile, os.path.join(grpoconfig.output_dir,
                    os.path.basename(configfile)))

    trainer = GRPOTrainer(
        model=model,
        args=grpoconfig,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_fn, # partial_reward_fn,
        peft_config=loraconfig,
    )

    resume_from_checkpoint = False

    if grpoconfig.local_rank == 0:
        import sys
        sys.stdout = open(os.path.join(grpoconfig.output_dir, 'training_details_stdout.log'),
            'wa'[int(resume_from_checkpoint)], buffering=1)
        sys.stderr = open(os.path.join(grpoconfig.output_dir, 'training_details_stderr.log'),
            'wa'[int(resume_from_checkpoint)], buffering=1)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if grpoconfig.local_rank == 0:
        sys.stdout.close()
        sys.stderr.close()
