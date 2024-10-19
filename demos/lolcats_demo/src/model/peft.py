"""
Helpers for parameter-efficient finetuning via low-rank adapters (LoRA)
-> Mainly follow PEFT / llama recipes

Right now quantization not super tested
"""
import torch
from torch.nn import Module


# Modified from https://github.com/facebookresearch/llama-recipes/blob/main/examples/quickstart.ipynb
def create_peft_config(model: Module, 
                       peft_config: dict, 
                       target_dtype: str = 'bfloat16',
                       preserve_requires_grad: bool = False,
                       use_gradient_checkpointing: bool = None,
                       add_self_attn_prefix: bool = True):
    """
    Create a parameter-efficient finetuning model (e.g., attaching LoRAs)
    -> Assumes that all non-trainable weights have been frozen already.
       If not, freeze them before calling this function.
    """
    if peft_config['method'] == 'lora':
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            prepare_model_for_kbit_training,
        )
        try:
            target_modules = []  # hack to only do self_attn terms
            for module_name in peft_config['kwargs']['target_modules']:
                if ('_proj' in module_name and 'self_attn' not in module_name 
                    and add_self_attn_prefix):
                    target_modules.append(f'self_attn.{module_name}')
                elif '_proj' in module_name:
                    target_modules.append(module_name)
            peft_config['kwargs']['target_modules'] = target_modules
        except Exception as e:
            print(e)
            target_modules = []

        if 'layers_to_ignore' in peft_config:
            peft_config['kwargs']['layers_to_transform'] = [
                i for i in range(len(model.model.layers))
                if i not in peft_config['layers_to_ignore']
            ]
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_config['kwargs'],
        )
        # Save parameters that did not have frozen weights before to unfreeze later
        trainable_weights = [
            n for n, p in model.named_parameters() if p.requires_grad 
        ]
        # Prepare int-8 or int-4 model for training
        loaded_in_kbit = (getattr(model, "is_loaded_in_8bit", False) or 
                          getattr(model, "is_loaded_in_4bit", False))
        if loaded_in_kbit:  # From https://huggingface.co/docs/peft/en/package_reference/peft_model:
            # This method wraps the entire protocol for preparing a model before running a training. 
            # 1- Cast the layernorm in fp32 
            # 2- making output embedding layer require grads 
            # 3- Add the upcasting of the lm head to fp32
            model.enable_input_require_grads()
            ugc = (use_gradient_checkpointing 
                   if use_gradient_checkpointing is not None else True)
            print('-> use_gradient_checkpointing:', ugc)
            # model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=ugc,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        for n, p in model.named_parameters():
            # Unfreeze weights frozen by get_peft_model()
            if preserve_requires_grad:
                if n[len('base_model.model.'):] in trainable_weights:
                    p.requires_grad = True

            # prepare_model_for_kbit_training will cast all non INT8 parameters to fp32
            # -> https://github.com/huggingface/peft/blob/7e84dec20b3106bdd0a90ba8e80187f0aec835b7/src/peft/utils/other.py#L103
            # So we'll cast these back to their prior dtype
            if p.requires_grad and loaded_in_kbit:
                p.data = p.data.to(getattr(torch, target_dtype))

        if not loaded_in_kbit:
            model.to(dtype=getattr(torch, target_dtype))

        return model, peft_config
    else:
        raise NotImplementedError(f"Sorry PEFT method {peft_config['method']} not implemented yet.")
