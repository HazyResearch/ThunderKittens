"""
Classes for loading pretrained models
"""
from omegaconf import OmegaConf

import torch
import torch.nn as nn

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def get_pretrained_loader(pretrained_model_name_or_path: str,
                          huggingface_token: str = None,
                          **model_kwargs: any):
    """
    Return the appropriate loader for the pretrained model
    """
    if 'qwen' in pretrained_model_name_or_path.lower(): 
        return PretrainedQwenLoader(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            huggingface_token=huggingface_token,
            **model_kwargs,
        )
    else:
        print(f'-> {pretrained_model_name_or_path} using default pretrained model loader')
        return PretrainedModelLoader(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            huggingface_token=huggingface_token,
            **model_kwargs,
        )


class PretrainedModelLoader():
    """
    Class for loading a pretrained model. 
    Example:
      model_loader = PretrainedModelLoader(**model_kwargs)
      model = model_loader.load()
    """
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 cache_dir: str = None,
                 return_dict: bool = True,  # False
                 device_map: str = 'auto',
                 low_cpu_mem_usage: bool = True,
                 torch_dtype: str = 'bfloat16',
                 rope_theta: float = 10000.,
                 attn_implementation: str = 'sdpa',  # eager
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 huggingface_token: str = None,
                 peft_id: str = None,
                 rope_scaling: dict = None,
                 **other_kwargs: any) -> None:

        print(f'-> Using {attn_implementation} attention')
        
        self.loading_kwargs = {
            'pretrained_model_name_or_path': pretrained_model_name_or_path,
            'cache_dir': cache_dir,
            'return_dict': return_dict,
            'load_in_8bit': load_in_8bit,
            'load_in_4bit': load_in_4bit,
            'device_map': device_map,
            'low_cpu_mem_usage': low_cpu_mem_usage,
            'torch_dtype': getattr(torch, torch_dtype),
            'rope_theta': rope_theta,
            'attn_implementation': attn_implementation,
        }
        if rope_scaling is not None:  # Llama 3.1 patch
            rope_scaling = OmegaConf.to_container(rope_scaling)
            self.loading_kwargs['rope_scaling'] = rope_scaling
        for k, v in other_kwargs.items():
            self.loading_kwargs[k] = v

        self.quantization = load_in_8bit or load_in_4bit
        self.peft_id = peft_id
        self.gradient_checkpointing = False
        if huggingface_token is not None:  # for gated models, e.g., Llama 3
            self.loading_kwargs['token'] = huggingface_token

        if self.quantization:
            raise NotImplementedError('Quantization not supported yet')
        
    def load(self) -> nn.Module:
        """
        Load pretrained model
        """
        model = AutoModelForCausalLM.from_pretrained(**self.loading_kwargs)
        if self.quantization:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.gradient_checkpointing,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
        return model

    def load_tokenizer(self):
        """
        Load pretrained tokenizer
        """
        try:
            return AutoTokenizer.from_pretrained(**self.loading_kwargs)
        except Exception as e:
            print("-> Error with `AutoTokenizer.from_pretrained(**self.loading_kwargs)`:", e)
            print("-> Trying `LlamaTokenizer.from_pretrained(**self.loading_kwargs)`")
            # MZ 6/1: Mistral-7B-Instruct-v0.3 in Transformers v4.36 doesn't work with the above
            return LlamaTokenizer.from_pretrained(**self.loading_kwargs)  


class PretrainedQwenLoader(PretrainedModelLoader):
    def load(self, model_type: str = 'flash_attention_2'):
        assert model_type in ['flash_attention_2', 'tk_attention', 'eager']
        self.loading_kwargs['attn_implementation'] = model_type
        from .transformers_modeling_qwen import Qwen2ForCausalLM as model_class
        print('-> Loading from Qwen2ForCausalLM')

        model = model_class.from_pretrained(**self.loading_kwargs)
            
        if self.quantization:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.gradient_checkpointing,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
        return model

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(**self.loading_kwargs)


