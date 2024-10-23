"""
Classes for loading pretrained models
"""
from os.path import join
from omegaconf import OmegaConf

import torch
import torch.nn as nn

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
# from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def get_pretrained_loader(pretrained_model_name_or_path: str,
                          huggingface_token: str = None,
                          **model_kwargs: any):
    """
    Return the appropriate loader for the pretrained model
    """

    if 'lama' in pretrained_model_name_or_path:  # Llama or llama
        return PretrainedLlamaLoader(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            huggingface_token=huggingface_token,
            **model_kwargs,
        )
    elif 'istral' in pretrained_model_name_or_path:  # Mistral or mistral; 
        return PretrainedMistralLoader(   
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
            raise NotImplementedError
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=load_in_8bit,
            #     load_in_4bit=load_in_4bit,
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            # )
            # del self.loading_kwargs['load_in_8bit']
            # del self.loading_kwargs['load_in_4bit']
            # self.loading_kwargs['quantization_config'] = bnb_config
        
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


class PretrainedLlamaLoader(PretrainedModelLoader):
    def load(self, model_type: str = None, ):
        llama3_1 = float('.'.join(transformers.__version__.split('.')[:2])) > 4.42  # 'Meta-Llama-3.1' in self.loading_kwargs['pretrained_model_name_or_path']
        if model_type is None:
            from transformers import LlamaForCausalLM as model_class

        elif 'lolcats_llama_sharded' in model_type:
            from .modeling_llama_sharded import ShardedLolcatsLlamaForCausalLM as model_class

        elif 'lolcats_long_llama' in model_type:
            from .modeling_llama import LooooolcatsLlamaForCausalLM as model_class
        
        elif 'lolcats_llama' in model_type:
            from .modeling_llama import LolcatsLlamaForCausalLM as model_class
            
        else:
            if model_type == 'flash_attention_2':
                self.loading_kwargs['attn_implementation'] = model_type
            from transformers import AutoModelForCausalLM as model_class
            print('-> Loading from AutoModelForCausalLM')

        model = model_class.from_pretrained(**self.loading_kwargs)
        if self.peft_id is not None:
            from peft import PeftModel
            print('-> Loading PEFT checkpoint')
            model = PeftModel.from_pretrained(
                model, 
                self.peft_id,
                torch_dtype=self.loading_kwargs['torch_dtype'],
                device_map='auto',
                cache_dir=self.loading_kwargs['cache_dir']
            ).merge_and_unload()
            
        if self.quantization:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.gradient_checkpointing,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
        return model

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(**self.loading_kwargs)


class PretrainedMistralLoader(PretrainedModelLoader):
    def load(self, model_type: str = None):
        if model_type is None:
            from transformers import MistralForCausalLM as model_class
        elif 'lolcats_long_llama' in model_type:
            from .modeling_mistral import LooooolcatsMistralForCausalLM as model_class
        elif 'lolcats_llama' in model_type:
            from .modeling_mistral import LolcatsMistralForCausalLM as model_class
        else:
            if model_type == 'flash_attention_2':
                self.loading_kwargs['attn_implementation'] = model_type
            from transformers import AutoModelForCausalLM as model_class
            print('-> Loading from AutoModelForCausalLM')
            
        model = model_class.from_pretrained(**self.loading_kwargs)
        if self.peft_id is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, 
                self.peft_id,
                torch_dtype=self.loading_kwargs['torch_dtype'],
                device_map='auto',
                cache_dir=self.loading_kwargs['cache_dir'],
            ).merge_and_unload()
            
        if self.quantization:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.gradient_checkpointing,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
        return model
