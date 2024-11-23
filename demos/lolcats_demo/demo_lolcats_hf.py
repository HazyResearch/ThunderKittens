"""
Quick demo of linearized LLM generations
"""
from typing import Optional, List
from os.path import join
import time
import argparse
import torch

from omegaconf import OmegaConf

from transformers import TextStreamer, TextIteratorStreamer, AutoTokenizer

from src.utils.setup import seed_everything
from src.utils.logging import print_header
from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please pip install huggingface-hub")


system_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request. 
{prompt}"""


def get_args():
    parser = argparse.ArgumentParser()
    # Model load + setup
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--finetune_config_path", type=str)
    parser.add_argument("--distill_config_path", type=str)
    parser.add_argument("--attn_mlp_checkpoint_path", type=str, default=None)
    parser.add_argument("--finetune_checkpoint_path", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default='configs')
    parser.add_argument("--seed", type=int, default=42)

    # Inference
    parser.add_argument("--use_cuda_kernels", type=int, default=0)
    parser.add_argument("--use_attention", action='store_true', default=False)

    # Generation
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2) 

    # Miscellaneous
    parser.add_argument("--benchmark", action='store_true', default=False)
    parser.add_argument("--print_model", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--huggingface_token", type=str, default=None)

    args = parser.parse_args()
    return args


def get_lm_eval_lolcats_model(model_kwargs: dict, lolcats_model: bool = True):
    lm_kwargs = copy.deepcopy(model_kwargs)
    lm_kwargs['pretrained'] = lm_kwargs['pretrained_model_name_or_path']
    lm_kwargs['dtype'] = str(lm_kwargs['torch_dtype']).split('.')[-1]
    del lm_kwargs['torch_dtype']

    if 'Llama' in lm_kwargs['pretrained_model_name_or_path']: 
        lm_kwargs['device_map'] = None
        from lm_eval_harness.models import ShardedLolcatsLlamaForCausalLM
        lm = ShardedLolcatsLlamaForCausalLM.create_from_arg_string(
            '', lm_kwargs,
        )
    else:
        sys.path.append(LM_EVALUATION_HARNESS_PATH)
        from lm_eval.models import get_model
        
        lm = get_model('hf-causal-experimental').create_from_arg_string(
            '', lm_kwargs,
        )
    return lm


class BatchTextIteratorStreamer(TextIteratorStreamer):
    """
    Copied from https://discuss.huggingface.co/t/textiteratorstreamer-compatibility-with-batch-processing/46763/2
    """
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 batch_size: int, 
                 skip_prompt: bool = False, 
                 timeout: Optional[float] = None, 
                 **decode_kwargs: any):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None
        self.go_up = 0 + batch_size
        self.stop_signal = tokenizer.eos_token

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(value, (self.batch_size, value.shape[0] // self.batch_size))

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                self.go_up += 1
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx] :]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx] : text.rfind(" ") + 1]
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

        try:
            text = [
                ''.join([x[i] if i < len(x) else self.stop_signal 
                         for x in self.text_queue.queue ]) 
                for i in range(len(self.text_queue.queue[0]))
            ]
            text = '\n------------\n'.join(text)
            go_up = "\033[F" * self.go_up  # len(text)  # Goes up this many lines
            print(f'{text}', flush=True, end="" if not stream_end else None)

        except Exception as e:
            print(self.stop_signal)

def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def setup_fsdp_config(config, args, checkpoint_name: str = 'finetune'):
    """
    Hacky arguments for llama-recipes training function
    """
    config.seed = args.seed
    config.enable_fsdp = args.enable_fsdp
    config.low_cpu_fsdp = args.low_cpu_fsdp
    config.dist_checkpoint_root_folder = args.checkpoint_dir
    config.dist_checkpoint_folder = checkpoint_name

    config.model_name = args.run_name
    config.use_peft = False  # We have custom logic for saving PEFT modules
    config.save_model = True
    config.run_validation = True
    config.use_fp16 = False
    config.save_model = True
    config.save_optimizer = False
    config.output_dir = args.checkpoint_dir
    config.save_metrics = not args.no_wandb
    config.gradient_clipping = False
    config.gradient_clipping_threshold = 1.0
    config.num_epochs = getattr(config.trainer, 'num_train_epochs', None)
    config.num_train_steps = getattr(args, 'num_train_steps', None)  # exit training loop early for debugging
    config.eval_steps = getattr(config.trainer, 'eval_steps', None)  # how many gradient updates before evaluating
    return config


def load_hf_weights(model, distill_repo_id, ft_repo_id, filename="model.pt"):
    for repo_id in [distill_repo_id, ft_repo_id]:
        if repo_id is None: continue 

        print(f"Loading weights from {repo_id}")

        local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)    
        state_dict = torch.load(local_file_path)
        if 'model_state_dict' in state_dict: 
            state_dict = state_dict['model_state_dict']
        else:
            pass
        _keys = model.load_state_dict(state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'base_model.model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)

        try:
            assert len(_keys.unexpected_keys) == 0
            print_header('*** All expected keys matched successfully ***')
        except Exception as e:
            print(e)
            print_header('*** Error: unexpected keys in checkpoint - please fix ***')
            print('Unexpected keys:')
            for k in _keys.unexpected_keys:
                print(k)
            exit()

    return model


def load_model_from_checkpoint(attn_mlp_checkpoint_path: str = None, 
                               finetune_checkpoint_path: str = None, 
                               model_config_path: str = None,
                               distill_config_path: str = None,
                               finetune_config_path: str = None,
                               config_dir: str = 'configs',
                               print_model: bool = False, 
                               debug: bool = False,
                               huggingface_token: str = None,
                               use_cuda_kernels: bool = False,
                               use_attention: bool = False):

    is_local = attn_mlp_checkpoint_path.endswith(".pt")
    
    rank = 0
    model_config = OmegaConf.load(model_config_path)
    distill_config = OmegaConf.load(distill_config_path)
    finetune_config = OmegaConf.load(finetune_config_path)

    # Load initial transformer model
    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    if use_attention:
        model = model_loader.load('softmax')
        return model, model_config, tokenizer

    model = model_loader.load(model_config['attention']['attention_type'])
    if use_cuda_kernels:
        print('*** Using TK CUDA kernels **')
        model_config['attention']['attention_type'] = 'lolcats_llama_window_tk_gen'

    # Swap the softmax to linear attention 
    model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                        attention_type=None, 
                                                        checkpoint_path=None,
                                                        print_model=debug,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=False,
                                                        train_attention=False)
    
    # Add PEFT parameters
    model, ft_peft_config = load_and_convert_finetune(model, finetune_config,
                                                      checkpoint_path=None,
                                                      print_model=debug,
                                                      merge_loras=False,
                                                      peft_gradient_checkpointing=False)

    # Load from huggingface checkpoints and insert into the model dict 
    model = load_hf_weights(
        model, 
        attn_mlp_checkpoint_path, finetune_checkpoint_path, 
        filename="model.pt"
    )
    if use_cuda_kernels:
        print('*** Using TK CUDA kernels ***')

    if print_model:
        print_header('*** Model after checkpoint load ***')
        print(model)

    return model, model_config, tokenizer


def main():
    args = get_args()
    seed_everything(args.seed)
    model, model_config, tokenizer = load_model_from_checkpoint(
        args.attn_mlp_checkpoint_path, args.finetune_checkpoint_path, 
        args.model_config_path, args.distill_config_path, args.finetune_config_path,
        config_dir=args.config_dir, print_model = args.print_model, debug = args.debug,
        use_cuda_kernels = args.use_cuda_kernels,
        use_attention = args.use_attention,
    )
    model = model.to('cuda')
    model.eval()
    input_len = len(tokenizer(system_prompt)['input_ids'])

    while True:
        print(f'\n>> Generating {args.num_generations} responses in parallel')
        prompt = input(f'>> Your prompt: (or cmd-c to quit)... ')
        # Sample prompt
        # prompt = "List the capitals of the United states in alphabetical order.\n"

        all_prompts = [system_prompt.format(prompt=prompt)] * args.num_generations

        
        if args.num_generations == 1:
            streamer = TextStreamer(tokenizer, skip_prompt=True,
                                    decode_kwargs={'skip_special_tokens': True})
        else:
            streamer = BatchTextIteratorStreamer(tokenizer=tokenizer, 
                                                 batch_size=args.num_generations,
                                                 skip_prompt=True,)
    
        with torch.no_grad():
            model_input = tokenizer(all_prompts, return_tensors="pt").to(model.device)
            print(model_input['input_ids'].shape)

            # pad the prompt with zeros to make it a multiple of 64
            if model_input['input_ids'].shape[1] % 64 != 0:
                model_input['input_ids'] = torch.cat([ 
                                                      torch.zeros((model_input['input_ids'].shape[0], 
                                                                   64 - model_input['input_ids'].shape[1] % 64), 
                                                                  dtype=model_input['input_ids'].dtype, 
                                                                  device=model_input['input_ids'].device),
                                                      model_input['input_ids']], 
                                                     dim=1)
                model_input['attention_mask'] = torch.cat([
                                                            torch.zeros((model_input['attention_mask'].shape[0], 
                                                                         64 - model_input['attention_mask'].shape[1] % 64), 
                                                                        dtype=model_input['attention_mask'].dtype, 
                                                                        device=model_input['attention_mask'].device),
                                                            model_input['attention_mask']], 
                                                             dim=1)
            # print(model_input['input_ids'])
            # assert model_input['input_ids'].shape[1] % 64 == 0, 'Prompt length mismatch' 

            if args.benchmark:
                torch.cuda.synchronize()
                start_time = time.time() 
            model_output = model.generate(**model_input, use_cache=True, 
                                          max_new_tokens=args.max_new_tokens, 
                                          do_sample=False,
                                          top_k=args.top_k,
                                          top_p=args.top_p,
                                          num_return_sequences=1,
                                          pad_token_id=tokenizer.eos_token_id,
                                          streamer=streamer)
            if args.benchmark:
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                total_tokens = (model_output != tokenizer.eos_token_id).sum().item()
                print_header('(Coarse) stats for nerds')
                print(f'├── Model data type:                      {model.dtype}')
                print(f'├── Time of longest response:             {elapsed:.3f} sec')
                print(f'├── Total tokens processed + generated:   {total_tokens}')
                print(f'├── Throughput (lagged by last response): {total_tokens / elapsed:.3f} tokens/sec')
                
        # if args.benchmark:
        #     break

if __name__ == '__main__':
    main()


