"""
Quick demo of linearized LLM generations
"""
from typing import Optional, List
import time
import argparse
import torch

from omegaconf import OmegaConf

from transformers import TextStreamer, TextIteratorStreamer, AutoTokenizer

from src.utils.setup import seed_everything
from src.utils.logging import print_header
from src.model.pretrained import get_pretrained_loader


system_prompt = """{prompt}"""


def get_args():
    parser = argparse.ArgumentParser()
    # Model load + setup
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--config_dir", type=str, default='configs')
    parser.add_argument("--seed", type=int, default=42)

    # Inference
    parser.add_argument("--model_type", type=str, default='flash_attention_2')

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

def load_model_from_checkpoint(model_type: str,
                               model_config_path: str = None,
                               huggingface_token: str = None):

    print(f"Using model type: {model_type}")

    model_config = OmegaConf.load(model_config_path)

    # Load initial transformer model
    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = model_loader.load(model_type = model_type)
    return model, model_config, tokenizer

def main():
    args = get_args()
    seed_everything(args.seed)
    model, model_config, tokenizer = load_model_from_checkpoint(
        args.model_type,
        args.model_config_path, 
        args.huggingface_token,
    )
    model = model.to('cuda')
    model.eval()
    input_len = len(tokenizer(system_prompt)['input_ids'])

    while True:
        print(f'\n>> Generating {args.num_generations} responses in parallel')
        prompt = input(f'>> Your prompt: (or cmd-c to quit)... ')

        # prompt hard coded to 192; no padding.
        # prompt = "Create a summary of the following passage: London is the capital city of England and the United Kingdom. It is a leading global city with strengths in the arts, commerce, education, entertainment, fashion, finance, healthcare, media, professional services, research and development, tourism, and transport all contributing to its prominence. It is one of the most populous cities in the world, with an estimated population of 8.9 million in 2019. At its centre stand the imposing Houses of Parliament, the iconic ‘Big Ben’ clock tower and Westminster Abbey, site of British monarch coronations. Across the Thames River, the London Eye observation wheel provides panoramic views of the South Bank cultural complex, and the entire city. London exerts a strong influence on world art,"
        #  entertainment, fashion, commerce, finance, education, healthcare, media, science, technology, tourism, transport, and communications. London's cultures encompass over 300 languages. The last time"

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
                
        if args.benchmark:
            break

if __name__ == '__main__':
    main()


