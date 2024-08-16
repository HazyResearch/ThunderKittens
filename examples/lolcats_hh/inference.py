import fire
import os
import sys
import time
import gradio as gr

import torch
from transformers import AutoTokenizer, AutoConfig

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_peft_model

from accelerate.utils import is_xpu_available

from custom_llama_model import CustomLlamaForCausalLM
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig

import matplotlib.pyplot as plt

def load_custom_llama_model(model_name, quantization, use_fast_kernels):
    config = AutoConfig.from_pretrained(model_name)
    model = CustomLlamaForCausalLM.from_pretrained(model_name, config=config, device_map="auto",
            low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    
    # if quantization:
    #     model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    if use_fast_kernels:
        print("Using fast kernels")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if use_fast_kernels else None,
            torch_dtype=torch.bfloat16
        )
    
    return model

def main(
    model_name,
    peft_model: str = None,
    quantization: bool = False,
    max_new_tokens = 20,
    prompt_file: str = None,
    seed: int = 42,
    do_sample: bool = True,
    min_length: int = None,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
    enable_azure_content_safety: bool = False,
    enable_sensitive_topics: bool = False,
    enable_salesforce_content_safety: bool = True,
    enable_llamaguard_content_safety: bool = False,
    max_padding_length: int = None,
    use_fast_kernels: bool = False,
    **kwargs
):
    def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, batch_size, use_fa, **kwargs):
        safety_checker = get_safety_checker(
            enable_azure_content_safety,
            enable_sensitive_topics,
            enable_salesforce_content_safety,
            enable_llamaguard_content_safety
        )

        safety_results = [check(user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User prompt deemed safe.")
            # print(f"User prompt:\n{user_prompt}")
        else:
            print("User prompt deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print("Skipping the inference as the prompt is not safe.")
            sys.exit(1)  # Exit the program with an error status

        if is_xpu_available():
            torch.xpu.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        model = load_custom_llama_model(model_name, quantization, use_fa)
        if peft_model:
            model = load_peft_model(model, peft_model)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        user_prompt = [str(user_prompt)] * batch_size
        
        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt", is_split_into_words=False)
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"The inference time is {e2e_inference_time} ms: batch size = {batch_size} and sequence length = {len(user_prompt[0])}")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        safety_results = [check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User input and model output deemed safe.")
            # print(f"Model output:\n{output_text}")
        else:
            print("Model output deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
        
        return e2e_inference_time, len(user_prompt[0])
        # return output_text
    
    user_prompt = "In the year 2045, humanity has achieved remarkable technological advancements. Flying cars zoom through the skies of megacities, while underwater colonies thrive in the depths of the oceans. Artificial intelligence has become an integral part of daily life, assisting in everything from healthcare to space exploration. Sarah, a brilliant neuroscientist, has been working on a groundbreaking project to merge human consciousness with AI. Her goal is to create a symbiotic relationship between organic brains and artificial neural networks, potentially unlocking unprecedented cognitive abilities and extending human lifespan. As Sarah prepares to present her findings at the World Science Summit, she encounters an ethical dilemma. Her research has revealed unforeseen consequences that could fundamentally alter the course of human evolution. She must decide whether to proceed with her work or suppress her discoveries for the greater good. Meanwhile, on Mars, the first human colony is facing a crisis. A mysterious illness is spreading among the settlers, and communication with Earth has been disrupted by an intense solar storm. The colony's leader, Commander Chen, must make difficult decisions to ensure the survival of his team and the future of Mars exploration. Back on Earth, deep in the Amazon rainforest, a team of environmentalists has made an astounding discovery. They've found a previously unknown species of plant with extraordinary properties. Initial tests suggest it could revolutionize medicine and potentially solve the global energy crisis. However, they soon realize that harvesting the plant could disrupt the delicate ecosystem and potentially lead to unforeseen ecological disasters. As these events unfold across the solar system, a young journalist named Alex embarks on a dangerous investigation. They've uncovered evidence of a secretive organization that seems to be manipulating global events from the shadows. Alex's pursuit of the truth will take them from the neon-lit streets of Tokyo to the hidden bunkers beneath the Antarctic ice. In this complex web of scientific breakthroughs, ethical challenges, and hidden agendas, the fate of humanity hangs in the balance. The decisions made by Sarah, Commander Chen, the environmentalists, and Alex will shape the future of our species and our place in the universe. As the world stands on the brink of a new era,"
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_lengths = [64, 128, 256, 512, 1024]
    
    flash_batch_size_results = {bs: [] for bs in batch_sizes}
    flash_seq_length_results = {sl: [] for sl in seq_lengths}
    tk_batch_size_results = {bs: [] for bs in batch_sizes}
    tk_seq_length_results = {sl: [] for sl in seq_lengths}
    
    # version without TK (using flash_attention)
    for bs in batch_sizes:
        for sl in seq_lengths:
            inference_time, actual_seq_length = inference(user_prompt[:sl], temperature, top_p, top_k, max_new_tokens, bs, True)
            flash_batch_size_results[bs].append(inference_time)
            flash_seq_length_results[sl].append(inference_time)
    
    for bs in batch_sizes:
        for sl in seq_lengths:
            inference_time, actual_seq_length = inference(user_prompt[:sl], temperature, top_p, top_k, max_new_tokens, bs, False)
            tk_batch_size_results[bs].append(inference_time)
            tk_seq_length_results[sl].append(inference_time)
    
    # Plotting
    plt.figure(figsize=(20, 10))

    # Batch size plot
    plt.subplot(1, 2, 1)
    for bs in batch_sizes:
        plt.plot(seq_lengths, flash_batch_size_results[bs], marker='o', linestyle='-', label=f'Flash Attn BS {bs}')
        plt.plot(seq_lengths, tk_batch_size_results[bs], marker='s', linestyle='--', label=f'TK BS {bs}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time vs Sequence Length')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    
    # Sequence length plot
    plt.subplot(1, 2, 2)
    for sl in seq_lengths:
        plt.plot(batch_sizes, flash_seq_length_results[sl], marker='o', linestyle='-', label=f'Flash Attn SL {sl}')
        plt.plot(batch_sizes, tk_seq_length_results[sl], marker='s', linestyle='--', label=f'TK SL {sl}')
    plt.xlabel('Batch Size')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time vs Batch Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('inference_time_comparison_plots.png', bbox_inches='tight')
    plt.close()

    print("Plots have been saved as 'inference_time_comparison_plots.png'")

    # if prompt_file is not None:
    #     assert os.path.exists(prompt_file), f"Provided Prompt file does not exist {prompt_file}"
    #     with open(prompt_file, "r") as f:
    #         user_prompt = "\n".join(f.readlines())
    #     inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
    # elif not sys.stdin.isatty():
    #     user_prompt = "\n".join(sys.stdin.readlines())
    #     inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
    # else:
    #     gr.Interface(
    #         fn=inference,
    #         inputs=[
    #             gr.components.Textbox(lines=9, label="User Prompt", placeholder="none"),
    #             gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature"),
    #             gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Top p"),
    #             gr.components.Slider(minimum=0, maximum=100, step=1, value=50, label="Top k"),
    #             gr.components.Slider(minimum=1, maximum=2000, step=1, value=200, label="Max tokens"),
    #         ],
    #         outputs=[
    #             gr.components.Textbox(lines=5, label="Output"),
    #         ],
    #         title="Meta Llama3 Playground",
    #         description="https://github.com/facebookresearch/llama-recipes",
    #     ).queue().launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    fire.Fire(main)