
import torch
import time
from transformers import AutoTokenizer

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load pretrained models
print("\nLoading pretrained models...")
from train.src.models.gpt import GPTLMHeadModel as BasedGPTLMHeadModel
from based.models.mamba import MambaLMHeadModel
from based.models.transformer.gpt import GPTLMHeadModel
based_model = BasedGPTLMHeadModel.from_pretrained_hf(
    "hazyresearch/based-360m", 
    device="cuda", 
    implementation='default',           # choices are [default, tk]
    swa_inference_mode = "fast_rotary", # choices [default, default_rotary, fast_rotary]
    silent=True           # will print more info during inference if set to False
).to(torch.bfloat16)
# mamba_model = MambaLMHeadModel.from_pretrained_hf("hazyresearch/mamba-360m").to("cuda").to(torch.bfloat16)
# mamba_2_model = MambaLMHeadModel.from_pretrained_hf("state-spaces/mamba2-370m").to("cuda").to(torch.float16)
# attn_model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/attn-360m").to("cuda").to(torch.bfloat16)

# Inputs
sample_inputs = [ 
    # "The capital of California is Sacramento. The capital of Italy is Rome. The capital of France is Paris and capital of New York is New York. The capital of Austria is Vienna. The capital of Greece is Athens. The capital of",
    # "There are fifty different,",
    # "The capital of California is Sacramento. The capital of Italy",
    # "After going to the movies,",
    # "1, 2, 3, 4,",
    # "Let me tell you about the difference between rats and mice. First, rats are",
    "SUBSTANTIAL EQUIVALENCE DETERMINATION DECISION SUMMARY A. 510(k) Number: K153137 B. Purpose for Submission: Clearance of a new device C. Measurand: Anti-PF4/Heparin Total Antibodies D. Type of Test: Automated, latex enhanced immuno-turbidimetric assay E. Applicant: Instrumentation Laboratory (IL) Co. F. Proprietary and Established Names: HemosIL HIT‐Ab(PF4‐H) HemosIL HIT‐Ab(PF4‐H) Controls G. Regulatory Information: 1. Regulation section: 21 CFR 864.7695, Platelet factor 4 radioimmunoassay 21 CFR 864.5425, Multipurpose system for in vitro coagulation studies 2. Classification: Class II 3. Product code: 2 LCO, Platelet factor 4 radioimmunoassay GGN, Plasma, Coagulation Control 4. Panel: Hematology (81) H. Intended Use: 1. Intended use(s): HemosIL HIT-Ab(PF4-H) is a qualitative, fully automated, latex enhanced immunoassay for the detection of anti-platelet factor 4/heparin (PF4/H) antibodies. The assay is for use in human 3.2% or 3.8% citrated plasma on the ACL TOP® Family of instruments in a laboratory setting. The result provided by the assay should be interpreted as either positive or negative based on the assay cut-off (1.0 U/mL). The positive or negative result aids in determining the risk for heparin induced thrombocytopenia (HIT) when used in conjunction with other laboratory and clinical findings. Anti-PF4/Heparin antibodies are commonly found in patients with HIT. For use in adult population suspected of HIT. Not for use in isolation to exclude HIT. HemoslL HIT-Ab(PF4-H) Controls are for the Quality Control of the HemosIL HIT-Ab(PF4- H) assay as performed on the ACL TOP® Family of instruments. For prescription use. 2. Indication(s) for use: Same as Intended Use 3. Classification:",
    "SUBSTANTIAL EQUIVALENCE DETERMINATION DECISION SUMMARY A. 510(k) Number: K153137 B. Purpose for Submission: Clearance of a new device C. Measurand: Anti-PF4/Heparin Total Antibodies D. Type of Test: Automated, latex enhanced immuno-turbidimetric assay E. Applicant: Instrumentation Laboratory (IL) Co. F. Proprietary and Established Names: HemosIL HIT‐Ab(PF4‐H) HemosIL HIT‐Ab(PF4‐H) Controls G. Regulatory Information: 1. Regulation section: 21 CFR 864.7695, Platelet factor 4 radioimmunoassay 21 CFR 864.5425, Multipurpose system for in vitro coagulation studies 2. Classification: Class II 3. Product code: 2 LCO, Platelet factor 4 radioimmunoassay GGN, Plasma, Coagulation Control 4. Panel: Hematology (81) H. Intended Use: 1. Intended use(s): HemosIL HIT-Ab(PF4-H) is a qualitative, fully automated, latex enhanced immunoassay for the detection of anti-platelet factor 4/heparin (PF4/H) antibodies. The assay is for use in human 3.2% or 3.8% citrated plasma on the ACL TOP® Family of instruments in a laboratory setting. The result provided by the assay should be interpreted as either positive or negative based on the assay cut-off (1.0 U/mL). The positive or negative result aids in determining the risk for heparin induced thrombocytopenia (HIT) when used in conjunction with other laboratory and clinical findings. Anti-PF4/Heparin antibodies are commonly found in patients with HIT. For use in adult population suspected of HIT. Not for use in isolation to exclude HIT. HemoslL HIT-Ab(PF4-H) Controls are for the Quality Control of the HemosIL HIT-Ab(PF4- H) assay as performed on the ACL TOP® Family of instruments. For prescription use. 2. Indication(s) for use: Same as Intended Use 3. Applicant:",
]

# Setup tokenizer
context_length, generation_length = 2020, 10
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer_mamba2 = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer_mamba2.padding_side = "left"
tokenizer_mamba2.pad_token = tokenizer.eos_token
tokenizer_mamba2.pad_token_id = tokenizer.eos_token_id

# Generate
print("\n\nStarting generation demo:\n\n")
for input_text in sample_inputs:
    inputs = tokenizer.batch_encode_plus(
        [input_text], return_tensors="pt", padding=True, truncation=True, max_length=context_length
    ).input_ids.to("cuda")
    limit = inputs.shape[-1] + generation_length
    start = inputs.shape[-1]
    print(f"{start=}, {limit=}")

    mamba2_inputs = tokenizer_mamba2.batch_encode_plus(
        [input_text], return_tensors="pt", padding=True, truncation=True, max_length=context_length
    ).input_ids.to("cuda")
    mamba2_limit = mamba2_inputs.shape[-1] + generation_length
    mamba2_start = mamba2_inputs.shape[-1]
    print(f"{mamba2_start=}, {mamba2_limit=}")

    for model_name, model in zip([
        'based', 
        # 'mamba', 'mamba2', 'attn'
        ], [based_model, 
        # mamba_model, mamba_2_model, attn_model
        ]):
        model.eval()
        fn = model.generate
        cur_inputs = mamba2_inputs if 'mamba2' in model_name else inputs
        cur_limit = mamba2_limit if 'mamba2' in model_name else limit

        if 'based' in model_name:
            with torch.no_grad():
                generations = fn(
                    input_ids=cur_inputs,
                    max_length=cur_limit,
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0,
                    implementation="tk"
                )
        else:
            with torch.no_grad():
                generations = fn(
                    input_ids=cur_inputs,
                    max_length=cur_limit,
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0
                )

        cur_start = mamba2_start if 'mamba2' in model_name else start
        preds = generations[:, cur_start:]
        pred_ids =  preds[0].tolist()
        if 'mamba2' in model_name:
            pred = tokenizer_mamba2.decode(pred_ids)
            input_text = tokenizer_mamba2.decode(cur_inputs[0].tolist()) 
        else:
            input_text = tokenizer.decode(cur_inputs[0].tolist()) 
            pred = tokenizer.decode(pred_ids)
        input_text = input_text.replace("\n", " ")
        if len(input_text) > 300:
            # truncate long prompts for inspection 
            input_text = input_text[:150] + " ... [more tokens] ... " + input_text[-150:]
        pred = pred.replace("\n", " ") 

        # after the SWA
        print(f"{model_name=}: {input_text} -> {pred}\n")
    print()
