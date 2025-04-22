import torch


instr_opcode = 4     # Opcode for the instruction
num_sms = 148        # Number of SMs in the GPU
bits_per_instr = 32  # 32-bit integers
num_work_units = 128 # # 2048x2048 matrix - the 2048 output units chopped into units of size 16 gives 128 work units


def make_instructions(DEPTH=1):
    arr = [[] for _ in range(num_sms)]

    instruction_idx = 0

    # my instruction
    for i in range(num_work_units):  # ceil
        arr[instruction_idx % num_sms].append([instr_opcode, 0, 16 * i] + [0] * 29)
        instruction_idx += 1

    # pad to 0 to pad the queues so they're evenly sized
    while instruction_idx % num_sms != 0:
        arr[instruction_idx % num_sms].append([0] * bits_per_instr)
        instruction_idx += 1

    instructions = torch.tensor(arr, dtype=torch.int32)
    instructions = instructions.repeat(1, DEPTH, 1)

    timings = torch.zeros(
        (num_sms, (instruction_idx // num_sms) * DEPTH, num_work_units), dtype=torch.int32
    )

    return instructions.to(0), timings.to(0)
