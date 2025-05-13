import torch
import mlp
from timings import save_gantt_chart

true_num_sms = 132
num_sms = 128

intermediate_dim = 4096

up = torch.randn(intermediate_dim, 2048, dtype=torch.bfloat16, device=0) / 2048**.25
down = torch.randn(2048, intermediate_dim, dtype=torch.bfloat16, device=0) / intermediate_dim**.25

upcopy = up.clone()
downcopy = down.clone()

vec = torch.randn(2048, dtype=torch.bfloat16, device=0) / 2048**.5

ref = torch.relu(vec @ up.T) @ down.T

def make_schedule():
    instructions = []
    up_work = list(range(intermediate_dim//16))
    for j in range(num_sms):
        work = up_work[j::num_sms]
        up_instruction = [1, len(work)] + work
        up_instruction += [0] * (32 - len(up_instruction))
        instructions.append(up_instruction)
    # for i in range(num_sms):
    #     down_instruction = [2, i, i+1, 0] + [0]*28
    #     instructions.append(down_instruction)
    num_down_stages = intermediate_dim//2048
    for j in range(0,num_down_stages):
        num_j_sms = num_sms*(j+1)//num_down_stages - num_sms*j//num_down_stages
        print('num_j_sms', num_j_sms)
        for i in range(num_j_sms):
            start, end = (2048//16)*i//num_j_sms, (2048//16)*(i+1)//num_j_sms
            down_instruction = [2, start, end, j] + [0]*28
            instructions.append(down_instruction)
    while len(instructions) % num_sms != 0:
        instructions.append([0] * 32)
    instructions = torch.tensor(instructions, dtype=torch.int32, device=0).reshape(-1,num_sms,32).transpose(0,1)
    instructions = torch.cat([instructions, torch.zeros((true_num_sms-num_sms, instructions.shape[1], instructions.shape[2]), device=0, dtype=torch.int32)], dim=0)
    timings = torch.zeros((instructions.shape[0], instructions.shape[1], 128), device=0, dtype=torch.int32)
    barriers = torch.zeros(4, device=0, dtype=torch.int32)
    return instructions, timings, barriers

ITERS = 1

instructions, timings, barriers = make_schedule()
many_barriers = torch.stack([barriers]*ITERS)
many_timings = torch.stack([timings]*ITERS)
print(instructions.shape)

inputs = vec.to(torch.float32)
hidden = torch.zeros(intermediate_dim, device=0, dtype=torch.float32)
outputs = torch.zeros(2048, device=0, dtype=torch.float32)

mlp.mlp_1stage(barriers, instructions, timings, upcopy, downcopy, inputs, hidden, outputs)

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# start_event.record()
# for i in range(ITERS):
#     mlp.mlp_1stage(many_barriers[i], instructions, many_timings[i], up, down, inputs, hidden, outputs)
# end_event.record()
# torch.cuda.synchronize()
# elapsed_time = start_event.elapsed_time(end_event)
# print(f"Execution time: {elapsed_time/ITERS*1000:.2f} us")

# print(timings[0])
# print(timings[127])

print(timings[:128,:,4].min(), timings[:128,:,4].max())

print('\n-------\n\n-------\n')

save_gantt_chart(timings, instructions, save_all=True, name='1stage', clock_rate=1000)

many_barriers.zero_()
many_timings.zero_()
outputs.zero_()

mlp.mlp_2stage(barriers, instructions, timings, upcopy, downcopy, inputs, hidden, outputs)
torch.cuda.synchronize()

# start_event.record()
# for i in range(ITERS):
#     mlp.mlp_2stage(many_barriers[i], instructions, many_timings[i], up, down, inputs, hidden, outputs)
# end_event.record()
# torch.cuda.synchronize()
# elapsed_time = start_event.elapsed_time(end_event)
# print(f"Execution time: {elapsed_time/ITERS*1000:.2f} us")

print(timings[:128,:,4].min(), timings[:128,:,4].max())

# print(timings[0])
# print(timings[127])

# print(instructions[:,:,0])


print(ref.shape)
print(ref)
print(outputs)

# breakpoint()
save_gantt_chart(timings, instructions, save_all=True, name='2stage', clock_rate=1000)

print(timings[0,0,80:83])
print(timings[0,1,80:83])