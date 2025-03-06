import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange

"""
Timing meanings, all relative to start of kernel, and measured in cycles.
(In other words, this is describing the value in Timings[processor, instruction, THIS_ID])

0 -- start of instruction.
1 -- start of consumer setup load.
2 -- end of consumer setup load.

8...31 -- start of consumer compute. Iters after 24 are elided.

32...55 -- producer loads launched. Iters after 56 are elided.

62 -- start of consumer finish write.
63 -- end of consumer finish write (and instruction), relative to start of kernel.
"""

def save_gantt_chart(Timings, Instructions, verbose=False):
    # Convert cycles to microseconds (1.8 GHz = 1800 MHz = 1.8 cycles/ns = 0.0018 cycles/us)
    timings_us = Timings.float() / 1800

    # Get unique instruction types for coloring
    instruction_types = Instructions[:,:,0].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(instruction_types)))
    color_map = dict(zip(instruction_types.cpu().numpy(), colors))
    color_map = {
        0: '#000000',
        1: '#FEC601',
        2: '#5DA271'
    }
    instruction_names = {
        1: 'Partial',
        2: 'Reduction'
    }

    # Define markers for different timing events
    timing_events = {
        1: {'marker': '*', 'color': 'black', 'label': 'Start of consumer setup load', 'size': 32},
        2: {'marker': 'x', 'color': 'black', 'label': 'End of consumer setup load', 'size': 32},
        62: {'marker': 'P', 'color': 'black', 'label': 'Start of consumer finish write', 'size': 32}
    }
    timing_events.update({
        # Consumer compute starts (8-31)
        i: {'marker': 'd', 'color': '#FF312E', 'label': f'Consumer compute iteration', 'size': 24}
        for i in range(8, 32)
    })
    timing_events.update({
        # Producer loads (32-55)
        i: {'marker': 'o', 'color': '#1F01B9', 'label': f'Producer load iteration', 'size': 16}
        for i in range(32, 56)
    })

    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
    
    # For each processor
    for proc in trange(Timings.shape[0]):
        for instr in range(Timings.shape[1]):
            # Only process if there's valid timing data
            if Timings[proc, instr, 0].item() > 0:
                start = timings_us[proc, instr, 0].item()
                end = timings_us[proc, instr, 63].item() if Timings[proc, instr, 63].item() > 0 else start
                duration = end - start
                
                if duration > 0:  # Only plot if there was actual work
                    instr_type = Instructions[proc, instr, 0].item()
                    ax.barh(proc, duration, left=start, 
                           color=color_map[instr_type],
                           alpha=0.7)
                    
                    # Add markers for timing events
                    for event_id, event_props in timing_events.items():
                        if Timings[proc, instr, event_id].item() > 0:
                            event_time = timings_us[proc, instr, event_id].item()
                            ax.scatter(event_time, proc, 
                                      marker=event_props['marker'], 
                                      color=event_props['color'], 
                                      s=event_props['size'],
                                      zorder=5)  # Ensure markers are on top

    # Customize the chart
    ax.set_xlabel('Time (microseconds)')
    ax.set_ylabel('Processor ID')
    ax.set_title('Instruction Execution Timeline with Event Markers')

    # Add legend for instruction bars
    bar_legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[t.item()], 
                          label=f'Instruction {t.item()}: {instruction_names[t.item()]}')
                          for t in instruction_types if t.item() in [1,2]]
    
    # Add legend for timing event markers
    marker_legend_elements = [plt.Line2D([0], [0], marker=props['marker'], color='w', 
                             markerfacecolor=props['color'], markersize=10, 
                             label=props['label'])
                             for event_id, props in timing_events.items()
                             if event_id in [1,2,8,32,62]]
    
    # Combine legends
    all_legend_elements = bar_legend_elements + marker_legend_elements
    
    # Add the legend outside the plot
    ax.legend(handles=all_legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'instruction_timeline_{int(time.time())}.png', dpi=200)

    if verbose:
        # Print timing statistics
        print("\nTiming Statistics (microseconds):")
        print(f"Average instruction time: {timings_us.mean():.3f}")
        print(f"Max instruction time: {timings_us.max():.3f}")
        print(f"Total execution time: {timings_us.sum():.3f}")
        print("\nBy instruction type:")
        for instr_type in instruction_types:
            mask = Instructions[:,:,0] == instr_type
            mean_time = timings_us[mask].mean()
            print(f"Instruction {instr_type.item()}: {mean_time:.3f}")
