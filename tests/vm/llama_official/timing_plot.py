import torch
import numpy as np
import time
from bokeh.plotting import figure, show, save
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Legend, LegendItem, Rect # Changed Segment to Rect
from bokeh.palettes import Category10, Viridis256 # Using Category10 for distinct instruction types
from bokeh.resources import CDN

# --- Configuration ---
CYCLE_FREQ_MHZ = 1800.0  # Clock frequency in MHz (e.g., 1.8 GHz = 1800 MHz)
# How many vertical sub-slots per processor lane to separate overlapping instructions
# Change K to adjust the number of vertical slots per processor lane
VERTICAL_SLOTS_K = 2
# How much of the vertical slot height the bar should occupy (e.g., 0.8 = 80%)
BAR_HEIGHT_RATIO = 2/3
# HTML Output file name prefix
OUTPUT_NAME_PREFIX = "timeline_bokeh"
# --- REMOVED Separator line style constants ---
# --- ADDED Background colors ---
BACKGROUND_COLOR_EVEN = "white"
BACKGROUND_COLOR_ODD = "#404040" # Light gray

# --- Updated Mappings ---
INSTRUCTION_MAP = {
    0: "No Op",
    1: "RMS QKV MatVec Rope Append",
    2: "Partial Attention",
    3: "Attention Reduction",
    4: "O Proj Residual",
    5: "RMS Double MatVec SiLU",
    6: "Down Proj Residual"
}

# Choose a color palette (using Category10, mapping known types)
palette = Category10[max(INSTRUCTION_MAP.keys())] if max(INSTRUCTION_MAP.keys()) > 0 else []
COLOR_MAP = {
    0: '#808080', # Dark Grey for No Op
    1: palette[0] if len(palette) > 0 else '#1f77b4',
    2: palette[1] if len(palette) > 1 else '#ff7f0e',
    3: palette[2] if len(palette) > 2 else '#2ca02c',
    4: palette[3] if len(palette) > 3 else '#d62728',
    5: palette[4] if len(palette) > 4 else '#9467bd',
    6: palette[5] if len(palette) > 5 else '#8c564b',
}

# --- Event Mapping ---
EVENT_CATEGORIES = {}
EVENT_DESCRIPTIONS = {}
# Event 0: Instruction Launch
EVENT_CATEGORIES[0] = 'launch'
EVENT_DESCRIPTIONS[0] = 'Instruction launch'
# Events 1-15: Initial Setup
for i in range(1, 16):
    EVENT_CATEGORIES[i] = 'setup'
    EVENT_DESCRIPTIONS[i] = f'Initial setup event {i}'
# Events 16-39: Loader Events
for i in range(16, 40):
    EVENT_CATEGORIES[i] = 'loader'
    EVENT_DESCRIPTIONS[i] = f'Loader event {i}'
# Events 40-103: Consumer / Launcher Events
for i in range(40, 104):
    EVENT_CATEGORIES[i] = 'consumer'
    EVENT_DESCRIPTIONS[i] = f'Consumer/launcher event {i}'
# Events 104-126: Storer / Synchronization Events
for i in range(104, 127):
    EVENT_CATEGORIES[i] = 'storer'
    EVENT_DESCRIPTIONS[i] = f'Storer/sync event {i}'
# Event 127: Instruction End
EVENT_CATEGORIES[127] = 'end'
EVENT_DESCRIPTIONS[127] = 'Instruction end'

# Define marker styles per category
MARKER_STYLES = {
    'launch':   {'marker': 'diamond', 'color': 'green',    'size': 8, 'legend': 'Launch'},
    'setup':    {'marker': 'asterisk','color': 'purple',   'size': 7, 'legend': 'Setup'},
    'loader':   {'marker': 'circle',  'color': 'blue',     'size': 5, 'legend': 'Loader'},
    'consumer': {'marker': 'triangle','color': 'red',      'size': 5, 'legend': 'Consumer/Launch'},
    'storer':   {'marker': 'square',  'color': 'orange',   'size': 5, 'legend': 'Storer/Sync'},
    'end':      {'marker': 'cross',   'color': 'black',    'size': 8, 'legend': 'End'},
}


def save_gantt_chart_bokeh(Timings, Instructions, k_slots=VERTICAL_SLOTS_K, save_all=False, name=None, verbose=False):
    """
    Generates an interactive Bokeh Gantt chart for instruction timings,
    using alternating background colors for processors.

    Args:
        Timings (torch.Tensor): Shape (num_procs, num_instr, 128) - timing data in cycles.
        Instructions (torch.Tensor): Shape (num_procs, num_instr, ...) - instruction info.
                                     Instructions[p, i, 0] is the instruction type opcode.
        k_slots (int): Number of vertical slots per processor lane.
        save_all (bool): Whether to include detailed event markers.
        name (str, optional): Custom name for the output HTML file.
        verbose (bool): Whether to print timing statistics.
    """
    print("Starting Bokeh chart generation...")
    start_time_proc = time.time()

    # Ensure tensors are on CPU and convert timings to microseconds
    if Timings.is_cuda: Timings = Timings.cpu()
    if Instructions.is_cuda: Instructions = Instructions.cpu()

    timings_us = Timings.float() / CYCLE_FREQ_MHZ
    num_processors, num_instructions, _ = Timings.shape

    # --- Prepare Data for Bokeh ColumnDataSources ---
    bar_data = {
        'left': [], 'right': [], 'bottom': [], 'top': [], 'color': [],
        'name': [], 'proc': [], 'instr_idx': [], 'duration': [], 'start_us': [], 'end_us': []
    }
    marker_data = {category: {'x': [], 'y': [], 'name': [], 'proc': [], 'instr_idx': []}
                   for category in MARKER_STYLES}

    bar_height = BAR_HEIGHT_RATIO / k_slots
    bar_margin = (1.0 - BAR_HEIGHT_RATIO) / (k_slots + 1)

    min_time_overall = float('inf')
    max_time_overall = float('-inf')

    print(f"Processing data for {num_processors} processors, {num_instructions} instructions...")
    for proc in range(num_processors):
        for instr in range(num_instructions):
            instr_type = Instructions[proc, instr, 0].item()
            start = timings_us[proc, instr, 0].item()
            end = timings_us[proc, instr, 127].item()

            if start > 0 and end > start :
                duration = end - start
                instr_name = INSTRUCTION_MAP.get(instr_type, f"Unknown ({instr_type})")
                instr_color = COLOR_MAP.get(instr_type, 'gray')

                min_time_overall = min(min_time_overall, start)
                max_time_overall = max(max_time_overall, end)

                slot_idx = instr % k_slots
                y_base = proc
                bottom = y_base - 0.5 + (slot_idx * bar_height) + (slot_idx+1) * bar_margin
                top = bottom + bar_height
                marker_y = (bottom+top)/2

                bar_data['left'].append(start)
                bar_data['right'].append(end)
                bar_data['bottom'].append(bottom)
                bar_data['top'].append(top)
                bar_data['color'].append(instr_color)
                bar_data['name'].append(instr_name)
                bar_data['proc'].append(proc)
                bar_data['instr_idx'].append(instr)
                bar_data['duration'].append(duration)
                bar_data['start_us'].append(start)
                bar_data['end_us'].append(end)

                if save_all:
                    for event_id in range(128):
                        event_time = timings_us[proc, instr, event_id].item()
                        if event_time > 0 and event_time >= start and event_time <= end:
                            category = EVENT_CATEGORIES.get(event_id)
                            if category:
                                event_name = EVENT_DESCRIPTIONS.get(event_id, f"Event {event_id}")
                                marker_data[category]['x'].append(event_time)
                                marker_data[category]['y'].append(marker_y)
                                marker_data[category]['name'].append(event_name)
                                marker_data[category]['proc'].append(proc)
                                marker_data[category]['instr_idx'].append(instr)

    print(f"Data processing finished in {time.time() - start_time_proc:.2f}s.")

    if min_time_overall == float('inf'):
        print("Warning: No valid instruction timings found. Generating empty plot.")
        min_time_overall = 0
        max_time_overall = 1

    # --- Create Bokeh Figure ---
    y_range_start = -0.5
    y_range_end = num_processors - 0.5

    plot_height = max(400, min(3000, num_processors * 20 * max(1, k_slots / 1.5)))
    time_diff = max(1.0, max_time_overall - min_time_overall)
    plot_width = max(800, min(6000, int(plot_height * max(5, time_diff / (num_processors * 5)))))

    print(f"Creating Bokeh figure (width={plot_width}, height={plot_height})...")
    tools = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(
        height=int(plot_height),
        width=int(plot_width),
        tools=tools,
        y_range=(y_range_start, y_range_end),
        x_range=(min_time_overall - 0.05 * time_diff,
                 max_time_overall + 0.05 * time_diff),
        title="Instruction Execution Timeline (Interactive)",
        x_axis_label="Time (microseconds)",
        y_axis_label="Processor ID"
    )
    p.xaxis.formatter = NumeralTickFormatter(format="0,0.0")
    p.yaxis.ticker = list(range(num_processors))
    p.yaxis.major_label_overrides = {i: str(i) for i in range(num_processors)}
    # Remove grid lines for cleaner look with background colors
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None


    # --- ADD ALTERNATING BACKGROUND RECTANGLES ---
    bg_colors = []
    bg_y = []
    bg_height = []
    for proc_idx in range(num_processors):
        bg_colors.append(BACKGROUND_COLOR_EVEN if proc_idx % 2 == 0 else BACKGROUND_COLOR_ODD)
        bg_y.append(proc_idx) # Center of the rectangle is the processor ID
        bg_height.append(1.0) # Height of the rectangle is 1 unit

    # Calculate the width needed to span the entire plot x-range
    bg_width = p.x_range.end - p.x_range.start
    # Calculate the center x-coordinate
    bg_x_center = p.x_range.start + bg_width / 2

    p.rect(
        x=[bg_x_center] * num_processors, # Center rectangles horizontally
        y=bg_y,
        width=[bg_width] * num_processors, # Span full plot width
        height=bg_height,
        fill_color=bg_colors,
        line_color=None, # No border for the background rectangles
        level='underlay' # Draw behind data glyphs
    )


    # --- Add Data Glyphs ---
    # Instruction Bars
    bar_source = ColumnDataSource(bar_data)
    quad_renderer = p.quad(
        left='left', right='right', bottom='bottom', top='top',
        color='color',
        source=bar_source,
        legend_label="Instructions",
        # line_color="black", line_width=0.5,
        name="bars"
    )

    # --- REMOVED SEPARATOR LINES ---

    # Event Markers (add only if save_all)
    marker_renderers = {}
    if save_all:
        for category, style in MARKER_STYLES.items():
            if category in marker_data and marker_data[category]['x']:
                source = ColumnDataSource(marker_data[category])
                renderer = p.scatter(
                    x='x', y='y', source=source,
                    marker=style['marker'], color=style['color'], size=style['size'],
                    legend_label=style['legend'],
                    name=f"marker_{category}"
                )
                marker_renderers[category] = renderer

    # --- Add Hover Tools ---
    bar_hover = HoverTool(
        renderers=[quad_renderer],
        tooltips=[
            ("Processor", "@proc"),
            ("Instruction Index", "@instr_idx"),
            ("Type", "@name"),
            ("Start", "@start_us{0,0.00} us"),
            ("End", "@end_us{0,0.00} us"),
            ("Duration", "@duration{0,0.00} us"),
        ]
    )
    p.add_tools(bar_hover)

    if save_all:
        marker_hover_tooltips=[
            ("Processor", "@proc"),
            ("Instruction Index", "@instr_idx"),
            ("Event", "@name"),
            ("Time", "@x{0,0.00} us"),
        ]
        marker_hover_renderers = [r for r in marker_renderers.values() if r is not None]
        if marker_hover_renderers:
             marker_hover = HoverTool(
                 renderers=marker_hover_renderers,
                 tooltips=marker_hover_tooltips,
             )
             p.add_tools(marker_hover)


    # --- Customize Legend ---
    instr_legend_items = []
    seen_types = set()
    if bar_data['left']:
        unique_proc_instr_pairs = set(zip(bar_data['proc'], bar_data['instr_idx']))
        present_instr_type_codes = set()
        for p_idx, i_idx in unique_proc_instr_pairs:
             if p_idx < Instructions.shape[0] and i_idx < Instructions.shape[1]:
                 present_instr_type_codes.add(Instructions[p_idx, i_idx, 0].item())
        sorted_types = sorted(list(present_instr_type_codes))
        for instr_type_code in sorted_types:
            if instr_type_code not in seen_types:
                instr_name = INSTRUCTION_MAP.get(instr_type_code, f"Unknown ({instr_type_code})")
                color = COLOR_MAP.get(instr_type_code, 'gray')
                dummy_renderer = p.rect(x=0, y=0, width=0, height=0, fill_color=color, line_color=color, visible=False)
                instr_legend_items.append(LegendItem(label=f"{instr_name} ({instr_type_code})", renderers=[dummy_renderer]))
                seen_types.add(instr_type_code)

    marker_legend_items = []
    if save_all:
        sorted_categories = sorted(marker_renderers.keys(), key=lambda cat: list(MARKER_STYLES.keys()).index(cat))
        for category in sorted_categories:
            renderer = marker_renderers.get(category)
            if renderer:
                marker_legend_items.append(LegendItem(label=MARKER_STYLES[category]['legend'], renderers=[renderer]))

    p.legend.visible = False
    combined_legend_items = instr_legend_items + marker_legend_items
    if combined_legend_items:
        legend = Legend(
            items=combined_legend_items,
            location="center_right",
            orientation="vertical",
            click_policy="hide",
            title="Legend",
            label_text_font_size="8pt",
            spacing=1, margin=10, padding=5,
            glyph_height=15, glyph_width=15,
            border_line_alpha=0.1
        )
        p.add_layout(legend, 'right')

    # --- Save and Show ---
    filename = f"{OUTPUT_NAME_PREFIX}_{name if name else int(time.time())}.html"
    save(p, filename=filename, title=f"Instruction Timeline {name if name else ''}", resources=CDN)
    print(f"Bokeh chart saved to: {filename}")
    print(f"Total generation time: {time.time() - start_time_proc:.2f}s.")

    # --- Verbose Statistics ---
    if verbose:
        if len(Timings.shape) == 3 and Timings.shape[2] >= 128:
            valid_mask = (Timings[:, :, 0] > 0) & (Timings[:, :, 127] > Timings[:, :, 0])
            if valid_mask.any():
                valid_start_us = timings_us[:, :, 0][valid_mask]
                valid_end_us = timings_us[:, :, 127][valid_mask]
                valid_durations_us = valid_end_us - valid_start_us

                print("\nTiming Statistics for valid instructions (microseconds):")
                print(f"Average instruction duration: {valid_durations_us.mean():.3f}")
                print(f"Min instruction duration: {valid_durations_us.min():.3f}")
                print(f"Max instruction duration: {valid_durations_us.max():.3f}")
                print(f"Total execution time (sum of durations): {valid_durations_us.sum():.3f}")
                print(f"Overall start time (min start time): {valid_start_us.min():.3f}")
                print(f"Overall end time (max end time): {valid_end_us.max():.3f}")

                print("\nAverage Duration by Instruction Type:")
                if len(Instructions.shape) >= 2:
                    valid_instructions = Instructions[valid_mask]
                    if valid_instructions.numel() > 0 and valid_instructions.shape[-1] > 0:
                        unique_types = valid_instructions[:, 0].unique()
                        for instr_type_code_tensor in sorted(unique_types):
                            instr_type_code = instr_type_code_tensor.item()
                            type_mask_on_valid = (valid_instructions[:, 0] == instr_type_code)
                            if type_mask_on_valid.any():
                                 mean_time = valid_durations_us[type_mask_on_valid].mean()
                                 count = type_mask_on_valid.sum().item()
                                 type_name = INSTRUCTION_MAP.get(instr_type_code, f"Unknown ({instr_type_code})")
                                 print(f"  {type_name} ({instr_type_code}): {mean_time:.3f} us (Count: {count})")
                    else: print("  Could not extract instruction types from valid instructions.")
                else: print("  Instructions tensor shape is not as expected for statistics.")
            else: print("\nNo valid instruction timings found for statistics.")
        else: print("\nTimings tensor shape is not as expected for statistics.")


# --- Example Usage ---
if __name__ == '__main__':
    print("Generating improved dummy data for testing...")
    NUM_PROCS = 148
    NUM_INSTR = 96
    K_DUMMY = VERTICAL_SLOTS_K
    GAP_CYCLES = 1000
    SETUP_EVENTS = list(range(1, 16))
    LOADER_EVENTS = list(range(16, 40))
    CONSUMER_EVENTS = list(range(40, 104))
    STORER_EVENTS = list(range(104, 127))

    dummy_timings = torch.zeros(NUM_PROCS, NUM_INSTR, 128, dtype=torch.int64)
    dummy_instructions = torch.zeros(NUM_PROCS, NUM_INSTR, 1, dtype=torch.int64)
    last_slot_end_time = torch.zeros(NUM_PROCS, K_DUMMY, dtype=torch.int64)

    for p in range(NUM_PROCS):
        for i in range(NUM_INSTR):
            if torch.rand(1) < 0.9:
                instr_type = torch.randint(1, 7, (1,)).item()
                dummy_instructions[p, i, 0] = instr_type
                slot_idx = i % K_DUMMY
                min_start_cycles = last_slot_end_time[p, slot_idx] + GAP_CYCLES
                start_cycles = min_start_cycles + torch.randint(0, 200, (1,)).item()
                duration_cycles = torch.randint(500, 5000, (1,)).item()
                end_cycles = start_cycles + duration_cycles
                last_slot_end_time[p, slot_idx] = end_cycles
                dummy_timings[p, i, 0] = start_cycles
                dummy_timings[p, i, 127] = end_cycles

                if duration_cycles > 100 and start_cycles < end_cycles:
                    # Setup Events
                    num_setup = torch.randint(1, 4, (1,)).item()
                    valid_setup_events = [e for e in SETUP_EVENTS if e != 0 and e != 127]
                    if valid_setup_events:
                        time_range_start_setup = start_cycles + 1
                        time_range_end_setup = max(time_range_start_setup, int(start_cycles + 0.15 * duration_cycles))
                        if time_range_end_setup > time_range_start_setup:
                            for _ in range(num_setup):
                                event_id = valid_setup_events[torch.randint(0, len(valid_setup_events), (1,)).item()]
                                event_time = torch.randint(time_range_start_setup, time_range_end_setup + 1, (1,)).item()
                                if dummy_timings[p, i, event_id] == 0: dummy_timings[p, i, event_id] = event_time
                    # Loader/Consumer Events
                    num_load_consume = torch.randint(5, 16, (1,)).item()
                    valid_load_consume_events = [e for e in LOADER_EVENTS + CONSUMER_EVENTS if e != 0 and e != 127]
                    if valid_load_consume_events:
                        time_range_start_lc = max(start_cycles + 1, int(start_cycles + 0.10 * duration_cycles))
                        time_range_end_lc = min(end_cycles - 1, int(end_cycles - 0.15 * duration_cycles))
                        if time_range_end_lc > time_range_start_lc:
                            for _ in range(num_load_consume):
                                event_id = valid_load_consume_events[torch.randint(0, len(valid_load_consume_events), (1,)).item()]
                                event_time = torch.randint(time_range_start_lc, time_range_end_lc + 1, (1,)).item()
                                if dummy_timings[p, i, event_id] == 0: dummy_timings[p, i, event_id] = event_time
                    # Storer Events
                    num_storer = torch.randint(2, 5, (1,)).item()
                    valid_storer_events = [e for e in STORER_EVENTS if e != 0 and e != 127]
                    if valid_storer_events:
                        time_range_start_st = max(start_cycles + 1, int(end_cycles - 0.15 * duration_cycles))
                        time_range_end_st = end_cycles - 1
                        if time_range_end_st > time_range_start_st:
                             for _ in range(num_storer):
                                event_id = valid_storer_events[torch.randint(0, len(valid_storer_events), (1,)).item()]
                                event_time = torch.randint(time_range_start_st, time_range_end_st + 1, (1,)).item()
                                if dummy_timings[p, i, event_id] == 0: dummy_timings[p, i, event_id] = event_time
            else:
                 dummy_instructions[p, i, 0] = 0

    print("Improved dummy data generated.")
    save_gantt_chart_bokeh(
        dummy_timings,
        dummy_instructions,
        k_slots=K_DUMMY,
        save_all=True,
        name="example_run_alternating_bg", # Updated name
        verbose=True
    )
