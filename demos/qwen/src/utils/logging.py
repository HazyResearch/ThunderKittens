"""
Logging utilities to make terminal slightly more delightful
"""
import rich.syntax
import rich.tree

from omegaconf import OmegaConf, DictConfig, ListConfig


def _format_arg(arg_name: str, cutoff=2) -> str:
    if arg_name is None:
        return arg_name
    arg_name = str(arg_name)
    
    # Hardcode to handle backslash
    name_splits = arg_name.split('/')
    if len(name_splits) > 1:
        return name_splits[-1]
    # Abbreviate based on underscore
    name_splits = arg_name.split('_')
    if len(name_splits) > 1:
        return ''.join([s[0] for s in name_splits])
    else:
        return arg_name[:cutoff]


def print_header(x: str) -> None:
    """
    Print a header with a line above and below
    """
    print('-' * len(x))
    print(x)
    print('-' * len(x))


def print_args(args, return_dict=False, verbose=True):
    """
    Print the arguments passed to the script
    """
    attributes = [a for a in dir(args) if a[0] != '_']
    arg_dict = {}  # switched to ewr
    if verbose:
        print('ARGPARSE ARGS')
    for ix, attr in enumerate(attributes):
        fancy = '└─' if ix == len(attributes) - 1 else '├─'
        if verbose:
            print(f'{fancy} {attr}: {getattr(args, attr)}')
        arg_dict[attr] = getattr(args, attr)
    if return_dict:
        return arg_dict


def update_description_metrics(description: str, metrics: dict):
    """
    Set the numbers that show up on progress bars
    """
    for split in metrics:
        if split != 'test':  # No look
            for metric_name, metric in metrics[split].items():
                description += f' | {split}/{metric_name}: {metric:.3f}'
    return description

        
# Control how tqdm progress bar looks        
def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'
    
# Progress bar 
def update_pbar_display(metrics, batch_ix, pbar, prefix, batch_size, accum_iter=1):
    description = f'└── {prefix} batch {int(batch_ix)}/{len(pbar)} [batch size: {batch_size} - grad. accum. over {accum_iter} batch(es)]'
    for metric_name, metric in metrics.items():
        if metric_name == 'correct':
            description += f' | {metric_name} (acc. %): {int(metric):>5d}/{int(metrics["total"])} = {metric / metrics["total"] * 100:.3f}%'
        elif metric_name == 'acc':
            description += f' | {metric_name}: {metric:.3f}'
        elif metric_name in ['perplexity']:  # , 'bpc']:
            description += f' | {metric_name}: {Decimal(metric):.3E}'            
        elif metric_name != 'total':
            description += f' | {metric_name}: {metric / metrics["total"]:.3f}'
    pbar.set_description(description)
    
    
def print_config(config: DictConfig,
                 resolve: bool = True,
                 name: str = 'CONFIG') -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "bright"  # "dim"
    tree = rich.tree.Tree(name, style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        elif isinstance(config_section, ListConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)