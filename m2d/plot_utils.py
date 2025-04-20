from typing import List, Optional

import matplotlib.pyplot as plt


def plot_values(
    token_str_list: List[str], 
    value_list: List[float], 
    save_path: str, 
    steps: Optional[List[int]] = None
):
    assert len(token_str_list) == len(value_list)
    token_str_list = [s.replace("\n", "\\n") for s in token_str_list]
    _, ax = plt.subplots(figsize=(4, 16))

    if steps is None:
        ax.barh(
            range(len(value_list)), 
            value_list
        )
    else:
        offset = 0
        for step in steps:
            ax.barh(
                range(len(value_list))[offset:offset+step], 
                value_list[offset:offset+step]
            )
            offset += step

    ax.set_yticks(
        range(len(token_str_list)), 
        token_str_list, 
        fontsize=8
    )
    ax.get_xaxis().set_visible(False)
    ax.invert_yaxis()
    ax.margins(y=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
