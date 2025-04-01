from typing import List

import matplotlib.pyplot as plt


def plot_entropies(
    token_str_list: List[str], 
    token_entropy_list: List[float], 
    save_path: str
):
    assert len(token_str_list) == len(token_entropy_list)
    token_str_list = [s.replace("\n", "\\n") for s in token_str_list]
    _, ax = plt.subplots(figsize=(4, 16))
    ax.barh(
        range(len(token_entropy_list)), 
        token_entropy_list
    )

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
