from typing import List

import matplotlib.pyplot as plt


def plot_entropies(
    token_str_list: List[str], 
    token_entropy_list: List[float]
):
    assert len(token_str_list) == len(token_entropy_list)
    token_str_list = [s.replace("\n", "\\n") for s in token_str_list]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        range(len(token_entropy_list)), 
        token_entropy_list
    )

    ax.set_xticks(
        range(len(token_str_list)), 
        token_str_list, 
        rotation=90, 
        fontsize=8
    )
    ax.get_yaxis().set_visible(False)
    ax.margins()
    plt.tight_layout()
    plt.savefig('./local/entropies.png', dpi=400)
