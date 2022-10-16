import matplotlib.pyplot as plt

def render_table(row_labels, col_labels, cell_data, table_scale=(1, 4), **kwargs):
    fig, ax = plt.subplots(**kwargs)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(
        cellText=cell_data, 
        rowLabels=col_labels, 
        colLabels=row_labels, 
        cellLoc='center',
        loc='upper left')
    table.scale(*table_scale)
    fig.tight_layout()
    fig.show()
    return fig, ax