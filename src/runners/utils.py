from matplotlib import pyplot as plt
import seaborn as sn
from ..dataset.sentiment_dataloader import labels_to_idx


class PlotCMs:
    def __init__(self):
        self.cms = []
        self.titles = []
        self.classes = list(labels_to_idx.keys())
        sn.set(font_scale=1)

    def insert(self, cm, title):
        self.cms.append((cm*100).round())
        self.titles.append(title)

    def plot(self, save_url = None):
        n_cols = 4 if len(self.cms) > 4 else len(self.cms)
        n_rows = int(len(self.cms) / 3) + 1 if (len(self.cms) / 3) % 3 != 0 else int(len(self.cms) / 3)
        fig = plt.figure(figsize=(n_rows, n_cols))
        for i in range(len(self.cms)):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            plt.title(self.titles[i])
            sn.heatmap(self.cms[i], annot=True, cmap="Blues", square=True, ax=ax, fmt="g", cbar=False)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(25, 25)
        if save_url is not None:
            plt.savefig(save_url, bbox_inches='tight', dpi=400)
        else:
            plt.show()
        return fig
