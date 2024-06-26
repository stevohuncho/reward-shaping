import pandas as pd
import matplotlib.pyplot as plt
from typing import List

class PlotEvalData:
    def __init__(self,
        filepath: str = "",
        label: str = "",
        color: str = "blue",
        frame_offset: int = 0,
        show_min_max_fill: bool = True
    ):
        self.filepath = filepath
        self.label = label
        self.color = color
        self.frame_offset = frame_offset
        self.show_min_max_fill = show_min_max_fill

def plot_eval(data: List[PlotEvalData], title: str = "Data Plot", x_label: str = "Frames", y_label: str = "Reward"):
    ax = None
    for d in data:
        df = pd.read_csv(d.filepath)
        df["trained_frames"] = df["trained_frames"] - d.frame_offset
        df.set_index("trained_frames", inplace=True)
        return_min = df['return_min']
        return_max = df['return_max']
        df = df.filter(like='return_mean', axis=1)
        df = df.rename(columns={'return_mean': d.label})
        ax = df.plot(color=d.color, ax=ax)
        if d.show_min_max_fill:
            plt.fill_between(df.index, return_min, return_max, alpha=0.2, color=d.color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    return ax