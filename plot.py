from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(
        data: List[Dict[str, np.ndarray]],
        plot_titles: List[str],
        x_titles: List[str],
        y_titles: List[str]
):
    fig = make_subplots(
        rows=len(data),
        cols=1,
        subplot_titles=plot_titles
    )
    for plot_ind, one_plot_data in enumerate(data):
        for name, values in one_plot_data.items():
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=name
            ), row=plot_ind + 1, col=1)

    for title_ind, title in enumerate(x_titles):
        fig.update_xaxes(title_text=title, row=title_ind + 1, col=1)

    for title_ind, title in enumerate(y_titles):
        fig.update_yaxes(title_text=title, row=title_ind + 1, col=1)

    fig.show()
