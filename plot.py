from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _calc_row_col(ind, row_size):
    return \
        (ind // row_size) + 1, \
        (ind % row_size) + 1


def plot(
        data: List[Dict[str, np.ndarray]],
        plot_titles: List[str],
        x_titles: List[str],
        y_titles: List[str],
        histogram_data: List[np.ndarray] = None,
        histogram_titles: List[str] = None,
):
    num_rows = 2
    row_size = len(data) // num_rows + 1
    fig = make_subplots(
        rows=num_rows,
        cols=row_size,
        subplot_titles=plot_titles
    )
    for plot_ind, one_plot_data in enumerate(data):
        plot_row, plot_col = _calc_row_col(plot_ind, row_size)
        for name, values in one_plot_data.items():
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=name
            ), row=plot_row, col=plot_col)

    for title_ind, title in enumerate(x_titles):
        title_row, title_col = _calc_row_col(title_ind, row_size)
        fig.update_xaxes(title_text=title, row=title_row, col=title_col)

    for title_ind, title in enumerate(y_titles):
        title_row, title_col = _calc_row_col(title_ind, row_size)
        fig.update_yaxes(title_text=title, row=title_row, col=title_col)

    if histogram_data is not None:
        for hist_data, hist_title in zip(histogram_data, histogram_titles):
            fig.add_trace(
                go.Histogram(
                    x=hist_data,
                    name=hist_title,
                ),
                row=num_rows,
                col=row_size,
            )

    fig.show()


def plot_histogram(data1, data2):
    fig = go.Figure(go.Histogram(
        x=data1,
    ))

    fig.add_trace(go.Histogram(
        x=data2,
    ))

    fig.show()

