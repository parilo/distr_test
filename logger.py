from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np

from distr import BaseDist


class ValueLogger:

    def __init__(self):
        self._params_log = defaultdict(list)

    def log(
            self,
            step_ind: int,
            train_data: Dict[str, float],
            src_distr: BaseDist,
            dst_distr: BaseDist
    ):
        for key, value in train_data.items():
            self._params_log[key].append(value)
        for pname, value in src_distr.get_annotated_params().items():
            self._params_log[f'src {src_distr.name} {pname}'].append(value)
        for pname, value in dst_distr.get_annotated_params().items():
            self._params_log[f'dst {dst_distr.name} {pname}'].append(value)

        print(f'step: {step_ind} log: {train_data} '
              f'src {src_distr.get_annotated_params()} '
              f'dst {dst_distr.get_annotated_params()}')

    def get_plot_data(self, groups: Optional[Dict[str, List[str]]] = None) -> Tuple[List[Dict[str, np.ndarray]], List[str], List[str], List[str]]:
        if groups is None:
            groups = {key: [key] for key in self._params_log.keys()}

        print(f'--- g {groups}')

        plot_data = []
        plot_titles = []
        plot_x_titles = []
        plot_y_titles = []
        for plot_name, params_list in groups.items():
            subplot_data = {}
            for pname in params_list:
                subplot_data[pname] = np.array(self._params_log[pname])  #{key: np.array(value) for key, value in data.items()}

            plot_data.append(subplot_data)
            plot_titles.append(plot_name)
            plot_x_titles.append('train steps')
            plot_y_titles.append('')

        return plot_data, plot_titles, plot_x_titles, plot_y_titles
