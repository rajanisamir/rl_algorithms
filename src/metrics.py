import os
import logging

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Metric:
    def __init__(self, name, writer):
        self.name = name
        self.writer = writer

        self.x = []
        self.y = []

    def get_average(self, k):
        return sum(self.y[-k:]) / len(self.y[-k:])

    def update_metric(self, y):
        self.writer.add_scalar(self.name, y, len(self.x) + 1)
        self.x.append(len(self.x) + 1)
        self.y.append(y)

    def plot(self, ax):
        ax.set_title(self.name)
        ax.plot(self.x, self.y)


class MetricTracker:
    def __init__(self, save_dir):
        self.tb_path = os.path.join(save_dir, "tb_logs")
        self.fig_path = os.path.join(save_dir, "figs")

        logger.info(f"tensorboard logs will be saved at {self.tb_path}")
        logger.info(f"figures will be saved at {self.fig_path}")

        os.makedirs(self.tb_path, exist_ok=True)
        os.makedirs(self.fig_path, exist_ok=True)

        self.writer = SummaryWriter(self.tb_path)
        self.save_dir = save_dir
        self.metrics = {}

    def print_metrics(self, stats):
        metric_info = []
        for stat_name, stat_value in stats.items():
            metric_info.append(f"{stat_name}: {stat_value:5}")
        for metric in self.metrics.values():
            metric_info.append(f"{metric.name}: {metric.get_average(10):6.2f}")
        logger.info(", ".join(metric_info))

    def update_metric(self, name, y):
        if name not in self.metrics:
            self.metrics[name] = Metric(name, self.writer)
        self.metrics[name].update_metric(y)

    def save_plots(self):
        if len(self.metrics) == 0:
            logger.warn("no metrics to save")
            return

        matplotlib.style.use("fivethirtyeight")
        fig, axs = plt.subplots(
            len(self.metrics), 1, figsize=(8, 5 * len(self.metrics))
        )
        if len(self.metrics) == 1:
            axs = [axs]
        for ax, metric in zip(axs, self.metrics.values()):
            metric.plot(ax)
        fig.tight_layout()
        fig_save_path = os.path.join(self.fig_path, "metrics.png")
        plt.savefig(fig_save_path)
        logger.info(f"figure saved at {fig_save_path}")
        # logger.info(f"metrics saved at {metric_save_path}")
