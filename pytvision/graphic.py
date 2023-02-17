import os

import cv2
import numpy as np
from visdom import Visdom


class VisdomLogger(object):
    def __init__(self, env_name="main", server=None, port=None) -> None:
        if server is None:
            server = os.environ.get("VISDOM_SERVER_URL", "localhost")
        if port is None:
            port = int(os.environ.get("VISDOM_PORT", 8097))

        self.viz = Visdom(server=server, port=port, use_incoming_socket=False)
        self.env = env_name


class VisdomLinePlotter(VisdomLogger):
    """Plots to Visdom"""

    def __init__(self, env_name="main", server=None, port=None):
        super(VisdomLinePlotter, self).__init__(env_name, server, port)
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=var_name,
                    xlabel="Epoch",
                    ylabel=var_name,
                    showlegend=True,
                ),
            )

        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
                opts=dict(showlegend=True),
            )

            # self.viz.updateTrace(X=np.array([x]), Y=np.array([y]),
            #     env=self.env,
            #     win=self.plots[var_name],
            #     name=split_name
            #     )


class VisdomScatter(VisdomLogger):
    """Scatter to Visdom"""

    def __init__(self, env_name="main", server=None, port=None):
        super(VisdomScatter, self).__init__(env_name, server, port)
        self.scatters = {}

    def scatter(self, X, Y, title, legend, markersize=10):
        if title not in self.scatters:
            self.scatters[title] = self.viz.scatter(
                X=X,
                Y=Y,
                env=self.env,
                opts=dict(
                    # legend=legend,
                    markersize=markersize,
                    title=title,
                ),
            )
        else:
            self.viz.scatter(
                X=X,
                Y=Y,
                env=self.env,
                win=self.scatters[title],
                opts=dict(
                    # legend=legend,
                    markersize=markersize,
                    title=title,
                ),
            )


class HeatMapVisdom(VisdomLogger):
    """Heat Map to Visdom"""

    def __init__(self, env_name="main", heatsize=None, server=None, port=None):
        super(HeatMapVisdom, self).__init__(env_name, server, port)
        self.hmaps = {}
        self.heatsize = heatsize

    def show(self, title, image):
        if self.heatsize:
            image = cv2.resize(image, self.heatsize, interpolation=cv2.INTER_LINEAR)

        if title not in self.hmaps:
            self.hmaps[title] = self.vis.heatmap(image, env=self.env, opts=dict(title=title))
        else:
            self.vis.heatmap(image, env=self.env, win=self.hmaps[title], opts=dict(title=title))


class ImageVisdom(VisdomLogger):
    """Images to Visdom"""

    def __init__(self, env_name="main", imsize=None, server=None, port=None):
        super(ImageVisdom, self).__init__(env_name, server, port)
        self.images = {}
        self.imsize = imsize

    def show(self, title, image):
        if self.imsize:
            image = cv2.resize(image, self.imsize, interpolation=cv2.INTER_LINEAR)
        image = image.astype(float)
        image = np.transpose(image, (2, 0, 1))

        if title not in self.images:
            self.images[title] = self.vis.image(image, env=self.env, opts=dict(title=title))
        else:
            self.vis.image(image, env=self.env, win=self.images[title], opts=dict(title=title))
