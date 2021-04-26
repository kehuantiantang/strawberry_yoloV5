# coding=utf-8
import torch


class Detect(torch.nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.output_shapes = [
            (1, 3, 56, 56, self.no),
            (1, 3, 28, 28, self.no),
            (1, 3, 14, 14, self.no)
        ]
        self.stride = [8, 16, 32]
        self.anchors = a
        self.anchor_grid =  a.clone().view(self.nl, 1, -1, 1, 1, 2)

        # self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

    def forward(self, x):
        # x = x.copy()  # for profiling
        x = [torch.from_numpy(i) for i in x]
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx, _ =  self.output_shapes[i]
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        result = torch.cat(z, 1)
        return result

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()