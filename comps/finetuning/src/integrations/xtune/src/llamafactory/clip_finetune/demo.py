import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
try:
   import intel_extension_for_pytorch
except:
   print("cant't import ipex")

import oneccl_bindings_for_pytorch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-dev', type=str, default='cpu', help='Device type to use: cpu, xpu')
parser.add_argument('--dist_url', default='127.0.0.1', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_port', default='29800', type=str, help='url port used to set up distributed training')
args = parser.parse_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)


if __name__ == "__main__":

    mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
    mpi_rank = int(os.environ.get('PMI_RANK', -1))
    if mpi_world_size > 0:
        os.environ['RANK'] = str(mpi_rank)
        os.environ['WORLD_SIZE'] = str(mpi_world_size)
    else:
        # set the default rank and world size to 0 and 1
        os.environ['RANK'] = str(os.environ.get('RANK', 0))
        os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # your master address
    os.environ['MASTER_PORT'] = '29500'  # your master port
    rank = int(os.environ.get('PMI_RANK', -1)) # global rank
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    init_method = 'tcp://' + args.dist_url + ':' + args.dist_port

    # Initialize the process group with ccl backend
    dist.init_process_group(backend='ccl', init_method=init_method, world_size=world_size, rank=rank)

    local_rank = os.environ['MPI_LOCALRANKID']
    if args.device == 'xpu':
        device = "xpu:{}".format(local_rank)
    else:
        device = 'cpu'

    model = Model().to(device)
    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[device] if (device != 'cpu') else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss().to(device)
    for i in range(3):
        print("Runing Iteration: {} on device {}".format(i, device))
        input = torch.randn(2, 4).to(device)
        labels = torch.randn(2, 5).to(device)
        # forward
        print("Runing forward: {} on device {}".format(i, device))
        res = model(input)
        # loss
        print("Runing loss: {} on device {}".format(i, device))
        L = loss_fn(res, labels)
        # backward
        print("Runing backward: {} on device {}".format(i, device))
        with torch.autograd.profiler_legacy.profile(enabled=True) as prof:
              L.backward()
        #print(prof)
        # update
        print("Runing optim: {} on device {}".format(i, device))
        optimizer.step()
    print("Finish")