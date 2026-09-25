"""Reproduce py-meanflow's FID evaluation for the frozen flow map.

Uses the authors' own data loader and evaluation loop, with net_ema1, 50000 samples
and seed 0, which gives the FID of 2.812 quoted in the report. Also saves grids of
generated and real CIFAR-10 images side by side.
"""
import sys, os, logging
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "py-meanflow", "meanflow")
os.chdir(REPO)
sys.path.insert(0, REPO)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

from train_arg_parser import get_args_parser
from models.model_configs import instantiate_model
from training.eval_loop import eval_model
from train import get_data_loader

args = get_args_parser().parse_args([])
args.use_edm_aug = True
args.distributed = False
args.batch_size = 128
args.compute_fid = True
args.seed = 0
args.output_dir = f"{HERE}/pymf_eval_out"
args.data_path = f"{HERE}/cifar_data_pymf"

ck = torch.load(f"{HERE}/pymf_cifar10.pth", map_location="cpu", weights_only=False)
device = "cuda"
model = instantiate_model(args)
model.to(device)
model.load_state_dict(ck["model"])
epoch = ck["epoch"] + 1
print(f"loaded epoch {epoch}; fid_samples={args.fid_samples}", flush=True)

# side-by-side grids first (cheap): generated vs real
from torchvision.utils import make_grid, save_image
torch.manual_seed(123)
with torch.no_grad():
    gen = model.sample(samples_shape=(64, 3, 32, 32), net=model.net_ema1, device=device)
gen = (gen.clamp(-1, 1) + 1) / 2
loader = get_data_loader(args, is_for_fid=True)
real = next(iter(torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=True,
                                             generator=torch.Generator().manual_seed(7))))[0]
save_image(make_grid(gen, nrow=8), f"{HERE}/pymf_compare_generated.png")
save_image(make_grid(real, nrow=8), f"{HERE}/pymf_compare_real.png")
print("grids saved", flush=True)

data_loader_fid = get_data_loader(args, is_for_fid=True)
stats = eval_model(model, model.net_ema1, data_loader_fid, device,
                   epoch=epoch, args=args, suffix="_ema1")
print(f"FID RESULT (net_ema1, {args.fid_samples} samples vs 50k train): {stats['fid']:.3f}", flush=True)
print("FID REPRO DONE", flush=True)
