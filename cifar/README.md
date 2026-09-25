# CIFAR-10 experiments

The noise adapter and the VFM baseline on three tasks, all using the same frozen flow map.
Each script is configured through environment variables and run directly. For example:

```
OBJ=mix BETA=100 STEPS=6000 python train_inpaint_tarflow.py
```

| file | contents |
|---|---|
| `common/flow_map.py` | the frozen flow map, its one-step decode, and shared constants |
| `common/fid_check.py` | reproduces py-meanflow's reported FID for that checkpoint |
| `brightness-steering/task_setup.py` | calibrates the brightness target and observation noise, and reports the metric floors |
| `brightness-steering/train_thesis_gaussians.py` | Gaussian adapters: exact posterior by rejection, maximum likelihood fits, and reverse KL |
| `brightness-steering/train_thesis_nf.py` | normalising flow adapter, trained by reverse KL |
| `amortised-4x-SR/tarflow_adapter.py` | the conditional TarFlow adapter for 4x super-resolution |
| `amortised-4x-SR/train_thesis_ablation.py` | the objective ablation: forward KL, reverse KL, and the mixture of the two |
| `amortised-inpainting/train_inpaint_tarflow.py` | box inpainting, and the inversion diagnostic with `FULLOBS=1` |
| `vfm-sr/train_vfm_sr.py` | VFM baseline for super-resolution |
| `vfm-inpainting/train_vfm_inpaint.py` | VFM baseline for inpainting |
| `vfm-inpainting/vfm_eval24.py` | evaluates VFM inpainting checkpoints |

The VFM scripts train the adapter and the flow map jointly. They serve as the baselines. 

The flow map is the py-meanflow CIFAR-10 checkpoint and is not included, nor is CIFAR-10 itself.
