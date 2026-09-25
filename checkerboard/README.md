# 2D checkerboard

Reproduction of the VFM checkerboard experiment, and the normalising flow adapter on
the same problem. Run the notebooks in order.

| notebook | contents |
|---|---|
| `01_checkpoint_sanity.ipynb` | checks the pretrained flow maps |
| `02_train_baselines_and_vfm.ipynb` | trains three noise adapters, differing in whether the flow map is frozen, trained jointly, or trained with VFM's mean-flow constraint |
| `03_metrics.ipynb` | NLPD, CRPS, SACC and MMD for those three methods at K=1 and K=4 |
| `04_nf_adapter.ipynb` | the normalising flow adapter, with the flow map frozen |

Notebook 02 documents two defects in the released `train_vfm.py` and the fixes used here.

The pretrained flow maps (`fm_model.pt`, `mf_model.pt`) come from the VFM authors' repository
and are not included. Checkpoints written by these notebooks are not included either.
