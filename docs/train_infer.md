# Train & inference
## Train
You can train the model as follows (by default, training uses 2×8 GPUs; you can adjust this based on your compute resources. Be sure to update the GPU count in the config file to avoid unexpected iteration numbers):

1. ColaVLA
```bash
bash launch_train.sh projects/configs/colavla_epoch10_msarplv2_ms6_seqformer_wckpt_fullcontext_regw80_pretraintraj_globalreason_top3pred.py 8 2
```

2. SOLVE

```bash
bash launch_train.sh projects/configs/colavla_epoch10_msarplv2_ms6_seqformer_wckpt_fullcontext_regw80_pretraintraj_globalreason_top3pred.py 8 2
```


## Evaluation
**1. OpenLoop Planning**

We use ColaVLA as the main example. The process has two steps: first run model inference and save predicted trajectories locally, then compute metrics using [evaluation files](../evaluation/eval_planning_pkl.py). We use 8 GPUs for inference by default.

```bash
bash launch_test.sh projects/configs/colavla_epoch10_msarplv2_ms6_seqformer_wckpt_fullcontext_regw80_pretraintraj_globalreason_top3pred.py ckpts/colavla_iter8790.pth 8 --format-only
```

**2. ClosedLoop Planning**

First, prepare the required folders and checkpoints.

```bash
cd /path/to/ColaVLA
cd ..
git clone https://github.com/atonderski/neuro-ncap.git
git clone https://github.com/georghess/neurad-studio.git
cd ColaVLA
ln -s ../neuro-ncap neuro-ncap
ln -s ../neurad-studio neurad-studio
```

Then follow the [NeuroNcap](https://github.com/atonderski/neuro-ncap) documentation to download the required checkpoints and dependencies. You can still refer to the package versions in our [environment.yml](./environment.yml).

Finally, run the provided scripts directly. We additionally provide a closed-loop testing script that can run on a single local GPU; related files are in `./inference_closed_loop`.
The corresponding command is:

```bash
bash inference_closed_loop/run_colavla_vla.sh inference_closed_loop/configs/inference_vla.py [YOUR_CHECKPOINTS] 50
```

The results will be saved in `neuro-ncap/output`. Please use the scripts in [NeuroNcap](https://github.com/atonderski/neuro-ncap) to compute the final closed-loop metrics.
