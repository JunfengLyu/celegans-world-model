# Learning and behavioral physics in virtual *C. elegans*: A phase–amplitude hybrid

The three H.264 originals are in [`demos/`](demos) and stream on the [project's GitHub Pages site](https://junfenglyu.github.io/celegans-world-model/). The analysis and plotting scripts below reproduce the figures and movies used in the essay.

## Build a local copy of the manuscript

If you have the local manuscript files (`paper.tex` and `references.bib`), run from this directory:

```sh
tectonic -X compile paper.tex
```

The source uses REVTeX 4.2 and BibTeX. Keep the figure and demo subdirectories with the source. The three video titles in Appendix A open PDF attachments, and the linked five-frame strip opens the hosted copy of Movie S3.

## Reproduce the figures and movies

From the parent project directory, use its existing Python environment and raw inputs. The parent README describes downloading the archived posture and connectome data, fitting the reference models, and running modWorm. Install the CMasher figure dependency, then run:

```sh
.venv/bin/python -m pip install cmasher==1.9.2
.venv/bin/python -m wormlab.hybrid
.venv/bin/python essay/scripts/connectome_figure.py --movie
.venv/bin/python essay/scripts/virtual_dynamics_figure.py
.venv/bin/python essay/scripts/behavior_figure.py
.venv/bin/python essay/scripts/hybrid_figure.py
.venv/bin/python essay/scripts/embedding_dynamics.py
.venv/bin/python essay/scripts/eigenworm_phase_demo.py
.venv/bin/python essay/scripts/hybrid_demo.py
.venv/bin/python essay/scripts/video_snapshots.py
```

The matched fits, checkpoints, per-worm metrics, and paired rollouts are generated under `artifacts/hybrid/` at the project root. The compact [atlas](data/virtual-worm-atlas.json) records both upstream source revisions, with [attribution and license terms](data/ATLAS-NOTICE.txt). The atlas extraction script accepts the source VRML, c302 cell morphology directory, and modWorm neuron indices.

| Output | Inputs |
|---|---|
| `connectome.pdf` | Atlas registration and matched modWorm posterior-touch and sham outputs |
| `virtual-dynamics.pdf` | Simulated voltage and forces, the benchmark GRU and RSSM checkpoints, and a held-out observed sequence |
| `behavior.pdf` | Public posture time series, an atlas cuticle profile, and the fitted delay-Markov matrix |
| `hybrid.pdf` and `embedding-dynamics.pdf` | Matched GRU forecasts, distributions, and first-window examples rendered with the same cuticle profile |
| `video-snapshots.pdf` | Five decoded frames from the released `hybrid-rollout.mp4` |
