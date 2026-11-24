# Twig Capture: merge suggestions in the browser

![Twig Selector demo: an undertraced BANC neuron (SNpp47) is queued, merges are predicted in the browser, and the confident candidates are staged for review](docs/twig_selector_demo.gif)

*BANC SNpp47 `720575941544177124` has 100 L2 nodes, against a median of 150
for the 30 other neurons of its type. The neuron is queued in the **Twig
Selector**, and merge candidates are predicted on this laptop's GPU. The
clock shows real time: 13:05 for all 57 regions. The GIF is a time-lapse
while the model runs. The 63 candidates at p ≥ 0.9 are then staged with
**Stage all**. The cursor rests on **Merge staged** without clicking it.*

## Links

| dataset | viewer |
|---|---|
| BANC | https://radagast.hms.harvard.edu/twigcapture_banc/ |
| Aedes | https://radagast.hms.harvard.edu/twigcapture_aedes/ |
| CRANTb | https://radagast.hms.harvard.edu/twigcapture_crantb/ |

Each link opens its dataset, with the Twig Capture tab connected to the
cluster results of that dataset's newest model, served beside the viewer
(`twigcapture_<dataset>/results/`). Sign in to CAVE when asked (CRANTb has
its own login).

**Results folders.** On O2, `twig_capture/viewers/results_<dataset>/` holds
links to the newest model's result files plus `index.json` and
`candidates.json.gz` (`make_results_index.py`, for the root picker and "All
neurons") and `neurons.json` (`make_seatable_meta.py`, for the SeaTable
filters). Today: BANC `banc_v2.0` (banc_finetune 2026-11-13), Aedes
`aedes_v2.0_test` (aedes_finetune 2026-11-13), CRANTb
`crantb_test_aedes_v2.1`. When a newer model's results land, relink the
folder and re-run both scripts.

## Predicting merges for your own neurons

1. Select the segmentation layer and open the **Twig Selector** tab.
2. Queue neurons in any of these ways:
   - **Add visible segments**;
   - paste root ids;
   - point at a neuron and press **Alt+T** (Option+T on a Mac). This queues
     the neuron and starts the model.
3. Press **Run**. Every queued neuron runs in turn, on the GPU (WebGPU)
   where there is one, otherwise on the CPU (slower, but it works).
4. **Open in Twig Capture** shows the suggestions. To work on several
   neurons at once, tick them and use **Open ticked together**.
5. Review as usual: **Merge**, **Not a merge**, **Unsure**, **New neuron**,
   **Split needed**. Or **Stage all → Merge staged**: staging edits nothing,
   so untick what you don't want before merging.

Everything runs and is cached in your browser: the model is downloaded once,
and results survive a reload. Verdicts are still recorded in the dataset's
`twig_capture_decisions` CAVE table, as for cluster results.

**Level.** Level 1 suggests merges onto the neuron. Level 2 also suggests
merges onto the confident level-1 suggestions (p ≥ the chain threshold),
and so on. Those rows say which suggestion they attach to ("via").

**Speed.** On an Apple M-series GPU it takes about 10–15 s per region, so a
few minutes for a small neuron and longer for a big one. On the CPU, expect
about 3 s per patch. There, set sites/pair to 1 and cap max regions.

## Something went wrong?

Press **Copy diagnostics**, or **Copy report** next to a failed neuron, and
paste the text into a message to us. It holds your browser and OS, GPU
details, the settings, the error and a short run log. It contains no login
tokens.

## For developers

See [BRIEF.md](BRIEF.md) for how it works, what has been measured against
the cluster runner, and how to rebuild the model. `deploy/deploy.sh`
builds and deploys the three viewers.
