# Merge-Stable-Diffusion-models-without-distortion
I wrote the permutation spec for Stable Diffusion necessary to merge with the git-re-basin method outlined here - https://github.com/samuela/git-re-basin.
This is based on a 3rd-party pytorch implementation of that here - https://github.com/themrzmaster/git-re-basin-pytorch.

To merge, you may need to install pytorch 1.11.0 or lower (at some point, 1.12.0 did not work but the latest versions of pytorch may have resolved the issue). 

Download the code folder, open cmd in the directory, transfer the desired models to the same folder and run 
"python SD_rebasin_merge.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt"

If not in the same directory then 
pathofmodela.ckpt and pathofmodelb.ckpt instead

## Update on 250302 ##

- *Personal work, not suitable for PR.*
- Added **multithreading** to speed things up. 48 threads will speed up for around 6.75x. However it requries around 70GB of memory to merge.
- **Completely abondon the GPU approach.** The `linear_sum_assignment` will not work in GPU, moving data before after `matmul` will stall everything.
- Embedded original metrics (loss, iter etc) to the progress bar. *Would help analysis the algorithm.* 

### Notes for SDXL by DammK ###
- Tested in A1111 WebUI 1.9.3 and [sd-mecha](https://github.com/ljleb/sd-mecha) 
- [The SDXL code only permutates a few layers.](https://github.com/vladmandic/automatic/blob/dev/modules/merging/merge_rebasin.py)
- [However the full permutation spec is present.](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion/issues/44). [OK, it turns out being entirely different.](https://www.diffchecker.com/WZKq6YiP/) I have posted a [text file](./docs/CLIP_G.txt) describing the **tensor size** of each layer. This implementation requires matching of tensor size.
- **No pruning even it may not crash.** WebUI extensions / nodes will break.
- **No CLIP fix has been revised.** Use other tools instead.
- **Will detect SD1.5 / SD2.1 / SDXL in auto.** 
- SD2.1 will be in partial support: **Both model structure must be identical.** [See this comparasion for differnce (SD2.1 vs WD1.4).](https://github.com/6DammK9/nai-anime-pure-negative-prompt/blob/main/ch03/v1/json/sd2_sd20_wd14.json) Use [toolkit](https://github.com/silveroxides/stable-diffusion-webui-model-toolkit-revisited) to inject the layers
- Then I'll try my best to analysis the effect. Will post to [this article about the algorithm](https://github.com/6DammK9/nai-anime-pure-negative-prompt/blob/main/ch01/rebasin.md) and [my mega mix which is 70+ in 1](https://github.com/6DammK9/nai-anime-pure-negative-prompt/blob/main/ch05/README_XL.MD)
- **~~Bonus task (probably impossible): Implement Algorithm 3 MERGEMANY~~** Probably infeasible, even with [sd-mecha](https://github.com/ljleb/sd-mecha/tree/main)'s well structured codebase. This implementation requires its own layer loading structre for iterlation. 

```sh
python SD_rebasin_merge.py --model_a _x001-sd_xl_base_1.0.safetensors --model_b amp-AstolfoMix-25022801-1458190.safetensors --workers=48
```

- **SDXL will takes hours to merge! 6 minutes per permutation!** Default model name will be `merged.safetensors`.
```log
Applying weighted_sum to theta: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 2515/2515 [00:49<00:00, 51.33it/s]
weight_matching for special_layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1498/1498 [01:12<00:00, 20.60it/s] 
weight_matching for special_layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1498/1498 [01:24<00:00, 17.71it/s]
weight_matching in fp32:  33%|██████████████████████████████████████████████████                                                      | 1/3 [02:38<05:16, 158.01s/it]
weight_matching for special_layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1498/1498 [01:25<00:00, 17.53it/s]
weight_matching for special_layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1498/1498 [01:11<00:00, 21.01it/s]
weight_matching in fp32:  33%|██████████████████████████████████████████████████                                                      | 1/3 [02:37<05:14, 157.23s/it]
Applying weighted_sum to special_keys: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<?, ?it/s]
Main loop: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [1:19:05<00:00, 474.56s/it]

Saving...
Done!
```

- The final result (actually you can derive from paper) is *based from averaging* i.e. $(A\*0.5+B\*0.5)$. However similar to TIES and AutoMBW, it looks better from the plain averaging.

- Both comparasion are "avg / TIES-SOUP / avg(avg+TIES-SOUP) / rebasin(avg+TIES-SOUP)"

![xyz_grid-0841-740330577-8064-1623-3-48-20240428123657.jpg](docs/xyz_grid-0841-740330577-8064-1623-3-48-20240428123657.jpg)
![xyz_grid-0842-740330577-8064-1623-3-48-20240428125432.jpg](docs/xyz_grid-0842-740330577-8064-1623-3-48-20240428125432.jpg)

- Result by revisiting the merge (`x215b` and `x255b`):

![xyz_grid-0000-3501057452-9408-1081-6-48-20250302224301.jpg](docs/xyz_grid-0000-3501057452-9408-1081-6-48-20250302224301.jpg)
![xyz_grid-0001-744089893-9408-1081-6-48-20250302230521.jpg](docs/xyz_grid-0001-744089893-9408-1081-6-48-20250302230521.jpg)