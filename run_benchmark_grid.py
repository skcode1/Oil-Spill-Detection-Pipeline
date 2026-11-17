import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
checkpoints = ['models/oil_spill_segmentation_model.pth.bak', 'models/oil_spill_segmentation_model.pth']
normalisations = ['per_tile', 'scene']
thresholds = '0.50,0.70,0.80,0.90'
scenes = 15
downscale = 2
tile_size = 512
stride = 256
results_dir = ROOT / 'results'
results_dir.mkdir(exist_ok=True)
total_configs = len(checkpoints) * len(normalisations)
print(f'Running {total_configs} configurations...')
print(f'Each config: {scenes} scenes/class x 3 classes = {scenes * 3} scenes')
print(f'Estimated time: ~15s/scene x {scenes * 3} scenes = ~{scenes * 3 * 15 / 60:.0f} min per config')
print(f'Total estimated: ~{scenes * 3 * 15 * total_configs / 60 / 60:.1f} hours\n')
for i, ckpt in enumerate(checkpoints):
    for j, norm in enumerate(normalisations):
        config_num = i * len(normalisations) + j + 1
        ckpt_name = Path(ckpt).stem
        output_file = results_dir / f'model_metrics_{ckpt_name}_{norm}.json'
        print(f"\n{'=' * 60}")
        print(f'Config {config_num}/{total_configs}: {ckpt_name} + {norm}')
        print(f"{'=' * 60}")
        cmd = [sys.executable, 'scratch_eval_checkpoint.py', ckpt, '--scenes', str(scenes), '--thresholds', thresholds, '--tile-size', str(tile_size), '--stride', str(stride), '--normalisation', norm, '--downscale', str(downscale), '--output', str(output_file)]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
        if result.returncode != 0:
            print(f'[!] Config {config_num} failed with return code {result.returncode}')
        else:
            print(f'[+] Config {config_num} completed, results in {output_file}')
print('\n\nAll configurations completed!')
print('Results saved to results/model_metrics_*.json')