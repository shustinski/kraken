# Managed NeuralImage training

`dataset.model.train.v1` retains its interactive default. A registered process
advertising both modes can also accept `execution_mode: headless`; pipeline
training dispatch explicitly selects headless. An Agent rejects a mode absent
from its process registration. Interactive jobs retain their waiting-for-user
state and window.

The input is an immutable, SHA-pinned ZIP published as a dataset in the selected
project/layer. Manual publishers use `artifact_role: dataset`; plugin publishers
use the existing `plugin_output_role`. Conflicting roles are rejected. This
classification grants no permissions and does not manufacture a plugin run.

Example operation parameters:

```json
{
  "execution_mode": "headless",
  "dataset_version_id": "<published-version-uuid>",
  "training": {
    "epochs": 5,
    "seed": 20261025,
    "validation_relative_root": "validation",
    "settings": {
      "model": "S 660k",
      "train_patch_size": [512, 512],
      "use_validation": true,
      "validation_source": "external",
      "early_stopping_enabled": true,
      "dataloader_num_workers": 0
    }
  }
}
```

The ZIP contains `images/` and matching `cif/` files. External validation uses
`validation/images/` and `validation/cif/` in the same verified ZIP. Full image
hash comparison rejects train/validation overlap even after renaming. External
validation currently requires staged input; streamed input supports the existing
deterministic internal split. A missing external partition never silently falls
back to training images.

Training uses the existing workflow mapper, GeneralNeuralHandler and trainer.
`training.epochs` is a positive upper bound, including automatic early stopping.
The existing automatic policy derives its check frequency and patience from the
dataset; headless manifests cannot override its patience/min-delta/restore policy.
The existing interactive policy remains unchanged when no explicit cap is supplied.
Path overrides and custom generators are rejected; supported settings are listed
in `managed_training.HEADLESS_SETTINGS` and validated by the existing settings DTO.

ZIP preflight bounds members to 100000 and expanded size to 64 GiB by default.
It rejects duplicate/case-colliding paths, file/directory conflicts, unsafe paths
and symbolic links. Hashing and ZIP reading use the same verified file handle.
Extraction uses exclusive writes and checks directory reparse points. These
checks are not a sandbox against a native plugin with the same OS identity.

Success requires the trainer's actual completion signal and a model inside its
managed output directory. The normal model publisher returns immutable model
bytes and workflow provenance, including seed, dataset SHA, external partition
hash/counts, Torch/CUDA runtime, deterministic-algorithm status and epoch upper
bound. A fixed seed alone does not certify bit-identical GPU execution.

`test_neuralimage_headless_training.py` includes an opt-in real training process:

```powershell
$env:KRAKEN_RUN_TRAINING_LIVE='1'
.venv/Scripts/python.exe tools/verify_repository.py tests/test_neuralimage_headless_training.py --timeout 240 --report .test_runtime/managed-training
```

The small fixture validates transport and actual checkpoint production. Model
quality requires separate heldout evaluation; successful training is not an
acceptance of semiconductor geometry or electrical connectivity.
