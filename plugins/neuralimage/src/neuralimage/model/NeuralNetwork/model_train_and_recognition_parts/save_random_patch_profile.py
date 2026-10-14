"""model_train_and_recognition: save random patch profile."""
from __future__ import annotations
import csv


class SaveRandomPatchProfileMixin:
    def _save_random_patch_profile_csv(self) -> None:
        rows = getattr(self, '_random_patch_profile_rows', [])
        if not rows:
            return
        profiler_config = getattr(self, '_training_profiler_config', None)
        output_dir_name = getattr(profiler_config, 'output_dir_name', 'profiles')
        profile_dir = self._save_path.parent / output_dir_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        detail_path = profile_dir / 'random_patch_batches.csv'
        with detail_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        grouped: dict[tuple[int, int], list[dict[str, int | float]]] = {}
        for row in rows:
            grouped.setdefault((int(row['width']), int(row['height'])), []).append(row)
        summary_path = profile_dir / 'random_patch_shapes.csv'
        timing_names = ('data_wait_ms', 'augmentation_ms', 'forward_ms', 'backward_ms', 'optimizer_ms', 'total_ms')
        with summary_path.open('w', encoding='utf-8', newline='') as stream:
            fieldnames = ['width', 'height', 'batches', *[f'avg_{name}' for name in timing_names]]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for (width, height), shape_rows in sorted(grouped.items()):
                writer.writerow({
                    'width': width,
                    'height': height,
                    'batches': len(shape_rows),
                    **{
                        f'avg_{name}': sum(float(row[name]) for row in shape_rows) / len(shape_rows)
                        for name in timing_names
                    },
                })
