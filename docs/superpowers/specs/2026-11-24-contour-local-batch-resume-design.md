# Contour local batch pause and recovery

## Intent

A user processing thousands of frames in an interactive Kraken Contour job can pause the batch, close Contour, reopen the same job through **Продолжить работу**, inspect the settings and completed CIFs, and resume only unfinished frames. The same recovery path applies after Contour exits unexpectedly. Recovery is local to the computer that produced the CIFs.

The current incident has 4,846 nonempty CIF files in the first staging attempt for job `cf88251a-f944-4a43-8dc0-fc6b7d864e83`. A new lease created a second, empty staging attempt. Its `job.json` is identical, but neither attempt has a checkpoint; the original recognition settings were held in the process and are no longer recoverable from the job manifest. Preserve the original outputs and make them available to the resumed job without claiming to know the lost settings.

## Desired behavior

- Show **Пауза** while batch recognition is running. Pressing it stops scheduling frames, requests cancellation at a frame boundary, and leaves every completely written CIF intact. If workers have not stopped within ten seconds, terminate them without blocking the UI and recompute any incomplete frame later. The UI shows the count of completed and remaining frames. The user can then press **Продолжить** to process only frames without a valid completed CIF.
- Persist the active recognition configuration before starting work and when changed: the pipeline, recognition settings, save options, and any other inputs that affect generated CIFs. Store it per Kraken job on this machine using an atomic write. A batch resume uses the configuration captured for that batch. If the user changes settings while paused, warn that old and new CIFs will have different processing parameters and require explicit confirmation before continuing. Record which frames belong to each settings snapshot; never label the whole result with only the latest settings.
- On reopening the same job, compare the job manifest and frame identities to the saved local run. Reuse completed CIFs from the previous staging attempt, show them alongside input frames, restore the saved configuration, and open in **На паузе** state. Do not start workers automatically. The resume cursor is the set of verified completed frame outputs, since parallel workers finish out of frame order.
- A crash or force close has the same opening behavior. CIFs are written through temporary files and atomically published when complete; an interrupted or malformed CIF is ignored and recomputed. A missing or corrupt local state file produces a clear recovery error while preserving all original files; it must not silently pretend the job is fresh.
- Returning results to Kraken uses the existing result manifest and import path. A partial return remains possible, but never occurs automatically on pause or reopen. Once the full result is imported, mark the local recovery state complete. Do not delete original attempt data as part of this feature.
- The first restart after this change may show the existing 4,846 CIFs, but must state that their exact original settings cannot be restored, because no settings snapshot was written. Mark their parameter provenance as unknown. The user must explicitly confirm any continuation that would combine those CIFs with newly generated ones. Otherwise they can return the partial result or start a clean run.

## Approach

Selected: local, per-job recovery state that the desktop Agent and Contour both use. The Agent already keeps attempts under `desktop-jobs/<job-id>/staging/<lease-claim>/<job-id>`; it can discover older attempts only after verifying the same job manifest and output names, then make existing CIFs available in the current attempt without altering originals. Contour owns the recognition settings snapshot and pause/resume UI. Reuse the current batch processor and its stop mechanism, adding filtering by completed outputs and a visible paused state; avoid a second processing engine.

Considered: an in-process pause flag alone is simpler but cannot survive a crash or a new lease. Server-backed checkpoints would allow another computer to resume, but add upload and storage of hundreds of megabytes of intermediate CIFs and are outside the agreed local scope.

## Integrity and constraints

- No destructive relocation, overwrite, or trust based on filename alone. Match the immutable job manifest, check that each CIF is a regular file inside the expected output directory, and use an atomic copy or hard link only where safe. If the current attempt already has an output for a frame, resolve the collision explicitly rather than overwriting it.
- Use the existing Contour settings schema and project job identifiers. Do not put machine paths or debug transport data into server-visible job parameters.
- Include a compact, truthful settings history in the returned result metadata: snapshots and the corresponding completed frame IDs, including an `unknown` group for pre-feature results. Preserve the current single-settings representation for jobs that never mix settings.
- Keep the behavior of ordinary standalone Contour and other plugins unchanged.
- Show progress as completed CIFs out of total frames after reopening, not the number of tasks started or the last numeric frame ID.

## Validation

Test pause, resume, crash recovery, mixed completion order, an incomplete output, mismatched manifest, lost settings from the pre-feature job, partial return, and completion. Run focused Contour and Agent tests plus lint/type checks for touched files. Verify the existing incident files read-only before any recovery action; do not run all 4,900 frames merely as a test.
