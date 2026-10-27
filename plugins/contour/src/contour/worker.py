"""Headless Agent entry point using Contour's normal image processing pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def run(job_manifest, result_manifest, staging_root):
    import cv2
    import numpy as np

    from contour.application.processing import ContourExtractionSettings, SaveOptions
    from contour.application.use_cases.processing import process_image_path
    from contour.kraken_bridge import ContourKrakenSession

    session = ContourKrakenSession.load(
        job_manifest=job_manifest, result_manifest=result_manifest, staging_root=staging_root
    )
    if session.manifest.operation != "frames.vectorize.v1":
        raise ValueError("Headless Contour currently supports frame vectorization")
    parameters = dict(session.manifest.parameters)
    pipeline_config = parameters.get("pipeline_config", {})
    settings_data = parameters.get("contour_settings", {})
    if not isinstance(pipeline_config, dict) or not isinstance(settings_data, dict):
        raise ValueError("Contour processing parameters must be objects")
    settings = ContourExtractionSettings.from_dict(settings_data)
    cv2.setNumThreads(1)
    for item, source in zip(session.manifest.inputs, session.input_paths, strict=True):
        image = None
        if session.server_source is not None:
            with session.server_source.open(item.relative_path) as stream:
                image = cv2.imdecode(np.frombuffer(stream.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode immutable input: {item.relative_path}")
        result = process_image_path(
            str(source),
            pipeline_config,
            settings,
            output_directory=str(session.output_directory),
            save_options=SaveOptions(save_cif=True),
            source_image=image,
            include_images_in_result=False,
        )
        if result.error:
            raise RuntimeError(f"Contour failed for {source.name}: {result.error}")
    return session.write_result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kraken-job-manifest", default=os.environ.get("KRAKEN_JOB_MANIFEST"))
    parser.add_argument("--kraken-result-manifest", default=os.environ.get("KRAKEN_RESULT_MANIFEST"))
    parser.add_argument("--kraken-staging-root", default=os.environ.get("KRAKEN_STAGING_ROOT"))
    args = parser.parse_args(argv)
    if not all((args.kraken_job_manifest, args.kraken_result_manifest, args.kraken_staging_root)):
        parser.error("Job manifest, result manifest and staging root are required")
    run(Path(args.kraken_job_manifest), Path(args.kraken_result_manifest), Path(args.kraken_staging_root))


if __name__ == "__main__":
    from kraken_core.process_lifetime import bind_child_process_lifetime

    bind_child_process_lifetime()
    main()
