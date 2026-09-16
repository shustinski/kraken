# Build using the shared resource and hook configuration.
import os
from pathlib import Path
os.environ["NEURALIMAGE_BUILD_CLIENT"] = "1"
exec(compile((Path(SPECPATH) / "NeuralImage.spec").read_text(encoding="utf-8"), str(Path(SPECPATH) / "NeuralImage.spec"), "exec"))
