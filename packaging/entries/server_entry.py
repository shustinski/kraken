from multiprocessing import freeze_support
import sys

from kraken_server.__main__ import main

freeze_support()
if getattr(sys, 'frozen', False):
    from kraken_manager.infrastructure.compute_runtime import initialize_packaged_compute_runtime
    initialize_packaged_compute_runtime()
raise SystemExit(main())
