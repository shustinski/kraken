import multiprocessing
from kraken_hub.composition.frozen_entry import main

multiprocessing.freeze_support()
raise SystemExit(main())
