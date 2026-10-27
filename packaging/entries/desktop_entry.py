import multiprocessing
from kraken_hub.frozen_entry import main

multiprocessing.freeze_support()
raise SystemExit(main())
