"""Enable `python -m coconut_experiment.cheatchain [...]` (avoids the runpy double-import warning)."""

from .generate import main

if __name__ == "__main__":
    main()
