"""Allow ``python -m citation`` as a shortcut for ``python -m citation.cli``."""

from citation.cli import main

if __name__ == "__main__":
    main()
