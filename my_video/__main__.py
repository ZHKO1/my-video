"""Allow running as: python -m my_video"""

import sys

from my_video.cli.main import main

sys.exit(main())
