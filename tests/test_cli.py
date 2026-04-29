from my_video.cli.main import main

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from my_video.cli import exit_codes as EXIT

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
