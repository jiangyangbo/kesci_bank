#!/usr/bin/env python
import sys

import os

import config

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'garnet.settings')
    try:
        from django.core.management import execute_from_command_line

        config.server = True

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
