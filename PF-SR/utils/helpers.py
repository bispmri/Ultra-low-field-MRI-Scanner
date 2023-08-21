import os
import pathlib
from argparse import ArgumentParser

import yaml


def build_args(default_fname):
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        default=default_fname,
        type=str,
        help="Configuration file",
    )
    # parser.set_defaults(data_path=data_path, default_root_dir=default_root_dir)
    args = parser.parse_args()

    return args


def load_config(config_path, config_name):
    config_file = pathlib.Path(os.path.join(config_path, config_name))
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config
