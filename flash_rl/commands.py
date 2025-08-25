import argparse
import logging
import os
from dataclasses import asdict

import yaml

from .configs import get_default_config
from .flash_quantization import profiling_fp8, profiling_int8

logger = logging.getLogger(__name__)

def setup_flashrl(name, parser):
    subparser = parser.add_parser(
        name,
        description="setup Flash RL",
        help='setup Flash RL',
    )
    subparser.add_argument(
        '-m', '--model',
        required=False,
        type=str,
        default=None,
        help='path to the quantized model',
    )
    subparser.add_argument(
        '-p', '--profile',
        required=False,
        type=str,
        default=None,
        help='path to the profile file',
    )
    subparser.add_argument(
        '--fn',
        required=False,
        choices=['fp8', 'fp8_fast', 'fp8_vllm', 'fp8_vllm_fast', 'fp8_channel', 'fp8_tensor', 'int8', 'int8_fast', 'int8_wo_prune', 'int8_prune', 'bf16'],
        default='int8',
        help='quantization function to use',
    )
    subparser.add_argument(
        '-o', '--config-output',
        required=False,
        type=str,
        default=None,
        help='path to save the config file',
    )
    subparser.add_argument(
        '-a', '--append',
        action='store_true',
        help='append the config to the existing file',
    )
    subparser.add_argument(
        'columns',
        nargs=argparse.REMAINDER,
        help=[
            'other parameters for online quantization, format is'
            'distributed_executor_backend=\"external_launcher\"'
            'module_attribute_to_preserve=[\"workspace\"]'
            'params_to_ignore=[\"q_scale\"]'
        ],
    )
    subparser.set_defaults(func=setup_flashrl_runner)
    return subparser

def setup_flashrl_env() -> bool:
    """Install a usercustomize.py snippet to auto-import flash_rl.

    - Uses the user site-packages (safer than global site-packages).
    - Writes a proper multi-line try/except block.
    - Idempotent (skips if already present).
    - Permission-safe with clear warnings.

    Returns True if the snippet exists or was written successfully.
    """
    import site

    try:
        user_site = site.getusersitepackages()
    except Exception as e:
        logger.warning("flash-rl: Could not resolve user site-packages: %s", e)
        return False

    try:
        os.makedirs(user_site, exist_ok=True)
    except OSError as e:
        logger.warning("flash-rl: Cannot create user site dir %s: %s", user_site, e)
        return False

    path = os.path.join(user_site, 'usercustomize.py')

    # Proper multi-line try/except snippet
    snippet = (
        "try:\n"
        "    import flash_rl\n"
        "except ImportError:\n"
        "    pass\n"
    )

    # Idempotence: skip if already imported
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'import flash_rl' in content and not any(
                line.strip().startswith('#') and 'import flash_rl' in line for line in content.splitlines()
            ):
                logger.info("flash-rl: usercustomize.py already imports flash_rl; skipping.")
                return True
    except OSError:
        # Continue to attempt append
        pass

    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(snippet)
        logger.info("flash-rl: wrote autoload snippet to %s", path)
        return True
    except OSError as e:
        logger.warning(
            "flash-rl: unable to write usercustomize.py at %s: %s; "
            "Flash-RL still works if you import flash_rl manually.",
            path, e,
        )
        return False

def setup_flashrl_runner(args):
    if args.config_output is None:
        logger.info("No config output path provided, using default: ~/.flashrl_config.yaml")
        args.config_output = os.path.expanduser("~/.flashrl_config.yaml")

    config_data = asdict(get_default_config(args.fn))

    assert args.model is not None or args.fn in ['bf16', 'fp8_vllm', 'fp8', 'fp8_fast', 'fp8_vllm_fast'], \
        f"model path is required for quantization {args.fn}"

    if args.model is not None:
        config_data['model'] = args.model
    if args.profile is not None:
        config_data['profile'] = args.profile
    config_data['fn'] = args.fn
    for column in args.columns:
        key, value = column.split('=')
        config_data[key] = eval(value)

    assert config_data['load_format'] == 'auto' or args.fn in ['fp8', 'fp8_tensor', 'fp8_channel'], \
        f"load_format should be 'auto' for {args.fn}, but got {config_data['load_format']}"

    if args.append and os.path.exists(args.config_output):
        with open(args.config_output, 'r') as fin:
            meta_configs = yaml.safe_load(fin)
        meta_configs['configs'].append(config_data)
    else:
        meta_configs = {'configs': [config_data]}

    with open(args.config_output, 'w') as fout:
        yaml.dump(
            meta_configs,
            fout,
        )

    logger.info(f"FlashRL config saved to {args.config_output}")
    setup_flashrl_env()

    logger.info(
        f"FlashRL profile saved to {args.config_output}, "
        f"set FLASHRL_CONFIG={args.config_output} to enable it."
    )

def clean_up_flashrl(name, parser):
    subparser = parser.add_parser(
        name,
        description="setup Flash RL",
        help='setup Flash RL',
    )
    subparser.add_argument(
        '-p', '--path',
        required=False,
        type=str,
        default=None,
        help='path to save the config file',
    )
    subparser.set_defaults(func=cleanup_flashrl_runner)
    return subparser

def cleanup_flashrl_runner(args):
    if args.path is None:
        import site
        # Prefer user site where we wrote the snippet
        user_site = site.getusersitepackages()
        uc_path = os.path.join(user_site, 'usercustomize.py')
        if os.path.exists(uc_path):
            with open(uc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            need_write = False
            for i in range(len(lines)):
                if 'import flash_rl' in lines[i]:
                    l_processed = lines[i].strip()
                    if not l_processed.startswith("#"):
                        # Case 1: plain 'import flash_rl'
                        if l_processed == 'import flash_rl':
                            lines[i] = f"# {lines[i]}"
                            logger.info("flash_rl setup removed from usercustomize.py")
                            need_write = True
                        # Case 2: legacy one-line try format
                        elif l_processed == 'try: import flash_rl':
                            lines[i] = f"# {lines[i]}"
                            if i + 1 < len(lines):
                                lines[i+1] = f"# {lines[i+1]}"
                            logger.info("flash_rl setup removed from usercustomize.py")
                            need_write = True
                        # Case 3: new multi-line try/except block
                        elif l_processed == 'try:' and i + 3 < len(lines):
                            next1 = lines[i+1].strip()
                            next2 = lines[i+2].strip()
                            next3 = lines[i+3].strip()
                            if next1 == 'import flash_rl' and next2.startswith('except ImportError') and next3 == 'pass':
                                lines[i] = f"# {lines[i]}"
                                lines[i+1] = f"# {lines[i+1]}"
                                lines[i+2] = f"# {lines[i+2]}"
                                lines[i+3] = f"# {lines[i+3]}"
                                logger.info("flash_rl setup removed from usercustomize.py")
                                need_write = True
                        else:
                            logger.warning(
                                "flash_rl setup found in usercustomize.py, but not removed due to unknown format. "
                                f"Please remove the command {lines[i]} manually from {uc_path}"
                            )

            if need_write:
                with open(uc_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

def profile_flashrl(name, parser):
    subparser = parser.add_parser(
        name,
        description="setup Flash RL",
        help='setup Flash RL',
    )
    subparser.add_argument(
        '-m', '--model',
        required=False,
        type=str,
        help='path to the original model',
    )
    subparser.add_argument(
        '-q', '--quantized',
        required=True,
        type=str,
        help='path to the quantized model',
    )
    subparser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='path to save the profile file',
    )
    subparser.add_argument(
        '--fn',
        choices=['fp8', 'int8'],
        default='int8',
    )
    subparser.set_defaults(func=profile_runner)
    return subparser

def profile_runner(args):
    if args.fn == 'int8':
        assert args.model is not None, f"model path is required for quantization {args.fn}"
        profiling_int8(args.model, args.quantized, args.output)
    else:
        profiling_fp8(args.quantized, args.output)

def run():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
        "setup": setup_flashrl,
        "cleanup": clean_up_flashrl,
        "profile": profile_flashrl,
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand(name, subparsers)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()

