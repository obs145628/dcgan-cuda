import os

COLOR_DEFAULT = '\x1b[39m'
COLOR_BLUE = '\x1b[34m'
COLOR_RED = '\x1b[31m'
COLOR_GREEN = '\x1b[32m'


def parse_val(val):
    if isinstance(val, list):
        return [parse_val(x) for x in val]
    if isinstance(val, dict):
        res = {}
        for k in val:
            res[k] = parse_val(val[k])
        return res
    
    if not isinstance(val, str):
        return val

    return val.format(
        BUILD_DIR =  os.environ.get('PYL_BUILD_DIR', ''),
        ROOT_DIR = os.environ.get('PYL_ROOT_DIR', '')
    )


class NullStream:

    def write(self, str):
        return

    def flush(self):
        return
