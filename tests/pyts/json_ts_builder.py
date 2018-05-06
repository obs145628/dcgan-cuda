import json

import shell


class JsonTsBuilder:

    def __init__(self):
        self.tests = []

    def add_test(self, name, sub_name, cmd, code = None, stdout_file = None, stderr_file = None,
                 has_stdout = None, has_stderr = None, cwd = None, env = None, before = None,
                 init = None):
        obj = {}
        obj['name'] = name.replace('.', '_') + '.' + sub_name.replace('.', '_')
        obj['cmd'] = cmd

        if code != None:
            obj['code'] = code
        if stdout_file != None:
            obj['stdout_file'] = stdout_file
        if stderr_file != None:
            obj['stderr_file'] = stderr_file
        if has_stdout != None:
            obj['has_stdout'] = has_stdout
        if has_stderr != None:
            obj['has_stderr'] = has_stderr
        if cwd != None:
            obj['cwd'] = cwd
        if env != None:
            obj['env'] = env
        if before != None:
            obj['before'] = before
        if init != None:
            obj['init'] = init
        self.tests.append(obj)

    def save(self, path):
        with open(shell.parse_val(path), 'w') as f:
            f.write(json.dumps(self.tests))
