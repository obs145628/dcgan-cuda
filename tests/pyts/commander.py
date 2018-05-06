import os
import subprocess

import ioutils


_init_hooks_ = []
_before_hooks_ = []
_run_hooks_ = []

def hook_init(f): _init_hooks_.append(f)
def hook_before(f): _before_hooks_.append(f)
def hook_run(f): _run_hooks_.append(f)

def ebool(x):
    if x:
        return 'T'
    else:
        return 'F'

OUT_CMDS = {
    'eq': lambda x, y: x == y
}

class Commander:


    def __init__(self):
        self.use_valgrind = False
        self.timeout = -1
        self.run_init = False


    def run_cmd(self, args, read_stdout = True, read_stderr = True):
        run_dict = { 'args': args }
        if read_stderr:
            run_dict['stdout'] = subprocess.PIPE
        if read_stderr:
            run_dict['stderr'] = subprocess.PIPE
    
        return (res.returncode, res.stdout, res.stderr)

    def check_output(out_str, ref_str, output_cmp):
        return OUT_CMDS[output_cmp](out_str, ref_str)

    '''
    Returns tuple (valid, errs, res)
    valid: bool, true if no errors
    errs: string errors
    res: subprocess object
    '''
    def run_test(self, cmd, cwd = None, code = None, env = None,
                 has_stdout = None, stdout = None, stdout_file = None,
                 has_stderr = None, stderr = None, stderr_file = None,
                 output_cmp = 'eq', before = [], init = []):
        
        if env != None:
            new_env = os.environ.copy()
            for key in env.keys():
                new_env[key] = env[key]
            env = new_env

        errs = []

        for f in _init_hooks_: f(self)
        if not self.run_init:
            init = []
        for init_cmd in init:
            sub_res = subprocess.run(init_cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
                             cwd = cwd, env = env)
            sub_code = sub_res.returncode
            if sub_code != 0:
                errs.append(('INIT_CODE', sub_code, 0))
                print(sub_res.stderr)

        for f in _before_hooks_: f(self)
        for prev_cmd in before:
            sub_res = subprocess.run(prev_cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
                             cwd = cwd, env = env)
            sub_code = sub_res.returncode
            if sub_code != 0:
                errs.append(('BEFORE_CODE', sub_code, 0))
                print(sub_res.stderr)

        for f in _run_hooks_: f(self)
        res = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
                             cwd = cwd, env = env)
        cmd_code = res.returncode
        cmd_stdout = res.stdout
        cmd_stderr = res.stderr

        if code != None and code != cmd_code:
            errs.append(('CODE', cmd_code, code))

        if has_stdout != None and (len(cmd_stdout) == 0) == has_stdout:
            errs.append(('HAS_STDOUT', ebool(len(cmd_stdout)), ebool(has_stdout)))

        if stdout != None and not self.check_output(cmd_stdout.decode('ascii'), stdout, output_cmp):
            errs.append(('STDOUT', '.', '.'))

        if stdout_file != None and not ioutils.file_content_is(cmd_stdout, stdout_file):
            errs.append(('STDOUT_FILE', '.', '.'))

        if has_stderr != None and (len(cmd_stderr) == 0) == has_stderr:
            errs.append(('HAS_STDERR', ebool(len(cmd_stderr)), ebool(has_stderr)))

        if stderr != None and not self.check_output(cmd_stderr.decode('ascii'), stderr, output_cmp):
            errs.append(('STDERR', '.', '.'))

        if stderr_file != None and not ioutils.file_content_is(cmd_stderr, stderr_file):
            errs.append(('STDERR_FILE', '.', '.'))

        return (len(errs) == 0, errs, res)
