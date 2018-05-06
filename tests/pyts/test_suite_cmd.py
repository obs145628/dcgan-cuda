import sys

from commander import Commander
import shell

def get_output(arr, max_size):
    size = min(len(arr), max_size)
    arr = arr[:size]
    str = arr.decode('ascii')
    if (size == max_size):
        str += '...'
        str = str.replace('\n', '$\n')
    elif len(arr) == 0:
        str = '((EMPTY))\n'

    return str

class TestSuite:

    def __init__(self, name):
        self.name = name
        self.tests = []
        self.cmd = Commander()
        self.res = []

    def add_test(self, name, params):
        self.tests.append((name, params))

    def run(self):
        line_length = max([len(x[0]) for x in self.tests]) + 2

        self.out.write("Running test suite " + self.name + '\n')

        ntests = len(self.tests)
        nvalids = 0
        
        for test in self.tests:
            self.out.write(test[0] + "... " + (" " * (line_length - len(test[0]))))
            self.out.flush()

            valid, errs, proc = self.cmd.run_test(**test[1])

            if valid:
                self.out.write(shell.COLOR_GREEN + "[OK]\n" + shell.COLOR_DEFAULT)
                nvalids += 1
            else:
                self.out.write(shell.COLOR_RED + "[KO]\n")
                for e in errs:
                    self.out.write(' [{}={}|{}]'.format(e[0], e[1], e[2]))
                self.out.write(shell.COLOR_DEFAULT + '\n')
            self.res.append((valid, errs, proc))

        per = nvalids / ntests * 100

        self.out.write("Test suite {}: {} / {} ({:.2f}%)\n".format(self.name, nvalids, ntests, per))
        self.out.flush()

        self.nvalids = nvalids
        self.ntests = ntests

class TestSuites:

    def __init__(self):
        self.ts = []
        self.out = sys.stdout
        self.out_err = sys.stdout
        self.out_err = shell.NullStream()

    def add_test_suite(self, ts):
        self.ts.append(ts)
        ts.out = sys.stdout

    def set_run_init(self, run_init):
        for ts in self.ts:
            ts.cmd.run_init = run_init
        if run_init == True:
            print('Running initialization')

    def run(self):

        ntests = 0
        nvalids = 0
        
        for t in self.ts:
            t.run()
            self.out.write('\n')
            ntests += t.ntests
            nvalids += t.nvalids

        per = nvalids / ntests * 100

        self.out.write("Global results: {} / {} ({:.2f}%)\n".format(nvalids, ntests, per))

        self.nvalids = nvalids
        self.ntests = ntests
        self.out.flush()

        for ts in self.ts:
            for i in range(len(ts.tests)):
                test = ts.tests[i]
                res = ts.res[i]
                if res[0]:
                    continue
                self.out_err.write(ts.name + '.' + test[0])
                for e in res[1]:
                    self.out_err.write(' [{}={}|{}]'.format(e[0], e[1], e[2]))
                self.out_err.write('\n')
                self.out_err.write('EXIT CODE = ' + str(res[2].returncode) + '\n')
                self.out_err.write('======== STDOUT ========\n')
                self.out_err.write(get_output(res[2].stdout, 10000))
                self.out_err.write('========================\n')
                self.out_err.write('======== STDERR =========\n')
                self.out_err.write(get_output(res[2].stderr, 10000))
                self.out_err.write('========================\n')
                self.out_err.write('\n\n')
                self.out_err.flush()

        
        
