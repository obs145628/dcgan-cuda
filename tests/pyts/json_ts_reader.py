import json
import sys

import shell
from test_suite_cmd import TestSuite, TestSuites

class JsonTsReader:

    def __init__(self, path, is_obj = False):

        ts = {}
        ts_list = []
        self.ts = TestSuites()

        data = path
        if not is_obj:
            with open(shell.parse_val(path), 'r') as f:
                data = json.load(f)

        data = shell.parse_val(data)

        for test in data:
            ts_name, test_name = tuple(test['name'].split('.'))
            del test['name']
            if not ts_name in ts:
                ts_list.append(ts_name)
                ts[ts_name] = TestSuite(ts_name)
            ts[ts_name].add_test(test_name, test)

        for name in ts_list:
            self.ts.add_test_suite(ts[name])
    
