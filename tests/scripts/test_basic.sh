#!/bin/sh

TESTS_DIR="$1"
BUILD_DIR="$2"
REF_PY="$3"
MY_BIN="$4"
REF_OUT="$5"
MY_OUT="$6"


python ${REF_PY} ${REF_OUT}
${MY_BIN} ${MY_OUT}
${BUILD_DIR}/bin/tbin-dump ${REF_OUT}
${BUILD_DIR}/bin/tbin-dump ${MY_OUT}
${BUILD_DIR}/bin/tbin-diff ${REF_OUT} ${MY_OUT}
