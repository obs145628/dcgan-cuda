#!/bin/sh

TESTS_DIR="$1"
BUILD_DIR="$2"
REF_PY="$3"
MY_BIN="$4"
DATASET="$5"
WEIGHTS="$6"
REF_OUT="$7"
MY_OUT="$8"


python ${REF_PY} ${WEIGHTS} ${REF_OUT} ${DATASET}
${MY_BIN} ${DATASET} ${WEIGHTS} ${MY_OUT}
${BUILD_DIR}/bin/tbin-dump ${REF_OUT}
${BUILD_DIR}/bin/tbin-dump ${MY_OUT}
${BUILD_DIR}/bin/tbin-diff ${REF_OUT} ${MY_OUT}
