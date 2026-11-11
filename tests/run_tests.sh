#!/usr/bin/env bash
# Test suite: torch reference path (test_gates.py) + fused Triton path (test_triton.py).
# Run: HIP_VISIBLE_DEVICES=0 tests/run_tests.sh   (exit 0 only if both pass)
cd "$(dirname "$0")/.."
export HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH:-$PWD/hipblaslt-lib/gfx1201}
rc=0
for t in tests/test_gates.py tests/test_triton.py; do
    echo "=== $t"
    .venv/bin/python "$t" || rc=1
done
echo "=== suite $([ $rc = 0 ] && echo PASSED || echo FAILED)"
exit $rc
