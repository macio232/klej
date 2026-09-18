set -eu

mkdir -p markers
printf '%s\n' "host=$(hostname) pid=$$ time=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> markers/wrapper.log

search=$PWD
while [ "$search" != / ]; do
    if [ -x "$search/.venv/bin/python" ]; then
        PATH="$search/.venv/bin:$PATH"
        export PATH
        break
    fi
    search=$(dirname "$search")
done

if ! python -c 'import dvc' >/dev/null 2>&1; then
    echo "smoke-test wrapper: Python cannot import DVC; activate or load the compute-node environment in wrapper.sh" >&2
    exit 1
fi

exec sh -c "$1"
