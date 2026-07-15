#!/usr/bin/env python3
# Run a remote command on the GPU box, keeping ssh stdin open (PIPE) to avoid hang.
import subprocess
import sys

if __name__ == "__main__":
    cmd = sys.argv[1]
    p = subprocess.Popen(
        ["/bin/ssh", "aistudio-workflow_56900302-ssctl", cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    data = p.stdout.read()
    rc = p.wait()
    sys.stdout.write(data)
    sys.exit(rc)
