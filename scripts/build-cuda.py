#!/usr/bin/env python3
"""Build pinned llama.cpp CUDA libraries and Abel's native training tools."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='7ceed8737fdb4eb09b4760e77bd12d38012de5a8'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--arch',default='native',help='CUDA architecture, e.g. 75 for T4, 89 for L4')
    a=p.parse_args()
    if not 1<=a.jobs<=32:p.error('jobs must be 1..32')
    if a.arch!='native' and (not a.arch.isdigit() or not 50<=int(a.arch)<=150):p.error('invalid CUDA architecture')
    for name in ('nvcc','cmake','make','git'):
        if not shutil.which(name):raise SystemExit(f'{name} missing; select a CUDA GPU runtime and install build dependencies')
    source=ROOT/'vendor/llama.cpp'
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=PIN:raise SystemExit('llama.cpp revision differs from pinned training backend')
    subprocess.run(['cmake','-S',str(source),'-B',str(source/'build-cuda'),
        '-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=ON','-DGGML_NATIVE=OFF',
        '-DGGML_RPC=OFF','-DGGML_BACKEND_DL=OFF',f'-DCMAKE_CUDA_ARCHITECTURES={a.arch}',
        '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_SERVER=OFF'],check=True)
    subprocess.run(['cmake','--build',str(source/'build-cuda'),'--target','llama','-j',str(a.jobs)],check=True)
    subprocess.run(['make','-B',f'-j{a.jobs}','LLAMA_LIBDIR=vendor/llama.cpp/build-cuda/bin',
                    'distill','native'],cwd=ROOT,check=True)
if __name__=='__main__':main()
