import glob
import json
import os
import subprocess
from pathlib import Path

import filelock

import humming.jit.utils as jit_utils


class Compiler(object):
    @classmethod
    def signature():
        raise NotImplementedError

    @staticmethod
    def humming_include_dir():
        dirname = os.path.dirname(__file__)
        dirname = os.path.abspath(dirname + "/../include/")
        return dirname

    @staticmethod
    def include_dirs():
        return [Compiler.humming_include_dir()]

    @staticmethod
    def cuh_last_update_time():
        dirname = Compiler.humming_include_dir()
        data = {}
        for filename in sorted(glob.glob(f"{dirname}/**/*.cuh", recursive=True)):
            data[filename] = os.stat(filename).st_mtime
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def compile(cls, code, sm_version):
        flags = cls.get_flags(sm_version)
        signature = f"{cls.__name__}$${cls.signature()}$${flags}$${code}"
        signature += "$$" + Compiler.cuh_last_update_time()
        hash_hex = jit_utils.hash_to_hex(signature)

        cache_dirname = os.path.join(jit_utils.get_humming_cache_dir(), hash_hex)
        cache_dirname = Path(cache_dirname)
        cache_filename = cache_dirname / "kernel.cubin"
        cache_dirname.mkdir(exist_ok=True, parents=True)

        lock_filename = (cache_dirname / "lock").as_posix()
        with filelock.FileLock(lock_filename):
            if cache_filename.exists():
                return cache_filename.as_posix()

            cache_dirname.mkdir(exist_ok=True, parents=True)
            source_path = os.path.join(cache_dirname, "kernel.cu")
            with open(cache_dirname / "kernel.cu", "w") as f:
                f.write(code)
            with open(cache_dirname / "signature.txt", "w") as f:
                f.write(signature)

            compile_res = cls._compile(source_path, cache_dirname, sm_version)
            returncode, stdout, stderr = compile_res

        with open(cache_dirname / "stdout.log", "w") as f:
            f.write(stdout)
        with open(cache_dirname / "stderr.log", "w") as f:
            f.write(stderr)

        if returncode != 0:
            print(stderr, flush=True)
            (cache_dirname / "kernel_tmp.cubin").unlink(missing_ok=True)
            raise RuntimeError(f"{cls} run failed")

        os.replace(cache_dirname / "kernel_tmp.cubin", cache_dirname / "kernel.cubin")
        return cache_filename.as_posix()


class NVCCCompiler(Compiler):
    @classmethod
    def signature(cls):
        nvcc_path = jit_utils.get_cuda_command_path("nvcc")
        nvcc_version = jit_utils.get_cuda_nvcc_version(nvcc_path)
        return "nvcc+" + nvcc_version

    @classmethod
    def get_flags(cls, sm_version):
        cxx_flags = [
            "-fPIC",
            "-O3",
            "-Wno-deprecated-declarations",
            "-Wno-abi",
            "-fopenmp",
            "-lgomp",
        ]

        return [
            "-std=c++17",
            "--ptxas-options=--register-usage-level=10",
            "--diag-suppress=39,161,174,177,940,177",
            *[f"-I{d}" for d in cls.include_dirs()],
            f"-gencode=arch=compute_{sm_version},code=sm_{sm_version}",
            "-cubin",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            f"--compiler-options={','.join(cxx_flags)}",
        ]

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version):
        nvcc_path = jit_utils.get_cuda_command_path("nvcc")
        target_path = (cache_dirname / "kernel_tmp.cubin").as_posix()
        keep_dirname = cache_dirname / "tmp"
        keep_dirname.mkdir(exist_ok=True, parents=True)
        keep_dirname = keep_dirname.as_posix()
        flags = cls.get_flags(sm_version) + ["--keep", f"--keep-dir={keep_dirname}"]

        cmd = [nvcc_path, source_path, "-o", target_path] + flags
        with open(cache_dirname / "cmdline.json", "w") as f:
            json.dump(cmd, f, ensure_ascii=False)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode, result.stdout, result.stderr
