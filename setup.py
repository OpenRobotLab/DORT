import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources, define_macros=None, extra_compile_args=None):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'det3d/version.py')

    setup(
        name='det3d',
        version=version,
        description='',
        install_requires=[
            'pyyaml'
        ],
        author='qing lian',
        author_email='lianqing1997@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='build_cost_volume_cuda',
                module='det3d.ops.build_cost_volume',
                sources=[
                    'src/BuildCostVolume.cpp',
                    'src/BuildCostVolume_cuda.cu',
                ],
                define_macros=[("WITH_CUDA", None)]
            ),
            make_cuda_ext(
                name='bev_pool_v2_ext',
                module='det3d.ops.bev_pool_v2',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
            ),
        ],
    )
