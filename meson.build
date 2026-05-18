project(
  'forgeff',
  'c',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

add_project_arguments(
  '-O2',
  language: 'c',
)

py = import('python').find_installation(pure: false)

install_subdir(
  'forgeff',
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__'],
)

inc_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()
