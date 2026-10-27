"""Contracts, optional activation, and runtime failures for llama.cpp routing."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aurauv.config import load_config
from aurauv.errors import AurauvError, ProviderPreflightError, ProviderVerificationError
from aurauv.invocation import parse_invocation
from aurauv.process import CommandRunner
from aurauv.providers.registry import build_provider
from aurauv.routes import select_routes, selected_extras
from helpers import make_wheel


CONFIG = """[project]
name = "llama-route-test"
version = "0.0.0"
requires-python = ">=3.11"
[project.optional-dependencies]
llama = []
llama-cpu = ["llama-cpp-python>=0.3.36"]
llama-cuda = ["llama-cpp-python>=0.3.36", "nvidia-cuda-runtime>=13,<14", "nvidia-cublas>=13,<14"]
[tool.uv]
conflicts = [[{extra="llama-cpu"}, {extra="llama-cuda"}]]
[[tool.uv.index]]
name = "llama-cpu"
url = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
explicit = true
[[tool.uv.index]]
name = "llama-cuda"
url = "https://abetlen.github.io/llama-cpp-python/whl/cu130"
explicit = true
[tool.uv.sources]
llama-cpp-python = [
{index="llama-cpu", extra="llama-cpu"},
{index="llama-cuda", extra="llama-cuda"},
]
[tool.aurauv]
schema-version = 1
python-install-policy = "never"
uv-update-policy = "never"
[tool.aurauv.routes.llama]
default = "auto"
detector = "llama"
providers = ["llama"]
activation-extra = "llama"
[tool.aurauv.routes.llama.options.cpu]
extras = ["llama-cpu"]
[tool.aurauv.routes.llama.options.cuda]
extras = ["llama-cuda"]
[tool.aurauv.providers.llama]
type = "llama-cpp"
route = "llama"
indexes = {cpu="llama-cpu", cuda="llama-cuda"}
"""


def provider_at(tmp_path, monkeypatch, config_text=CONFIG):
    (tmp_path / 'pyproject.toml').write_text(config_text)
    config = load_config(tmp_path)
    provider = build_provider(config.providers['llama'], config=config, runner=CommandRunner())
    monkeypatch.setattr(provider, '_nvidia', lambda: {'present': False, 'available': False})
    provider.validate_contract()
    return config, provider


def test_contract_indexes_and_cpu_detection(tmp_path, monkeypatch):
    config, provider = provider_at(tmp_path, monkeypatch)
    assert config.routes['llama'].activation_extra == 'llama'
    assert provider.detect().option == 'cpu'
    assert provider.preflight('cpu')['nvcc_required'] is False
    assert provider.protected_packages() == ('llama-cpp-python',)
    assert provider.current_target_install_args('cuda') == (
        '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cu130',
        '--only-binary', 'llama-cpp-python',
    )


@pytest.mark.parametrize('old,new', [
    ('explicit = true', 'explicit = false'),
    ('abetlen.github.io', 'untrusted.invalid'),
    ('/whl/cu130', '/whl/cpu'),
    ('activation-extra = "llama"', 'activation-extra = "missing"'),
    ('activation-extra = "llama"', 'activation-extra = "llama-cpu"'),
    ('{index="llama-cuda", extra="llama-cuda"}', '{index="llama-cpu", extra="llama-cuda"}'),
])
def test_invalid_contract_is_rejected(tmp_path, monkeypatch, old, new):
    with pytest.raises(AurauvError):
        provider_at(tmp_path, monkeypatch, CONFIG.replace(old, new))


@pytest.mark.parametrize('arguments,extras', [
    ([], ()),
    (['--extra', 'llama'], ('llama-cpu',)),
    (['--extra', 'llama-cpu'], ('llama-cpu',)),
    (['--extra', 'llama', '--no-extra', 'llama'], ()),
])
def test_activation_preserves_optional_dependency(tmp_path, monkeypatch, arguments, extras):
    config, provider = provider_at(tmp_path, monkeypatch)
    selected = select_routes(config, {'llama': provider}, parse_invocation(['sync']).aura,
        arguments, {}, tmp_path / 'absent-python')
    assert selected_extras(config, selected) == extras
    assert provider._required == bool(extras)


def test_inactive_route_does_not_reactivate_from_saved_state(tmp_path, monkeypatch):
    config, provider = provider_at(tmp_path, monkeypatch)
    selected = select_routes(config, {'llama': provider}, parse_invocation(['sync']).aura,
        [], {'llama': 'cuda'}, tmp_path / 'absent-python')
    assert selected['llama'].selected == 'cuda'
    assert selected_extras(config, selected) == ()
    assert not provider._required


def test_explicit_override_activates_route_and_exclusion_conflicts(tmp_path, monkeypatch):
    config, provider = provider_at(tmp_path, monkeypatch)
    aura = replace(parse_invocation(['sync']).aura, route_overrides={'llama': 'cpu'})
    selected = select_routes(config, {'llama': provider}, aura, [], {}, tmp_path / 'python')
    assert selected_extras(config, selected) == ('llama-cpu',)
    with pytest.raises(AurauvError, match='contradicts'):
        select_routes(config, {'llama': provider}, aura, ['--no-extra', 'llama'], {}, tmp_path / 'python')


def test_present_but_broken_nvidia_does_not_silently_choose_cpu(tmp_path, monkeypatch):
    config, provider = provider_at(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, '_nvidia', lambda: {'present': True, 'available': False})
    assert provider.detect().option == 'cuda'
    with pytest.raises(ProviderPreflightError):
        select_routes(config, {'llama': provider}, parse_invocation(['sync']).aura,
            ['--extra', 'llama'], {}, tmp_path / 'python')


def test_cuda_detection_and_hardware_fingerprint(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    before = provider.machine_fingerprint()
    monkeypatch.setattr('aurauv.providers.llama_cpp.platform.system', lambda: 'Linux')
    monkeypatch.setattr('aurauv.providers.llama_cpp.platform.machine', lambda: 'x86_64')
    monkeypatch.setattr(provider, '_nvidia', lambda: {'present': True, 'available': True, 'gpus': ['NVIDIA test']})
    assert provider.machine_fingerprint() != before
    assert provider.detect().option == 'cuda'
    assert provider.preflight('cuda')['nvidia']['available']


@pytest.mark.parametrize('option,gpu,backend,allocated,success', [
    ('cpu', False, 'CPU', True, True),
    ('cuda', True, 'CUDA', True, True),
    ('cuda', False, 'CPU', True, False),
    ('cuda', True, 'Vulkan', True, False),
    ('cuda', True, 'CUDA', False, False),
    ('cpu', True, 'CUDA', True, False),
])
def test_verification_requires_selected_native_backend(tmp_path, monkeypatch, option, gpu, backend, allocated, success):
    _, provider = provider_at(tmp_path, monkeypatch)
    state = {'installed': True, 'version': '0.3.36', 'gpu_offload': gpu,
        'devices': [{'backend': backend, 'allocation_verified': allocated}]}
    monkeypatch.setattr(provider, '_probe', lambda *args, **kwargs: state)
    if success:
        assert provider.verify(Path(sys.executable), option=option, cwd=tmp_path).details['active']
    else:
        with pytest.raises(ProviderVerificationError) as error:
            provider.verify(Path(sys.executable), option=option, cwd=tmp_path)
        assert not error.value.repairable
        assert not error.value.fallback_safe


def test_missing_optional_is_allowed_only_before_activation(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, '_probe', lambda *args, **kwargs: {'installed': False})
    assert not provider.verify(Path(sys.executable), option='cpu', cwd=tmp_path).details['active']
    provider.preflight('cpu')
    with pytest.raises(ProviderVerificationError):
        provider.verify(Path(sys.executable), option='cpu', cwd=tmp_path)


def test_probe_uses_requested_interpreter_and_bounds_native_work(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"installed": false}', '')
    monkeypatch.setattr('aurauv.providers.llama_cpp.subprocess.run', execute)
    assert provider._probe(Path('/selected/python'), tmp_path, option='cpu') == {'installed': False}
    assert calls[0][0][0] == '/selected/python'
    assert calls[0][1]['timeout'] == 30
    assert 'allocate(backend, 4096)' in calls[0][0][2]


def test_uv_sync_does_not_install_unselected_feature_then_installs_selection(tmp_path, uv_path):
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    wheel = make_wheel(wheels, 'optional-engine', 'optional_engine')
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'pyproject.toml').write_text(f"""[project]
name = "optional-routing"
version = "0.0.0"
requires-python = ">=3.11"
[project.optional-dependencies]
feature = []
cpu = ["optional-engine @ {wheel.as_uri()}"]
cuda = ["optional-engine @ {wheel.as_uri()}"]
[tool.uv]
conflicts = [[{{extra="cpu"}}, {{extra="cuda"}}]]
[tool.aurauv]
schema-version = 1
python-install-policy = "never"
uv-update-policy = "never"
[tool.aurauv.routes.engine]
default = "cpu"
activation-extra = "feature"
providers = []
[tool.aurauv.routes.engine.options.cpu]
extras = ["cpu"]
[tool.aurauv.routes.engine.options.cuda]
extras = ["cuda"]
""")
    env = {**os.environ, 'AURAUV_UV': str(uv_path),
        'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    base = [sys.executable, '-m', 'aurauv', '--offline', 'sync', '--python', sys.executable]
    def sync(*args):
        result = subprocess.run([*base, *args], cwd=project, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
    def installed():
        python = project / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
        result = subprocess.run([str(python), '-c', 'import optional_engine'], capture_output=True)
        return result.returncode == 0
    sync()
    assert not installed()
    sync('--extra', 'feature')
    assert installed()
    sync()
    assert not installed()


def test_upgrade_activates_optional_provider(tmp_path, uv_path, monkeypatch):
    from aurauv.engine import AurauvEngine
    config, provider = provider_at(tmp_path, monkeypatch)
    monkeypatch.setenv('AURAUV_UV', str(uv_path))
    engine = AurauvEngine(parse_invocation(['aura', 'upgrade', 'llama']), cwd=tmp_path)
    engine.config = config
    engine.providers = {'llama': provider}
    commands = []
    selected_arguments = []
    monkeypatch.setattr(engine, '_prepare_project', lambda **kwargs: None)
    monkeypatch.setattr(engine, '_run_uv', lambda args, **kwargs: commands.append(args))
    def select(args):
        selected_arguments.extend(args)
        return select_routes(config, {'llama': provider}, engine.parsed.aura,
            args, {}, tmp_path / 'python')
    monkeypatch.setattr(engine, '_select', select)
    monkeypatch.setattr(engine, '_verify_with_runtime_fallback', lambda selections, **kwargs: (selections, {}))
    monkeypatch.setattr(engine, '_write_state', lambda *args: None)
    assert engine.upgrade_provider('llama') == 0
    assert selected_arguments[-2:] == ['--extra', 'llama']
    assert 'llama-cpu' in commands[-1]


def test_rejects_unrouted_llama_dependency(tmp_path, monkeypatch):
    with pytest.raises(AurauvError, match='only in'):
        provider_at(tmp_path, monkeypatch, CONFIG.replace('llama = []', 'llama = ["llama-cpp-python"]'))


def test_rejects_untested_platform_instead_of_pretending_cpu_wheel_is_metal(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    monkeypatch.setattr('aurauv.providers.llama_cpp.platform.system', lambda: 'Darwin')
    with pytest.raises(ProviderPreflightError, match='Linux'):
        provider.preflight('cpu')


def fake_distribution(site, name, files):
    """Create metadata-only distributions for subprocess library discovery tests."""
    normalized = name.replace('-', '_')
    metadata = site / f'{normalized}-13.0.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: 13.0\n')
    (metadata / 'RECORD').write_text(''.join(f'{file},,\n' for file in files))
    for file in files:
        path = site / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'metadata discovery only')


def test_cuda_runtime_paths_use_target_metadata_and_preserve_parent(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    site = tmp_path / 'python-site'
    fake_distribution(site, 'llama-cpp-python', [])
    fake_distribution(site, 'nvidia-cuda-runtime', ['nvidia/runtime/lib/libcudart.so.13'])
    fake_distribution(site, 'nvidia-cublas', [
        'nvidia/blas/lib/libcublas.so.13', 'nvidia/blas/lib/libcublasLt.so.13',
    ])
    monkeypatch.setenv('PYTHONPATH', str(site))
    monkeypatch.setenv('LD_LIBRARY_PATH', '/caller/lib')
    values = provider.runtime_environment(Path(sys.executable), option='cuda', cwd=tmp_path)
    assert values['LD_LIBRARY_PATH'].split(os.pathsep) == [
        str(site / 'nvidia/runtime/lib'), str(site / 'nvidia/blas/lib'), '/caller/lib',
    ]
    assert os.environ['LD_LIBRARY_PATH'] == '/caller/lib'


def test_cuda_runtime_missing_dependency_reports_actionable_failure(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    site = tmp_path / 'python-site'
    fake_distribution(site, 'llama-cpp-python', [])
    fake_distribution(site, 'nvidia-cuda-runtime', [])
    fake_distribution(site, 'nvidia-cublas', [])
    monkeypatch.setenv('PYTHONPATH', str(site))
    with pytest.raises(AurauvError, match='libcudart.so.13'):
        provider.runtime_environment(Path(sys.executable), option='cuda', cwd=tmp_path)


def test_cpu_route_does_not_probe_nvidia_libraries(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    assert provider.runtime_environment(tmp_path / 'missing-python', option='cpu', cwd=tmp_path) == {}


def test_absent_optional_cuda_provider_does_not_change_environment(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    monkeypatch.setattr('aurauv.providers.llama_cpp.subprocess.run', lambda *args, **kwargs:
        subprocess.CompletedProcess(args[0], 0, '{"installed": false, "directories": []}', ''))
    assert provider.runtime_environment(Path(sys.executable), option='cuda', cwd=tmp_path) == {}


def test_cuda_extra_requires_its_runtime_packages(tmp_path, monkeypatch):
    with pytest.raises(AurauvError, match='NVIDIA runtime packages'):
        provider_at(tmp_path, monkeypatch, CONFIG.replace(', "nvidia-cublas>=13,<14"', ''))


def test_native_probe_receives_runtime_environment(tmp_path, monkeypatch):
    _, provider = provider_at(tmp_path, monkeypatch)
    monkeypatch.setattr(provider, 'runtime_environment', lambda *args, **kwargs:
        {'LD_LIBRARY_PATH': '/runtime/lib'})
    calls = []
    def execute(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0, '{"installed": false}', '')
    monkeypatch.setattr('aurauv.providers.llama_cpp.subprocess.run', execute)
    provider._probe(Path(sys.executable), tmp_path, option='cuda')
    assert calls[0]['env']['LD_LIBRARY_PATH'] == '/runtime/lib'


def test_runtime_environment_reaches_managed_run_and_import_checks(tmp_path, uv_path, monkeypatch):
    from aurauv.engine import AurauvEngine
    from aurauv.models import VerificationResult
    from aurauv.providers.llama_cpp import LlamaCppProvider

    (tmp_path / 'pyproject.toml').write_text(CONFIG + '\n[tool.aurauv.verify]\nimports = ["runtime_check"]\n')
    (tmp_path / 'runtime_check.py').write_text(
        'import os\nassert os.environ["AURA_TEST_RUNTIME"] == "ready"\n'
    )
    subprocess.run([str(uv_path), 'venv', '--offline', '--python', sys.executable, '.venv'],
        cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setenv('AURAUV_UV', str(uv_path))
    monkeypatch.delenv('AURA_TEST_RUNTIME', raising=False)
    monkeypatch.setattr(LlamaCppProvider, '_nvidia', staticmethod(lambda:
        {'present': False, 'available': False}))
    monkeypatch.setattr(LlamaCppProvider, 'verify', lambda self, python, **kwargs:
        VerificationResult(self.name, self.route.name, kwargs['option'], {'installed': False}))
    def runtime(self, python, **kwargs):
        assert python.parent.parent == tmp_path / '.venv'
        return {'AURA_TEST_RUNTIME': 'ready'}
    monkeypatch.setattr(LlamaCppProvider, 'runtime_environment', runtime)
    engine = AurauvEngine(parse_invocation([
        'run', '--no-sync', '--offline', 'python', '-c',
        'import runtime_check; print("runtime environment propagated")',
    ]), cwd=tmp_path)
    assert engine.execute() == 0
    assert 'AURA_TEST_RUNTIME' not in os.environ
