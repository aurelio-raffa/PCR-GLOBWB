"""Tests for the sensitive-information git hooks (.githooks/) and the job-generator guard (create_job_file.py).

Each test builds a throwaway repository with the hooks enabled and a site-pattern file holding a fake token.
Every sensitive-looking string below is assembled at runtime, so this file passes the hooks it tests.
"""
import importlib.util
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_TOKEN = 'zz' + 'sitetoken'
HOME_PATH = '/Us' + 'ers/alice/run01'
PRIVATE_IP = '10.1' + '.2.3'
BSUB_QUEUE = '#BSUB' + ' -q realq'


def git(cwd, *args, stdin=None):
    return subprocess.run(('git',) + args, cwd=str(cwd), input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)


def commit(repo, files, message='change'):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    git(repo, 'add', '-A')
    return git(repo, 'commit', '-q', '-m', message)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    for var in ('XDG_CONFIG_HOME', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'EMAIL',
                'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL'):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / 'work'
    work.mkdir()
    git(work, 'init', '-q')
    shutil.copytree(str(REPO_ROOT / '.githooks'), str(work / '.githooks'))
    for key, value in (('core.hooksPath', '.githooks'), ('user.useConfigOnly', 'true'),
                       ('user.name', 'Test User'), ('user.email', 'test@example.invalid')):
        git(work, 'config', key, value)
    (work / '.git' / 'info' / 'sensitive-patterns').write_text('# fake site token\n' + SITE_TOKEN + '\n')
    assert commit(work, {'README.md': 'clean\n'}, 'initial').returncode == 0
    return work


def blocked(result):
    return result.returncode != 0 and 'BLOCKED' in result.stdout


def test_clean_commit_passes(repo):
    assert commit(repo, {'notes.txt': 'nothing to see\n'}).returncode == 0


def test_commit_refused_without_pattern_file(repo):
    (repo / '.git' / 'info' / 'sensitive-patterns').unlink()
    result = commit(repo, {'notes.txt': 'nothing to see\n'})
    assert blocked(result) and 'site-pattern file is missing' in result.stdout


def test_configured_pattern_file_is_used(repo, tmp_path):
    outside = tmp_path / 'patterns'
    (repo / '.git' / 'info' / 'sensitive-patterns').rename(outside)
    git(repo, 'config', 'hooks.sensitivePatternsFile', str(outside))
    assert blocked(commit(repo, {'notes.txt': SITE_TOKEN + '\n'}))


def test_site_token_in_readme_blocked(repo):
    assert blocked(commit(repo, {'README.md': 'host: ' + SITE_TOKEN + '\n'}))


def test_token_in_commit_message_blocked(repo):
    assert blocked(commit(repo, {'notes.txt': 'x\n'}, 'ran on ' + SITE_TOKEN))


def test_binary_file_scanned(repo):
    assert blocked(commit(repo, {'data.bin': b'\x00\x01' + SITE_TOKEN.encode() + b'\x00\xff'}))


def test_rename_into_job_file_blocked(repo):
    assert commit(repo, {'notes.txt': 'x\n'}).returncode == 0
    git(repo, 'mv', 'notes.txt', 'job.lsf')
    assert blocked(git(repo, 'commit', '-q', '-m', 'rename'))


def test_lsf_template_allowed(repo):
    assert commit(repo, {'mlflow_server.lsf': '#BSUB -J mlflow_ui\n'}).returncode == 0


@pytest.mark.parametrize('line', [BSUB_QUEUE, HOME_PATH, PRIVATE_IP + ':5050', 'password = "' + 'hunter2hunter2"',
                                  '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----', 'prefix: /opt/' + 'envs/x'])
def test_generic_pattern_blocked(repo, line):
    assert blocked(commit(repo, {'script.sh': line + '\n'}))


@pytest.mark.parametrize('line', ['#BSUB -q <queue>', '/Users/<user>/run01', '/home/$USER/run01',
                                  'mlflow ui --host 0.0.0.0', 'http://127.0.0.1:5050', '  - libxyz=1.2.15.3=h0',
                                  'token = line.strip()'])
def test_placeholders_and_lookalikes_pass(repo, line):
    assert commit(repo, {'script.sh': line + '\n'}).returncode == 0


def test_auto_generated_identity_blocked(repo):
    git(repo, 'config', 'user.email', 'me@' + socket.gethostname())
    assert blocked(commit(repo, {'notes.txt': 'x\n'}))


@pytest.fixture
def remote(repo, tmp_path):
    bare = tmp_path / 'remote.git'
    git(tmp_path, 'init', '-q', '--bare', str(bare))
    git(repo, 'remote', 'add', 'origin', str(bare))
    assert git(repo, 'push', '-q', 'origin', 'HEAD:refs/heads/master').returncode == 0
    return bare


def test_push_rescans_commits_made_with_no_verify(repo, remote):
    (repo / 'notes.txt').write_text(SITE_TOKEN + '\n')
    git(repo, 'add', '-A')
    assert git(repo, 'commit', '-q', '--no-verify', '-m', 'sneaky').returncode == 0
    assert blocked(git(repo, 'push', 'origin', 'HEAD:refs/heads/master'))


def test_clean_push_passes(repo, remote):
    assert commit(repo, {'notes.txt': 'x\n'}).returncode == 0
    assert git(repo, 'push', '-q', 'origin', 'HEAD:refs/heads/master').returncode == 0


def test_push_refused_without_pattern_file(repo, remote):
    assert commit(repo, {'notes.txt': 'x\n'}).returncode == 0
    (repo / '.git' / 'info' / 'sensitive-patterns').unlink()
    assert blocked(git(repo, 'push', 'origin', 'HEAD:refs/heads/master'))


def test_push_to_remote_named_upstream_refused(repo, remote):
    git(repo, 'remote', 'add', 'upstream', str(remote))
    assert blocked(git(repo, 'push', 'upstream', 'HEAD:refs/heads/other'))


def test_push_to_upstream_org_url_refused():
    # Called directly: git contacts the remote before running pre-push, and tests must stay offline.
    findings = _load('sensitive_scan', REPO_ROOT / '.githooks' / 'sensitive_scan.py').pre_push(
        'mirror', 'git@github.com:UU-Hydro/PCR-GLOBWB_model.git', [], [])
    assert findings and 'upstream' in findings[0][1]


def test_scan_range_redacts_matches(repo):
    (repo / 'notes.txt').write_text(SITE_TOKEN + '\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '--no-verify', '-m', 'sneaky')
    result = subprocess.run(['python3', str(repo / '.githooks' / 'sensitive_scan.py'), '--redact', 'scan-range',
                             '--', 'HEAD~1..HEAD'], cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    assert blocked(result) and 'notes.txt:1' in result.stdout and SITE_TOKEN not in result.stdout


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('key', ['core.hooksPath', 'user.useConfigOnly'])
def test_job_generation_requires_hooks(repo, key):
    module = _load('create_job_file', REPO_ROOT / 'create_job_file.py')
    module._check_repo_hygiene(str(repo))
    git(repo, 'config', '--unset', key)
    with pytest.raises(SystemExit, match=key):
        module._check_repo_hygiene(str(repo))
