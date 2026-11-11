#!/usr/bin/env python3
"""Sensitive-information checks for this public repository (see CONTRIBUTING.md).

Run by the git hooks in this directory (enable once per clone with `git config core.hooksPath .githooks`)
and by CI:

    sensitive_scan.py pre-commit                staged file names, added lines and the commit identity
    sensitive_scan.py commit-msg <file>         the commit message
    sensitive_scan.py pre-push <remote> <url>   every commit being pushed that no remote has yet
    sensitive_scan.py scan-range -- <revs...>   every commit `git rev-list <revs...>` selects (CI)

Two pattern sets are applied to every added line (renamed, copied and binary files included, read as raw
bytes), to commit messages and to author/committer identities:

- generic patterns, defined below (home-directory paths, private keys, credential formats, IP addresses,
  real LSF queue/project values, conda env prefixes);
- site-specific patterns (usernames, hostnames, cluster paths, queue names), one regular expression per line
  in an untracked file handed out privately by the maintainer: `<git-common-dir>/info/sensitive-patterns`,
  or the path set with `git config hooks.sensitivePatternsFile <absolute path>`, or `--patterns-file`.

Every mode refuses to run without the site-pattern file, so nothing is committed or pushed unchecked.
Standard library only, Python 3.6+: the hooks run with whatever `python3` is on PATH.
"""
import argparse
import os
import re
import socket
import subprocess
import sys

# (category, compiled regex). Each is written so that it does not match its own source line.
GENERIC_PATTERNS = [
    ('home-directory path', re.compile(r'(?:/Users|/home)/(?![<${\[(*.])[\w.-]+')),
    ('private key', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')),
    ('AWS access key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('GitHub token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})')),
    ('Slack token', re.compile(r'\bxox[abprs]-[A-Za-z0-9-]{10,}')),
    ('credential assignment', re.compile(
        r'(?i)\b(?:passw(?:or)?d|secret|api[_-]?key|access[_-]?key|token)\b\s*[:=]\s*[\'"][^\'"\s]{8,}[\'"]')),
    # Not preceded/followed by version-string characters, so `pkg=1.2.15.3=h0` is not taken for an address.
    ('IP address', re.compile(r'(?<![\w.=-])(?!0\.0\.0\.0\b|127\.)(?:\d{1,3}\.){3}\d{1,3}(?![\w.-])')),
    # A real value starts with an alphanumeric; `<queue>`, `{queue}` and `$QUEUE` placeholders pass.
    ('LSF queue/project value', re.compile(r'#BSUB\s+-[qPG]\s+(?![<{$])\S')),
    ('conda env prefix', re.compile(r'^\s*prefix:\s*/')),
]

# New files whose kind embeds real paths, hosts or credentials. Templates that hardcode nothing are allowed.
BLOCKED_NAMES = re.compile(
    r'(?:^|/)(?:errfile|logfile)\.|\.(?:lsf|out|err|log)$|(?:^|/)mlflow\.db|(?:^|/)mlruns/|(?:^|/)\.env$'
    r'|(?:^|/)id_(?:rsa|dsa|ecdsa|ed25519)|\.(?:pem|key|keytab|p12|pfx)$')
ALLOWED_NAMES = {'mlflow_server.lsf', 'config/pipeline/submit_pipeline.lsf'}  # placeholder-only templates

# This is a fork: changes go upstream through pull requests, never by pushing there.
PROTECTED_REMOTE_NAME = 'upstream'
PROTECTED_PUSH_URL = re.compile(r'github\.com[:/]UU-Hydro/', re.I)

DIFF_OPTIONS = ('-p', '-U0', '-M', '-C', '--text', '--no-color', '--no-ext-diff', '--no-textconv',
                '--src-prefix=a/', '--dst-prefix=b/')
HUNK_HEADER = re.compile(rb'^(@@+) -.*?\+(\d+)')
SCISSORS = '# ------------------------ >8 ------------------------'

MISSING_PATTERNS = """sensitive-scan: BLOCKED -- the site-pattern file is missing: {path}
Commits and pushes are refused until it is present. Ask the maintainer for the file and save it at that
path, or anywhere outside the repository and run: git config hooks.sensitivePatternsFile <absolute path>
Never commit or share it. See CONTRIBUTING.md."""


def git(*args, input_bytes=None):
    """Run git and return its stdout (bytes); a git failure aborts the check, which blocks the operation."""
    proc = subprocess.run(('git',) + args, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.exit('sensitive-scan: BLOCKED -- `git {}` failed: {}'.format(
            ' '.join(args), proc.stderr.decode('utf-8', 'replace').strip()))
    return proc.stdout


def git_config(key):
    proc = subprocess.run(('git', 'config', '--get', key), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout.decode('utf-8', 'replace').strip()


def default_patterns_file():
    configured = git_config('hooks.sensitivePatternsFile')
    if configured:
        return os.path.expanduser(configured)
    common_dir = git('rev-parse', '--git-common-dir').decode().strip()
    return os.path.join(os.path.abspath(common_dir), 'info', 'sensitive-patterns')


def load_patterns(path):
    """Generic patterns plus the site patterns from `path`; exits (blocking) if the file is missing or empty."""
    if not os.path.isfile(path):
        sys.exit(MISSING_PATTERNS.format(path=path))
    site = []
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                site.append(('site pattern #{}'.format(number), re.compile(line, re.I)))
            except re.error as error:
                sys.exit('sensitive-scan: BLOCKED -- {}:{}: invalid regular expression: {}'.format(path, number, error))
    if not site:
        sys.exit('sensitive-scan: BLOCKED -- the site-pattern file {} contains no patterns'.format(path))
    return GENERIC_PATTERNS + site


def scan_text(text, where, patterns):
    return [(where, category, text) for category, pattern in patterns if pattern.search(text)]


def _diff_path(target):
    path = target.decode('utf-8', 'replace')
    if path == '/dev/null':
        return None
    if path.startswith('"'):
        path = path[1:-1]
    return path[2:] if path.startswith('b/') else path


def added_lines(diff):
    """Yield (path, line number, text) for every line a diff adds, in plain or combined (--cc) format.

    In a combined diff only lines new with respect to every parent count: lines taken from one parent were
    scanned in that parent's own commits.
    """
    path, number, width, in_hunk = None, 0, 1, False
    for raw in diff.split(b'\n'):
        if raw.startswith((b'diff --git ', b'diff --cc ', b'diff --combined ')):
            path, in_hunk = None, False
            continue
        header = HUNK_HEADER.match(raw)
        if header:
            width, number, in_hunk = len(header.group(1)) - 1, int(header.group(2)), True
            continue
        if not in_hunk:
            if raw.startswith(b'+++ '):
                path = _diff_path(raw[4:])
            continue
        prefix = raw[:width]
        if b'-' in prefix:
            continue
        if path is not None and prefix == b'+' * width:
            yield path, number, raw[width:].decode('utf-8', 'replace')
        number += 1


def content_findings(diff, label, patterns):
    findings = []
    for path, number, text in added_lines(diff):
        findings += scan_text(text, '{}{}:{}'.format(label, path, number), patterns)
    return findings


def name_findings(names, label):
    return [('{}{}'.format(label, name), 'file type that embeds real paths/hosts/credentials', name)
            for name in names.decode('utf-8', 'replace').split('\0')
            if name and name not in ALLOWED_NAMES and BLOCKED_NAMES.search(name)]


def identity_findings(ident, where, patterns):
    """Pattern hits in 'Name <email>', plus emails git derived from user@host because none was configured."""
    findings = scan_text(ident, where, patterns)
    domain = ident[ident.find('<') + 1:ident.rfind('>')].rpartition('@')[2].lower()
    host = socket.gethostname().lower()
    if ('.' not in domain or domain == host or domain.startswith(host + '.')
            or domain.endswith(('.local', '.localdomain', '.(none)'))):
        findings.append((where, 'auto-generated identity: set user.name/user.email (see CONTRIBUTING.md)', ident))
    return findings


def message_findings(message, where, patterns):
    findings = []
    for number, line in enumerate(message.splitlines(), 1):
        if line == SCISSORS:
            break
        if not line.startswith('#'):
            findings += scan_text(line, '{}:{}'.format(where, number), patterns)
    return findings


def scan_commits(commits, patterns):
    findings = []
    for sha in commits:
        label = sha[:10] + ' '
        author, committer, message = git(
            'show', '-s', '--format=%an <%ae>%x00%cn <%ce>%x00%B', sha).decode('utf-8', 'replace').split('\0', 2)
        findings += identity_findings(author, label + 'author', patterns)
        findings += identity_findings(committer, label + 'committer', patterns)
        findings += message_findings(message, label + 'message', patterns)
        findings += name_findings(git('diff-tree', '-r', '-z', '--name-only', '--no-commit-id', '--root', '-M', '-C',
                                      '--diff-filter=ACR', sha), label)
        findings += content_findings(git('diff-tree', '--no-commit-id', '--root', '--cc', *DIFF_OPTIONS, sha),
                                     label, patterns)
    return findings


def ident_without_date(variable):
    return git('var', variable).decode('utf-8', 'replace').strip().rsplit(' ', 2)[0]


def pre_commit(patterns):
    findings = identity_findings(ident_without_date('GIT_AUTHOR_IDENT'), 'author', patterns)
    findings += identity_findings(ident_without_date('GIT_COMMITTER_IDENT'), 'committer', patterns)
    findings += name_findings(git('diff', '--cached', '-z', '--name-only', '-M', '-C', '--diff-filter=ACR'), '')
    findings += content_findings(git('diff', '--cached', '--diff-filter=ACMRT', *DIFF_OPTIONS), '', patterns)
    return findings


def commit_msg(path, patterns):
    with open(path, 'rb') as handle:
        return message_findings(handle.read().decode('utf-8', 'replace'), 'commit message', patterns)


def pre_push(remote, url, patterns, updates):
    if remote == PROTECTED_REMOTE_NAME or PROTECTED_PUSH_URL.search(url):
        return [(remote, 'push to the upstream repository (open a pull request from the fork instead)', url)]
    tips = [fields[1] for fields in (line.split() for line in updates)
            if len(fields) == 4 and set(fields[1]) != {'0'}]
    if not tips:
        return []
    return scan_commits(git('rev-list', *tips, '--not', '--remotes').decode().split(), patterns)


def report(findings, redact):
    if not findings:
        return 0
    print('sensitive-scan: BLOCKED -- these match the sensitive-information policy (CONTRIBUTING.md):',
          file=sys.stderr)
    for where, category, text in findings:
        print('  {}: [{}]{}'.format(where, category, '' if redact else '  ' + text.strip()[:160]), file=sys.stderr)
    print('Remove them from every commit concerned (amend or rebase; a follow-up commit is not enough), or ask\n'
          'the maintainer if this is a false positive.', file=sys.stderr)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--patterns-file', help='site-pattern file (default: see above)')
    parser.add_argument('--redact', action='store_true', help='report locations only, never matched text (CI)')
    sub = parser.add_subparsers(dest='mode')
    sub.required = True
    sub.add_parser('pre-commit')
    sub.add_parser('commit-msg').add_argument('message_file')
    push = sub.add_parser('pre-push')
    push.add_argument('remote')
    push.add_argument('url')
    sub.add_parser('scan-range').add_argument('revs', nargs='+', help='git rev-list arguments (after --)')
    args = parser.parse_args(argv)

    patterns = load_patterns(args.patterns_file or default_patterns_file())
    if args.mode == 'pre-commit':
        findings = pre_commit(patterns)
    elif args.mode == 'commit-msg':
        findings = commit_msg(args.message_file, patterns)
    elif args.mode == 'pre-push':
        findings = pre_push(args.remote, args.url, patterns, sys.stdin.read().splitlines())
    else:
        findings = scan_commits(git('rev-list', *args.revs).decode().split(), patterns)
    return report(findings, args.redact)


if __name__ == '__main__':
    sys.exit(main())
