# Contributing

This fork is **public**: a commit becomes public the moment it is pushed to any branch, and removing it later
does not fully undo that (forks, caches and the fork network keep old commits reachable). Please do the
one-time setup below before your first commit, and read the sensitive-information policy.

## One-time setup (every clone)

From the repository root:

```shell
git config core.hooksPath .githooks          # enable the checks described below
git config user.useConfigOnly true           # never let git invent a user@host identity
git config user.name  "Your Name"
git config user.email "<id>+<login>@users.noreply.github.com"   # your GitHub noreply address
git remote set-url --push upstream DISABLED  # only if you added the UU-Hydro repository as `upstream`
```

Your GitHub noreply address is shown under *Settings > Emails* on GitHub.

Then get the **site-pattern file** from the maintainer (it is handed out privately) and save it as
`.git/info/sensitive-patterns` (in a worktree: `$(git rev-parse --git-common-dir)/info/sensitive-patterns`),
or keep it anywhere outside the repository and run
`git config hooks.sensitivePatternsFile <absolute path>`. **Never commit, paste or share this file**: it
lists exactly the values that must stay out of the repository.

Without the site-pattern file every commit and push is refused. `create_job_file.py` refuses to generate jobs
until `core.hooksPath` and `user.useConfigOnly` are set as above.

## Sensitive information policy

None of the following may appear in any tracked file, commit diff, commit message or commit metadata:

1. **Absolute paths on any machine** (cluster, laptop, workstation), including home, work and scratch
   directories.
2. **Usernames, user IDs and home-directory names.**
3. **Hostnames, node names, IP addresses, internal domains**, and server or tracking-URI hosts.
4. **Scheduler and account identifiers**: LSF project/account codes (`-P`), queue names (`-q`),
   reservation names.
5. **HPC architecture details**: filesystem layout, module names, machine-specific conda environment names
   or prefixes, node specifications, partition sizes.
6. **Security posture of the infrastructure**: authentication methods, multi-factor or SSH/Kerberos setup,
   open ports, mechanisms that are broken or disabled.
7. **Credentials**: passwords, tokens, API keys, SSH or Kerberos material, secrets passed with `--env`.
8. **Restricted or licensed data**, and anything derived from it (for example per-site records).
9. **Generated artifacts that embed any of the above**: generated `.lsf` job files and INI files, LSF and
   model logs, the MLflow store, tool-output dumps, notebook outputs, `conda env export` files with a
   `prefix:` line.
10. **An auto-generated identity** (`user@host`) as commit author or committer: set your identity as above.

How to keep them out:

- Real values are supplied at submission time as command-line arguments or environment variables. They only
  appear in the generated `.lsf` (git-ignored) and in the running job's environment.
- Committed configuration refers to them only through `{{$ENV_VAR}}` placeholders.
- In documentation, examples and tests use placeholders such as `/path/to/run`, `$USER`, `<user>`,
  `<compute-node>`, `<queue>`, or obviously fake values.
- Run outputs, logs, environments and MLflow data are git-ignored (see `.gitignore`). Do not force-add them.

## What the checks do

The hooks in `.githooks/` (run by `sensitive_scan.py`) apply generic patterns (home-directory paths,
private keys, credential formats, IP addresses, real `#BSUB` queue/project values, conda prefixes) together
with the site patterns:

- **pre-commit**: checks your author/committer identity, refuses new files of sensitive kinds (generated
  `.lsf`, LSF logs, MLflow store, keys, `.env`), including files renamed or copied into those names, and scans
  every added line of every staged file, documentation and binary files included.
- **commit-msg**: scans the commit message.
- **pre-push**: refuses pushes to the UU-Hydro repository, then rescans every commit being pushed that no
  remote has yet. This also covers commits created by rebase, cherry-pick or `--no-verify`, which skip
  pre-commit.
- **CI** (GitHub Actions) repeats the scan on every push and pull request; its log shows locations only.

Limits: compressed content (zip, `.npz`, NetCDF-4) cannot be read by the scanner, and the hooks only run in
clones where they are enabled. `git commit --no-verify` exists for genuine false positives only, and the
push will still be checked. If a check flags something that is not sensitive, ask the maintainer to adjust
the patterns rather than bypassing it.

Changes meant for the original model go to UU-Hydro through a pull request from this fork, never by pushing
there directly.

## If something sensitive was committed

- **Not pushed yet**: remove it from every commit that contains it (`git commit --amend`, or an interactive
  rebase), not just with a follow-up commit, then push.
- **Already pushed**: do not just push a fix. Tell the maintainer immediately: removal needs a history
  rewrite and possibly GitHub Support. Treat any exposed credential as compromised and rotate it.
