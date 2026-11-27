# Security policy

Conjectra runs code written by models inside git worktrees on your machine. Treat every backend
as able to execute arbitrary commands with your user's permissions: run it on a dedicated
user or machine, keep secrets out of reach of the worktrees, and list sensitive folders in
`banned_paths` / `CONJECTRA_READ_BANS`.

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") on this repository. We aim to acknowledge reports
within a week.
