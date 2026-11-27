# README artwork

| File | What it is | How to regenerate |
|---|---|---|
| `banner-{light,dark}.svg` | header banner | `python3 scripts/gen_readme_assets.py` |
| `how-it-works-{light,dark}.svg` | architecture diagram | `python3 scripts/gen_readme_assets.py` |
| `pipeline-{light,dark}.svg` | stage pipeline | `python3 scripts/gen_readme_assets.py` |
| `demo.svg` | animated terminal recording of `./examples/quickstart/run.sh` | see below |
| `demo.cast` | the asciinema recording `demo.svg` was rendered from | see below |

The README shows the light or dark variant through `<picture>` + `prefers-color-scheme`.

## Re-recording the demo

The recording is real output; do not edit numbers by hand.

1. Copy the repository to a neutral path (e.g. `/tmp/conjectra`) so no personal paths appear.
2. Record with [asciinema](https://asciinema.org) 2.x at 110 × 34, typing the command and then
   `head -n 9 .conjectra-demo/takeover.md` (run the script with `KEEP=1`):

   ```bash
   asciinema rec --cols 110 --rows 34 -c ./record.sh demo.cast
   ```

3. Shorten only the silent controller run: cap every pause in the cast at 2.5 s, add a 6 s hold
   on the last frame, and drop the `timestamp` and personal `env` entries from the header.
4. Render with [svg-term-cli](https://github.com/marionebl/svg-term-cli):

   ```bash
   svg-term --in demo.cast --out demo.svg --window --width 110 --height 34 --padding 18
   ```

5. Update the plain-text output block in the README to match.
