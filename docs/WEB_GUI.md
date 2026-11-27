# Browser GUI

VideoAtlas has a browser GUI that runs on the processing server. You open it in a browser on your own computer through an SSH tunnel. The server never opens a port to your network.

**Status: stage 1 of 5.** You can sign in, check that every model runs on the GPU, see free disk space, and follow or stop jobs. Viewing photo groups, the evaluation screen, uploads, and corrections come in later stages.

## Install it as a service

Run these commands inside the server container (`videoatlas.incus`, user `dev`). The GUI gets its own checkout, `~/videoatlas-app`, which stays on a released commit of `main`. Switching branches in the development checkout `~/videoatlas` cannot change or stop the running GUI. Analysis jobs use the code in `~/videoatlas-app` and the GPU environment in `~/videoatlas/.venv`.

1. Create the GUI checkout. Model files are not in Git, so link the folder that holds them.

   ```bash
   git -C ~/videoatlas fetch origin main
   git -C ~/videoatlas worktree add --detach ~/videoatlas-app origin/main
   ln -s ~/videoatlas/videoatlas/resources ~/videoatlas-app/videoatlas/resources
   ```

2. Create a separate environment for the GUI's packages, so they never change the analysis environment. Every package is pinned with a hash.

   ```bash
   cd ~/videoatlas-app
   uv venv --python 3.12 .venv-web
   uv pip install --python .venv-web/bin/python --require-hashes -r requirements-web.txt
   ```

3. Start it now and at every login. The server listens on `127.0.0.1:8765` only.

   ```bash
   mkdir -p ~/.config/systemd/user
   cp Scripts/videoatlas-web.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now videoatlas-web
   systemctl --user status videoatlas-web
   ```

   To keep it running after reboots without a login session, an administrator runs `sudo loginctl enable-linger dev` once.

**Update** to the latest `main`:

```bash
git -C ~/videoatlas fetch origin main
git -C ~/videoatlas-app checkout --detach origin/main
cd ~/videoatlas-app && uv pip install --python .venv-web/bin/python --require-hashes -r requirements-web.txt
cp Scripts/videoatlas-web.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user restart videoatlas-web
```

Restarting ends all sign-ins, and a job that was running is marked as interrupted.

**Remove** it. The data folder (`~/videoatlas-data`) is kept.

```bash
systemctl --user disable --now videoatlas-web
rm ~/.config/systemd/user/videoatlas-web.service
systemctl --user daemon-reload
git -C ~/videoatlas worktree remove --force ~/videoatlas-app
```

**Run it by hand** for development, from any checkout with the web packages installed:

```bash
.venv-web/bin/python -m videoatlas.web serve --engine-python ~/videoatlas/.venv/bin/python --engine-root .
```

## Open it from your Mac

1. Open a tunnel and leave the terminal window open:

   ```bash
   ssh -N -L 8765:127.0.0.1:8765 videoatlas.incus
   ```

2. Open <http://localhost:8765> in your browser.

3. Sign in with the access key. To see it, run this on the server. Run it from the GUI checkout, because Python must find the `videoatlas` package there:

   ```bash
   cd ~/videoatlas-app && .venv-web/bin/python -m videoatlas.web print-key
   ```

To replace the key, run `rotate-key` and restart the server. The running server accepts the old key until it restarts.

## How it is protected

- The server listens on a loopback address (`127.0.0.1`, `::1`, or `localhost`). It refuses to start on any other address.
- The SSH tunnel is the only way in, so your SSH key is the first lock.
- The access key is the second lock. Other programs on your Mac, and web pages open in your browser, can reach the forwarded port, but they do not know the key. The key file is readable only by its owner.
- A session cookie is sent only to this site (`SameSite=Strict`) and is hidden from scripts (`HttpOnly`). Sessions last up to seven days and end when the server restarts.
- Every change needs the session's CSRF token and must come from the GUI's own pages. Requests with another site's `Origin` are refused.
- Only the Host names `localhost`, `127.0.0.1`, and `[::1]` are accepted, which blocks DNS rebinding.
- The pages load no script from the internet. A Content Security Policy allows only the server's own scripts and styles.
- Jobs run as separate processes, one at a time, so a model crash cannot stop the GUI and two jobs never compete for the GPU. A lock file stops a second GUI server from running jobs from the same data folder. The GPU check always requests the GPU and shows a CPU fallback as a failure.
- The server keeps no access log. Logs never contain the access key, face images, or face embeddings.

## Where data is kept

The GUI uses `~/videoatlas-data` (or the folder in `VIDEOATLAS_DATA_DIR`, or `--data-dir`):

| Path | Contents |
| --- | --- |
| `web/access-key` | The access key (owner-only permissions) |
| `web/jobs.sqlite3` | Queued, running, and finished jobs with their results and last log lines |
