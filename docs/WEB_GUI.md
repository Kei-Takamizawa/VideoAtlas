# Browser GUI

VideoAtlas has a browser GUI that runs on the processing server. You open it in a browser on your own computer through an SSH tunnel. The server never opens a port to your network.

**Status: stage 1 of 5.** You can sign in, check that every model runs on the GPU, see free disk space, and follow or stop jobs. Viewing photo groups, the evaluation screen, uploads, and corrections come in later stages.

## Start the server

Run these commands inside the server container (`videoatlas.incus`, user `dev`) from the repository folder `~/videoatlas`.

1. Install the web dependencies into the existing environment. Every package is pinned with a hash.

   ```bash
   uv pip install --python .venv/bin/python --require-hashes -r requirements-web.txt
   ```

2. Start the server. It listens on `127.0.0.1:8765` only.

   ```bash
   .venv/bin/python -m videoatlas.web serve
   ```

3. Optional: start it automatically. The unit file explains each step.

   ```bash
   mkdir -p ~/.config/systemd/user
   cp Scripts/videoatlas-web.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now videoatlas-web
   ```

   To keep it running after reboots without a login session, an administrator runs `sudo loginctl enable-linger dev` once.

To keep the GUI's packages out of the analysis environment, install `requirements-web.txt` into a separate environment and tell the server where the analysis runs:

```bash
.venv-web/bin/python -m videoatlas.web serve --engine-python ~/videoatlas/.venv/bin/python --engine-root ~/videoatlas
```

## Open it from your Mac

1. Open a tunnel and leave the terminal window open:

   ```bash
   ssh -N -L 8765:127.0.0.1:8765 videoatlas.incus
   ```

2. Open <http://localhost:8765> in your browser.

3. Sign in with the access key. To see it, run this on the server. Run it from the repository folder, because Python must find the `videoatlas` package there:

   ```bash
   cd ~/videoatlas && .venv/bin/python -m videoatlas.web print-key
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
