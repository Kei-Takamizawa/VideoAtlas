# Vendored front-end files

The GUI loads no script from the internet. These files are copied from their npm packages and checked against the registry.

| File | Package | Published | SHA-256 |
| --- | --- | --- | --- |
| `htmx-2.0.11.min.js` | [htmx.org 2.0.11](https://www.npmjs.com/package/htmx.org/v/2.0.11), `dist/htmx.min.js` | 2026-09-22 | `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717` |

The npm tarball's SHA-1 (`a85363c4c5af94cd634df04c4b83d06dd0e91aa7`) matched the registry's `dist.shasum` when the file was copied. htmx is licensed under the Zero-Clause BSD license; see `htmx-LICENSE.txt`.

To update, choose a release at least 14 days old, copy `dist/htmx.min.js` from its tarball after checking the tarball against the registry, rename it with the new version, and update the template reference and this table.
