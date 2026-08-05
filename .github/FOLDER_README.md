# Maintenance tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines who may trigger expensive LangSlanger CI. Keep the
list explicit and small. `update_ci_permission.py` targets this repository when
a maintainer deliberately refreshes contributor permissions.

## Others
- `MAINTAINER.md` defines the code maintenance model.
