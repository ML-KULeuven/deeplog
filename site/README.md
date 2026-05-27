# Deeplog Site

## Requirements
All requirements are listed in the [requirements.txt](requirements.txt) file,
and can be installed with:
```
pip install -r requirements.txt
```
Note that the deeplog package is a requirement. If you want to build for the latest version,
 you need to install it manually.

## Building
Build the static files using the command:
```bash
make html
```

## Versioned builds
Generate the version switcher data and build all tagged releases plus the nightly
branch with:
```bash
make multiversion
```

If you are hosting under a subpath (e.g., GitHub Pages), set a base URL so the
switcher links resolve correctly:
```bash
DOCS_BASE_URL=/deeplog/ make multiversion
```
