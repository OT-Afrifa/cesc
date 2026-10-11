#!/usr/bin/env python
"""
setup_env.py
============

This environment module makes the locally installed ``cesc`` package importable
without having to manually modify ``sys.path`` in every notebook or script you use.

This helps because the cesc package exists here under the ``src`` directory here:
        cesc/
        ├── setup_env.py
        ├── pyproject.toml
        └── src/
            └── cesc/
                ├── contrib/
                    └── __init__.py
                    └── linker.py
                ├── __init__.py
                └── _compat.py
                └── adapters.py
                └── attribute.py
                └── cli.py
                └── detect.py
                └── distance.py
                └── environment.py
                └── gate.py
                └── params.py
                └── pipeline.py
                └── preflight.py
                └── quicklook.py
                └── thermo.py

Python must know about the ``src`` directory before `import cesc` works.

Here's how to get it set up:

===========================================================================
OPTION 1: PERMANENT SETUP FOR A PYTHON ENVIRONMENT
===========================================================================
Run:
    python setup_env.py

This is needed only once for each Python/Conda environment.

setup_env.py first attempts to install CESC in editable mode using pip so the
activated environment points to the source code in this project.

If you edit files under:
    src/cesc/

those edits are available the next time Python imports CESC.
No need reinstalling the package after every spurce-code change.
"""

import os
import sys
import subprocess
from pathlib import Path

# -----------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent  # Top-level CESC project directory
SRC = ROOT/'src'


def add_src_to_path():
    """
    Add 'src/' directory to 'sys.path' for the current Python process.

    This function is idempotent in that: * if 'src/' is already in 'sys.path',
                                        it is not added again.

    Returns
    -------
    True
        'src/' exists, regardless of whether it was newly added.

    False
        'src/' does not exist.
    """

    s = str(SRC)  # Convert pathlib Path to string format used by 'sys.path'

    if SRC.is_dir():
        if s not in sys.path:
            sys.path.insert(0, s)

        return True

    return False


def verify_import():
    """
    Test whether the CESC package can currently be imported.

    Also determines whether CESC has access to 'Numba' through
    the package compatibility module.

    Returns
    -------
    (version, numba_flag) if import succeeds. Example: ("1.0.0", True)

    (None, error) if importing CESC fails.

    """

    try:
        import cesc

        try:
            # Compatibility flag to indicate whether NUMBA is available
            # in the active Python environment
            from cesc._compat import HAVE_NUMBA

        except Exception:
            # Failure to determine Numba availability should not cause
            # the entire CESC import check to fail.
            HAVE_NUMBA = None

        return cesc.__version__, HAVE_NUMBA

    except Exception as exc:
        # Return the exception to the caller rather than stopping the
        # entire setup program here.
        return None, exc


def _site_packages_dir():
    """
    Fnd the site-packages directory belonging to the active Python environment.

    This location is needed only for the .pth fallback installation.

    The function first uses sysconfig, which is the preferred method.
    If that fails, it tries Python's site module.

    Returns
    -------
    str
        Path to site-packages if it can be determined.

    None
        No suitable site-packages directory could be identified.

    """
    try:
        import sysconfig
        # "purelib" normally points to the environment's site-packages
        # directory for pure Python packages.
        return sysconfig.get_paths()["purelib"]

    except Exception:
        import site
        # getsitepackages() may not exist in every Python configuration,
        # so getattr supplies an empty-list fallback.
        cands = getattr(site, "getsitepackages", lambda: [])()

        return cands[0] if cands else None


def write_pth_fallback():
    """
    Create a 'cesc.pth' file in site-packages.

    This is used only if the normal editable pip installation
    cannot be completed.

    A .pth file containing:

        /path/to/cesc/src

    tells Python to add that directory to its module search path whenever
    the environment starts.

    This makes:

        import cesc

    work in future sessions without requiring a package build.

    Important
    ---------
    The .pth method does NOT register the 'cesc' command-line tool.
    The editable pip installation is required for that.

    Returns
    -------
    (True, path)
        The .pth file was successfully written.

    (False, message)
        The .pth file could not be created

    """
    # Locate the site-packages directory for the Python currently running
    # this script.
    sp = _site_packages_dir()

    # We need both a valid directory and permission to write to it.
    if not sp or not os.access(sp, os.W_OK):

        return False, (
            "no writable site-packages found (%r); cannot write "
            "a .pth file. Use the per-session snippet in the "
            "docstring instead." % sp
        )

    # Example resulting file:
    #
    #     .../site-packages/cesc.pth
    #
    pth = Path(sp) / "cesc.pth"

    try:
        # Store the absolute src/ path followed by a newline.
        #
        # Python automatically processes .pth files when the environment
        # starts.
        pth.write_text(str(SRC) + "\n")

    except Exception as exc:  # noqa: BLE001

        return False, "could not write %s: %s" % (pth, exc)

    return True, str(pth)

'''
def editable_install():
    """
    Install this CESC package into the active Python enviroment
    in editable mode equivalent to 'python -m pip install -e /path/to/cesc'

    Several installation forms are attempted because HPC systems,
    Conda environments, older setuptools installations, and PEP 668
    protections can behave differently.

    Attempts
    --------
    1. Editable install without build isolation.

       This is preferred on HPC compute nodes because pip uses the
       setuptools already available in the environment rather than
       attempting to download build dependencies.

    2. Standard editable install.

    3. Same as attempt 1, but with --break-system-packages.

    4. Same as attempt 2, but with --break-system-packages.

    Returns
    -------
    0 if One of the installation attempts succeeded.

    nonzero integer if Every installation attempt failed.
    """

    base = [sys.executable, "-m", "pip", "-e", str(ROOT),]

    # Try several forms in order
    attempts = [
        # Preferred HPC/offline-friendly method.
        base + ["--no-build-isolation"],

        # Standard editable pip installation.
        base,

        # Some externally-managed Python installations require this.
        base + [
            "--no-build-isolation",
            "--break-system-packages",
        ],

        # Final installation attempt.
        base + ["--break-system-packages"],
    ]

    # Default to a failure code until one attempt succeeds.
    rc = 1

    for k, cmd in enumerate(attempts, start=1):
        # Display the installation form being attempted
        print(" attempt %d: %s" % (k, " ".join(cmd[3:])))

        rc = subprocess.call(cmd)  # run pip

        # Exit immediately after the first successful installation.
        if rc == 0:
            return 0

        print("  failed (exit %d)" % rc)

    # All attempts failed.
    return rc
'''
def editable_install():
    """
    Install the local CESC package in editable mode.

    The command is equivalent to:

        python -m pip install -e /path/to/cesc

    ``sys.executable`` ensures that pip belongs to the same Python
    environment that is currently running setup_env.py.

    On HPC systems, the first attempt disables build isolation so pip
    uses build tools already installed in the active environment rather
    than trying to download them.

    Returns
    -------
    0
        Editable installation succeeded.

    nonzero integer
        All installation attempts failed.
    """

    # This is the required beginning of every pip installation command.
    #
    # Example:
    #
    # /path/to/conda-env/bin/python
    #     -m pip install
    #
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
    ]

    # Editable installation of the current CESC project.
    editable_args = [
        "-e",
        str(ROOT),
    ]

    # Try the HPC-friendly form first. If that fails, try the standard
    # editable installation.
    attempts = [
        install_cmd + editable_args + ["--no-build-isolation"],
        install_cmd + editable_args,
    ]

    rc = 1

    for k, cmd in enumerate(attempts, start=1):

        print(
            "  attempt %d: %s"
            % (k, " ".join(cmd))
        )

        rc = subprocess.call(cmd)

        if rc == 0:
            return 0

        print(
            "    failed (exit %d)"
            % rc
        )

    return rc


def do_install():
    """
    Perform the permanent CESC environment setup

    Workflow
    --------
    1. Confirm src/ exists.
    2. Try an editable pip installation.
    3. Verify that ``import cesc`` works.
    4. If pip installation fails, try the .pth fallback.
    5. Verify the import again.
    6. Return success or failure to the shell.

    """
    if not SRC.is_dir():
        print(
            "ERROR: %s does not exist. Run this from the package root "
            "(the folder that contains pyproject.toml and src/)." % SRC
        )

        return 1

    # Show exactly which Python environment is being configured
    print("Setting up cesc for: %s" % sys.executable)
    print("Package root       : %s" % ROOT)

    # -----------------------------------------------------------------
    # First choice: editable pip installation
    # -----------------------------------------------------------------

    rc = editable_install()

    if rc == 0:
        # Also make src/ immediately available to this running Python
        # process. An editable installation may only become visible to
        # a future interpreter startup depending on the environment.
        add_src_to_path()

        ver, numba = verify_import()  # confirm CESC is importable
        if ver:
            print("\nDone. cesc %s is installed editable in this"
                  "environment." % ver)
            print("numba available  : %s" % numba)
            print("Every notebook and script here can now 'import cesc'"
                  "with no setup, and the 'cesc' command works.")
            return 0

        print("\nEditable install reported success but the import still"
              "fails: %s" % numba)

    # -----------------------------------------------------------------
    # Second choice: .pth fallback
    # -----------------------------------------------------------------
    print("\nEditable install did not work; trying the '.pth' fallback so"
          "import at least work without a build.")
    ok, info = write_pth_fallback()
    if ok:
        add_src_to_path()  # Make src/ visible in the current process as well.
        ver, numba = verify_import()  # Confirm that the package now imports.

        if ver:
            print("Wrote %s" % info)
            print(
                "cesc %s now imports in this environment (numba: %s). "
                "The `cesc` command is NOT registered by this route; "
                "re-run the editable install for that."
                % (ver, numba)
            )
            return 0

        print("Wrote the .pth but the import still fails: %s" % numba)
        return 1

    # Neither editable installation nor .pth setup succeeded.
    print("Fallback failed: %s" % info) 
    return 1


def do_check():
    """
    Inspect the current CESC environment configuration.

    This function does not install, remove, or modify anything.

    It reports:
        * project root,
        * whether src/ exists,
        * whether CESC imports,
        * CESC version,
        * where the imported package comes from,
        * whether Numba is available, and
        * whether a cesc.pth file exists.

    Returns
    -------
    0
        ``import cesc`` works.

    1
        ``import cesc`` currently fails.
    """
    print("Package root : %s" % ROOT)
    print("src/ exists  : %s" % SRC.is_dir())

    # Check the import before adding src/ to sys.path. This tells us
    # whether the active Python environment is already configured.
    ver, numba = verify_import()
    if ver:
        import cesc

        where = Path(cesc.__file__).resolve()

        # Determine whether Python resolved CESC to this project's src/
        # directory.
        editable = SRC in where.parents

        print("import cesc : OK, version %s" % ver)
        print(" resolves to: %s" % where)
        print(" source : %s" % (
            "this package (src/, editable or .pth)"
            if editable
            else "a different install"
                )
              )
        print(" numba : %s" % numba)
    else:
        print("import cesc : FAILS (%s)"
              % type(numba).__name__
              )
        print(
            "  Run `python setup_env.py` to fix it permanently, or use "
            "the per-session snippet in the docstring."
        )

    # -----------------------------------------------------------------
    # Check for the .pth fallback
    # -----------------------------------------------------------------

    sp = _site_packages_dir()

    pth = (
        Path(sp) / "cesc.pth"
        if sp
        else None
    )

    if pth and pth.exists():

        print(
            ".pth file    : %s -> %s"
            % (
                pth,
                pth.read_text().strip(),
            )
        )

    # Treat successful import as a successful check.
    return 0 if ver else 1


# =========================================================================
# FEW NOTES IF YOU DECIDE TO IMPORT THIS DIRECTLY INTO YOUR CODE
# =========================================================================
# 
# Example:
#
#     import setup_env
#
# In that situation:
#
#     __name__ != "__main__"
#
# so we do NOT run pip or write a .pth file.
#
# We only add this project's src/ directory to sys.path for the current
# Python process.

if __name__ != "__main__":
    add_src_to_path()


# =========================================================================
# FEW NOTES IF YOU DECIDE TO RUN THIS DIRECTLY
# =========================================================================
#
# Example:
#
#     python setup_env.py
#
# gives:
#
#     __name__ == "__main__"
#
# If --check is present, inspect the environment without changing it.
# Otherwise perform the permanent setup

if __name__ == "__main__":

    if "--check" in sys.argv:
        sys.exit(do_check())

    sys.exit(do_install())