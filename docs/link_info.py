"""Sphinx linkcode_resolve function to link to GitHub source code."""

import importlib as il
import inspect
import pathlib as pl

this_dir = pl.Path(__file__).parent  # location of conf.py
# project_root should be set to the root of the git repo
project_root = this_dir.parent
# module_src_abs_paths should be a list of absolute paths to folders which contain modules
module_src_abs_paths = [project_root / "src"]


# https://stackoverflow.com/questions/48298560/how-to-add-link-to-source-code-in-sphinx
def linkcode_resolve_file_suffix(domain, info):
    if domain != "py":
        return None
    modulename, fullname = info.get("module", None), info.get("fullname", None)
    if not modulename and not fullname:
        return None
    filepath = None

    # first, let's get the file where the object is defined

    # import the module containing a reference to the object
    module = il.import_module(modulename)

    # We don't know if the object is a class, module, function, method, etc.
    # The module name given might also not be where the object code is.
    # For instance, if `module` imports `obj` from `module.submodule.obj`.
    objname = fullname.split(".")[0]  # first level object is guaranteed to be in module
    obj = getattr(module, objname)  # get the object, i.e. `module.obj`
    # inspect will find the canonical module for the object
    realmodule = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
    if realmodule is None or realmodule.__file__ is None:
        return
    abspath = pl.Path(
        realmodule.__file__
    )  # absolute path to the file containing the object
    # If the package was installed via pip, then the abspath here is
    # probably in a site-packages folder.

    # Let's find the abspath relative to the location of the top-level module.
    toplevel_name = modulename.split(".")[0]
    toplevel_module = il.import_module(toplevel_name)
    toplevel_paths = [
        pl.Path(path)
        for path in toplevel_module.__spec__.submodule_search_locations or []
    ]

    # There may be multiple top-level paths, so pick the first one that matches
    # the absolute path of the file we want to link to.
    for toplevel_path in toplevel_paths:
        toplevel_path = toplevel_path.parent
        if abspath.is_relative_to(toplevel_path):
            filepath = abspath.relative_to(toplevel_path)
            break

    # Now let's make it relative to the same directory in the correct src folder.
    for src_path in module_src_abs_paths:
        if not (src_path / filepath).exists():
            msg = f"Could not find {filepath} in {src_path}"
            raise FileNotFoundError(msg)
    src_rel = src_path.relative_to(project_root)  # get rid of the path anchor
    # src_rel is now the relative path from the project_root folder to the correct module folder
    filepath = (src_rel / filepath).as_posix()

    # now, let's try to get the line number where the object is defined

    # If fullname is something like `MyClass.property`, getsourcelines() will fail.
    # In this case, let's return the next best thing, which in this case is the line number of the class.
    name_parts = fullname.split(".")  # get the different components to check

    obj = module  # start with the module
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except TypeError:
        lineno = None  # default to no line number
    # try getting line number for each component and stop on failure
    for child_name in name_parts:
        try:
            child = getattr(obj, child_name)  # get the next level object
        except AttributeError:
            print(f"Failed to resolve {objname}.{child_name}")
            break
        try:
            lineno = inspect.getsourcelines(child)[
                1
            ]  # getsourcelines returns [str, int]
        except Exception:
            # getsourcelines throws TypeError if the object is not a class, module, function, method
            # i.e. if it's a @property, float, etc.
            break  # if we can't get the line number, let it be that of the previous
        obj = child  # update the object to the next level

    suffix = f"#L{lineno}" if lineno else ""
    return f"{filepath}{suffix}"
