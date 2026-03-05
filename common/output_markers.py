"""Parse tool results for edit success and error markers."""

# File-mutating tools whose results carry a positive success marker.
# ``contains_error_marker`` is a NEGATIVE heuristic and misses failures whose
# message contains no error token (e.g. the syntax-guard "Edit rejected: it
# would make <file> unparseable" — file NOT written, yet no token matches).
# For these tools success must be decided by the positive marker instead.
FILE_EDIT_TOOLS = frozenset({"edit_file", "write_file", "revert_file"})

# Every FileInterfaceHarness success message for the tools above starts with
# one of these prefixes ("Successfully replaced ...", "Successfully wrote to
# ...", "Restored <file> to its original state ..."). Anything else — errors,
# rejections, blocks — means the file was not changed.
_FILE_EDIT_SUCCESS_PREFIXES = ("Successfully ", "Restored ")


def file_edit_applied(result: str) -> bool:
    """True when a file-edit tool's result reports an APPLIED change."""
    return str(result or "").lstrip().startswith(_FILE_EDIT_SUCCESS_PREFIXES)
