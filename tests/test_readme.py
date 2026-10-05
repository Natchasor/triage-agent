import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_command_blocks_contain_no_comments():
    """macOS zsh does not treat '#' as a comment when pasted, so a trailing '# note' on a
    command is passed to that command as extra arguments and breaks it."""
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)

    assert blocks, "expected at least one bash block in the README"
    for block in blocks:
        for line in block.splitlines():
            assert "#" not in line, f"comment inside a command block: {line!r}"
