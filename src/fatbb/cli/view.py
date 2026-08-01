"""Prompt-toolkit rendering kept separate from UI state transitions."""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML

from .controller import CliController


def header(controller: CliController) -> HTML:
    state = controller.state
    current = state.active_knowledge_base_name or "No knowledge base selected"
    return HTML(f"<b>FatBB</b>  <ansibrightblack>· {current}</ansibrightblack>")


def body(controller: CliController) -> HTML:
    state = controller.state
    parts: list[str] = []
    if controller.is_progress_page():
        parts.extend(("Creating knowledge base", state.progress or "Starting…"))
    elif controller.is_home_page():
        parts.append(state.status)
        if state.progress:
            parts.append(f"  {state.progress}")
        if state.lines:
            parts.append("")
            parts.extend(state.lines)
    elif state.status.startswith("Error:"):
        parts.append(state.status)
    if hint := controller.page_hint():
        parts.append(hint)
    if controller.items() and not controller.is_palette_page():
        parts.extend(("", _menu_text(controller)))

    # Build raw text, escaping user content while preserving HTML colour tags
    # that were embedded by ``_format_evidence``.
    text = "\n".join(
        part if _has_html_tags(part) else _escape(part)
        for part in parts
    )
    raw_lines = text.split("\n")

    # Because ``wrap_lines=False`` on the Window, we must wrap long lines
    # ourselves.  This also lets us compute the *visual* line count so that
    # ``max_scroll`` reflects what the user actually sees on screen.
    import shutil
    import textwrap
    term_size = shutil.get_terminal_size()
    width = max(40, term_size.columns - 1)  # one-column safety margin
    visible_height = max(1, term_size.lines - 3)  # header + prompt area

    wrapped: list[str] = []
    for line in raw_lines:
        # Strip prompt-toolkit HTML tags to measure true displayed length.
        # A line with tags is always wrapped as a single logical line (the
        # tags don't produce visible width), so we wrap by character count
        # on the plain-text form then re-apply the original markup on the
        # first wrapped segment.
        plain = _strip_html(line)
        if len(plain) <= width:
            wrapped.append(line)
        else:
            # Wrap the plain text, then insert the original (tagged) line
            # as the first segment and append continuation lines.
            plain_wrapped = textwrap.wrap(plain, width=width)
            if _has_html_tags(line):
                wrapped.append(line)
                wrapped.extend(plain_wrapped[1:])
            else:
                wrapped.extend(plain_wrapped)

    max_scroll = max(0, len(wrapped) - visible_height)
    offset = min(state.scroll_offset, max_scroll)
    visible = wrapped[offset:]

    if offset > 0:
        visible.insert(0, "--- scrolled up (PageDown to return) ---")

    return HTML("\n".join(visible))


def palette(controller: CliController) -> HTML:
    """Render the command palette that floats above the chat view."""
    return HTML("\n".join(_escape(part) for part in ("Command palette", "", _menu_text(controller))))


def prompt(controller: CliController) -> str:
    return "" if controller.is_progress_page() else "FatBB > "


def _menu_text(controller: CliController) -> str:
    return "\n".join(
        f"{'>' if index == controller.state.selected_index else ' '} {item}"
        for index, item in enumerate(controller.items())
    )


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _has_html_tags(value: str) -> bool:
    """Return ``True`` when *value* embeds prompt-toolkit HTML colour tags."""
    return "<ansi" in value


def _strip_html(value: str) -> str:
    """Remove prompt-toolkit HTML tags, leaving only visible text.

    >>> _strip_html("<ansiblue>Hello</ansiblue> World")
    'Hello World'
    """
    import re
    return re.sub(r"</?[a-zA-Z][^>]*>", "", value)
