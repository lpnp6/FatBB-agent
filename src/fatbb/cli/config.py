"""Presentation-only configuration for the terminal user interface."""

from __future__ import annotations

from pathlib import Path
import tomllib


class ItemSource:
    """A configured ``module:function`` item-source handler."""

    def __init__(self, handler: str):
        self.handler = handler


class Command:
    """One configured UI command, visible in a menu or emitted by the state machine."""

    def __init__(
        self, value: str, *, label: str | None = None,
        handler: str | None = None, menu: str | None = None,
    ):
        self.value = value
        self.label = label
        self.handler = handler
        self.menu = menu


class Page:
    """A configured page and the interaction it accepts."""

    def __init__(self, interaction: str, submit_action: str | None = None, hint: str | None = None):
        self.interaction = interaction
        self.submit_action = submit_action
        self.hint = hint


class CliConfig:
    """Read menu labels and ordering without mixing them with capabilities."""

    def __init__(self, path: Path):
        # Load the source-controlled TOML document once; the remaining blocks
        # validate and normalize each top-level configuration section.
        payload = tomllib.loads(path.read_text(encoding="utf-8"))

        raw_cli = payload.get("cli")
        if not isinstance(raw_cli, dict):
            raise ValueError("CLI config requires a [cli] section.")
        self.home_page = self._required_string(raw_cli, "home_page", "CLI config")
        self.palette_page = self._required_string(raw_cli, "palette_page", "CLI config")

        # Parse [menus.*]. Each menu owns its static choices, including the
        # display label, stable value, and optional handler for that choice.
        raw_menus = payload.get("menus")
        if not isinstance(raw_menus, dict):
            raise ValueError("CLI config requires a [menus] section.")
        self._menus: dict[str, tuple[Command, ...]] = {}
        for name, raw_menu in raw_menus.items():
            if not isinstance(name, str) or not isinstance(raw_menu, dict):
                raise ValueError("Invalid CLI menu configuration.")
            items = raw_menu.get("items")
            if not isinstance(items, list):
                raise ValueError(f"CLI menu {name!r} requires an items list.")
            self._menus[name] = self._choices(items, f"CLI menu {name!r}")

        # Parse [item_sources.*]. These are dynamic option providers, used
        # when a page must show runtime data such as saved knowledge bases.
        raw_sources = payload.get("item_sources")
        if not isinstance(raw_sources, dict):
            raise ValueError("CLI config requires an [item_sources] section.")
        self._item_sources: dict[str, ItemSource] = {}
        for name, raw_source in raw_sources.items():
            if not isinstance(name, str) or not isinstance(raw_source, dict):
                raise ValueError("Invalid CLI item source configuration.")
            handler = raw_source.get("handler")
            if not isinstance(handler, str):
                raise ValueError(f"CLI item source {name!r} requires a handler string.")
            self._item_sources[name] = ItemSource(handler)

        # Parse [pages.*]. The page interaction and submit action let the
        # reducer work with arbitrary page ids instead of a Python enum.
        raw_pages = payload.get("pages")
        if not isinstance(raw_pages, dict):
            raise ValueError("CLI config requires a [pages] section.")
        self._page_sources: dict[str, str] = {}
        self._page_menus: dict[str, str] = {}
        self._pages: dict[str, Page] = {}
        for name, raw_page in raw_pages.items():
            if not isinstance(name, str) or not isinstance(raw_page, dict):
                raise ValueError("Invalid CLI page configuration.")
            item_source = raw_page.get("item_source")
            menu = raw_page.get("menu")
            interaction = self._required_string(raw_page, "interaction", f"CLI page {name!r}")
            submit_action = raw_page.get("submit_action")
            hint = raw_page.get("hint")
            if submit_action is not None and not isinstance(submit_action, str):
                raise ValueError(f"CLI page {name!r} submit_action must be a string.")
            if hint is not None and not isinstance(hint, str):
                raise ValueError(f"CLI page {name!r} hint must be a string.")
            if isinstance(item_source, str):
                self._page_sources[name] = item_source
            elif isinstance(menu, str):
                self._page_menus[name] = menu
            elif interaction == "menu":
                raise ValueError(f"CLI menu page {name!r} requires item_source or menu.")
            self._pages[name] = Page(interaction, submit_action, hint)

        for page_name in (self.home_page, self.palette_page):
            if page_name not in self._pages:
                raise ValueError(f"CLI config references unknown page: {page_name}")

        raw_routes = payload.get("routes")
        if not isinstance(raw_routes, dict):
            raise ValueError("CLI config requires a [routes] section.")
        self._routes: dict[str, str] = {}
        for name, target in raw_routes.items():
            if not isinstance(name, str) or not isinstance(target, str):
                raise ValueError("CLI routes must map string names to page ids.")
            if target not in self._pages:
                raise ValueError(f"CLI route {name!r} targets unknown page: {target}")
            self._routes[name] = target

        # Parse [actions.*]. An action either invokes a handler directly or
        # resolves the selected choice from a configured menu before invoking
        # that choice's handler.
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, dict):
            raise ValueError("CLI config requires an [actions] section.")
        self._actions: dict[str, Command] = {}
        for name, raw_action in raw_actions.items():
            if not isinstance(name, str) or not isinstance(raw_action, dict):
                raise ValueError("Invalid CLI action configuration.")
            handler, menu = raw_action.get("handler"), raw_action.get("menu")
            if isinstance(handler, str) and menu is None:
                self._actions[name] = Command(name, handler=handler)
            elif isinstance(menu, str) and handler is None:
                self._actions[name] = Command(name, menu=menu)
            else:
                raise ValueError(f"CLI action {name!r} requires exactly one handler or menu.")

    def menu_items(self, page: str) -> tuple[Command, ...] | None:
        """Return the static configuration-defined options for one page."""
        menu_name = self._page_menus.get(page)
        if menu_name is None:
            return None
        try:
            return self._menus[menu_name]
        except KeyError as error:
            raise ValueError(f"Unknown CLI menu: {menu_name}") from error

    def item_source(self, page: str) -> ItemSource | None:
        """Return the configured item-source definition for a page."""
        source_name = self._page_sources.get(page)
        return self._item_sources.get(source_name) if source_name is not None else None

    def action(self, name: str) -> Command | None:
        """Return the configuration definition for one state-machine action."""
        return self._actions.get(name)

    def page(self, name: str) -> Page:
        """Return one configured page, rejecting invalid navigation targets."""
        try:
            return self._pages[name]
        except KeyError as error:
            raise ValueError(f"Unknown CLI page: {name}") from error

    def route(self, name: str) -> str:
        """Resolve a semantic navigation route to its configured page id."""
        try:
            return self._routes[name]
        except KeyError as error:
            raise ValueError(f"Unknown CLI route: {name}") from error

    @staticmethod
    def _required_string(payload: dict[str, object], key: str, context: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{context} requires a {key} string.")
        return value

    @staticmethod
    def _choices(items: list[object], context: str) -> tuple[Command, ...]:
        choices: list[Command] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{context} has an invalid choice.")
            value, label, handler = item.get("value"), item.get("label"), item.get("handler")
            if not isinstance(value, str) or not isinstance(label, str):
                raise ValueError(f"{context} choices require value and label strings.")
            if handler is not None and not isinstance(handler, str):
                raise ValueError(f"{context} choice handlers must be strings.")
            choices.append(Command(value, label=label, handler=handler))
        return tuple(choices)
