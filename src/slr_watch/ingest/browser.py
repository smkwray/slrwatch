from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


PLAYWRIGHT_INSTALL_HINT = (
    "Playwright browser automation requires the Python playwright package and a "
    "Chromium install. Run `python -m pip install playwright` and "
    "`python -m playwright install chromium` in the project venv."
)


def _sync_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised by live env only
        raise RuntimeError(PLAYWRIGHT_INSTALL_HINT) from exc
    return sync_playwright


@contextmanager
def chromium_page() -> Iterator[object]:
    sync_playwright = _sync_playwright()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                accept_downloads=True,
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/145.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
                viewport={"width": 1440, "height": 900},
            )
            page = context.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
            try:
                yield page
            finally:
                context.close()
                browser.close()
    except Exception as exc:  # pragma: no cover - exercised by live env only
        if "Executable doesn't exist" in str(exc):
            raise RuntimeError(PLAYWRIGHT_INSTALL_HINT) from exc
        raise
