#!/usr/bin/env python3
"""Exercise the static research interface using only its checked-in artifacts."""

import functools
import json
import tempfile
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@contextmanager
def local_site():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT))
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/docs/"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def downloaded_json(page, selector, directory):
    with page.expect_download() as event:
        page.locator(selector).click()
    target = Path(directory) / event.value.suggested_filename
    event.value.save_as(target)
    return json.loads(target.read_text())


def no_overflow(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def main():
    data = json.loads((ROOT / "docs/evidence-audit-data.json").read_text())
    excerpts = {row["item_id"]: row for row in data["excerpts"]}
    items = {row["item_id"]: row for row in data["answer_space"]["items"]}
    with local_site() as base, tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            for item_id in data["answer_space"]["metadata_dependent_own_pass_ids"]:
                page.goto(
                    base
                    + "evidence-audit.html#question="
                    + item_id
                    + "&view=compare&compare=metadata"
                )
                page.wait_for_selector("html[data-audit-ready]")
                assert (
                    page.locator("#selected-question").text_content() == items[item_id]["question"]
                )
                assert page.locator(".compare-response .outcome").all_text_contents() == [
                    "LEXICAL PASS",
                    "NO LEXICAL PASS",
                ]
                assert (
                    page.locator(".compare-source blockquote").all_text_contents()
                    == [excerpts[item_id]["anchor_text"]] * 2
                )
                exported = downloaded_json(page, "#export-question", tmp)
                assert exported["comparison_pair"] == "metadata"
                assert exported["question"]["item_id"] == item_id
                for response, text_key, diagnostic in zip(
                    exported["responses"],
                    ("wrapped_text", "anchor_text"),
                    ("own_wrapped", "own_bare"),
                ):
                    assert response["selected_excerpt"]["item_id"] == item_id
                    assert response["exact_scored_text"] == excerpts[item_id][text_key]
                    assert response["diagnostics"] == items[item_id][diagnostic]
                no_overflow(page)
            page.select_option("#comparison-pair", "ranking")
            link = page.url
            page.reload()
            page.wait_for_selector("html[data-audit-ready]")
            assert page.url == link
            assert page.locator("#comparison-pair").input_value() == "ranking"
            page.locator("#open-counterexample").click()
            assert page.locator("#source-id").text_content() == "EXCERPT hn-001"
            exported = downloaded_json(page, "#export-question", tmp)
            assert exported["selected_excerpt"]["item_id"] == "hn-001"
            assert exported["exact_scored_text"] == excerpts["hn-001"]["wrapped_text"]
            page.locator('[data-mode="compare"]').click()
            assert "Counterexample view closed" in page.locator("#toast").text_content()
            no_overflow(page)
        # Bypass the 63-row ledger and retain the selected evidence state.
        page.goto(base + "evidence-audit.html#question=cs-004&view=compare&compare=metadata")
        page.wait_for_selector("html[data-audit-ready]")
        selected_url = page.url
        page.locator("#question-search").focus()
        page.keyboard.press("Tab")
        assert page.locator("#skip-to-evidence").evaluate("e => e === document.activeElement")
        page.keyboard.press("Enter")
        assert page.locator("#inspector").evaluate("e => e === document.activeElement")
        assert page.url == selected_url
        page.keyboard.press("Tab")
        assert page.locator("#copy-link").evaluate("e => e === document.activeElement")
        page.keyboard.press("Tab")
        assert page.locator("#tab-compare").evaluate("e => e === document.activeElement")
        page.keyboard.press("Home")
        assert page.locator("#tab-bm25").get_attribute("aria-selected") == "true"
        page.keyboard.press("End")
        assert page.locator("#tab-compare").get_attribute("aria-selected") == "true"
        page.locator('[data-question="hc-005"]').focus()
        page.keyboard.press("Enter")
        assert page.locator("#inspector").evaluate("e => e === document.activeElement")
        assert "question=hc-005" in page.url
        for name, payload in (("http", None), ("schema", {"schema": "invalid"})):
            page.route(
                "**/evidence-audit-data.json",
                lambda route, _request, value=payload: route.fulfill(
                    status=503 if value is None else 200,
                    content_type="application/json",
                    body=json.dumps(value),
                ),
            )
            page.goto(base + "evidence-audit.html?failure=" + name)
            page.wait_for_selector("html[data-audit-error]")
            assert page.locator("#comparison-pair").is_disabled()
            assert not page.locator("#comparison-panel").is_visible()
            assert not page.locator("#evidence-panel").is_visible()
            no_overflow(page)
            page.unroute("**/evidence-audit-data.json")
        assert not errors, errors
        browser.close()
    print(
        "PASS: paired exact evidence/gates/export; restored link; counterexample; unavailable state; desktop/mobile."
    )


if __name__ == "__main__":
    main()
