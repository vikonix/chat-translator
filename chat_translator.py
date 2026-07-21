"""
Chat Translator - a small multilingual translation helper for chat.

It talks to a local LM Studio server (OpenAI-compatible API) and, for each input:
  1. Translates the source text into the target language using a selected
     "purpose" (register / style).
  2. Produces a back-translation into the source language, so a user who does
     not read the target language can verify that the meaning was preserved.

The whole UI is in English. Language names sent to the model are English on
purpose, because models follow English language names more reliably.

Run:
    pip install gradio openai
    # Start LM Studio, load a model, and enable its local server (Developer tab).
    python chat_translator.py
"""

import collections
import json
import os
from datetime import datetime

import gradio as gr
from openai import OpenAI


# --- LM Studio connection -------------------------------------------------

# LM Studio exposes an OpenAI-compatible server. The base URL and the dummy API
# key below are LM Studio defaults. LM Studio ignores the exact model name and
# uses whatever model is currently loaded, so MODEL is just a placeholder.
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL = "local-model"

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")


# --- Configuration --------------------------------------------------------

# Supported languages. Names stay in English on purpose (see module docstring).
LANGUAGES = ["English", "Russian", "German", "French", "Spanish", "Italian"]

# Short codes used only for the compact "Direction" column in the History table
# (e.g. "RU-EN"). Any language missing here falls back to its first two letters.
LANGUAGE_CODES = {
    "English": "EN",
    "Russian": "RU",
    "German": "DE",
    "French": "FR",
    "Spanish": "ES",
    "Italian": "IT",
}


def _lang_code(name):
    """Return the short code for a language name (e.g. "Russian" -> "RU")."""
    return LANGUAGE_CODES.get(name, name[:2].upper())


# --- Persisted settings ---------------------------------------------------

# The user's last choices (languages, style, and whether the settings panel is
# expanded) are stored next to this script so they are restored on next launch.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "src_lang": "Russian",
    "tgt_lang": "English",
    "purpose": "universal",
    "settings_open": False,
}


def load_settings():
    """Return persisted settings, falling back to defaults for anything missing
    or invalid (e.g. a language that is no longer supported)."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return settings

    if saved.get("src_lang") in LANGUAGES:
        settings["src_lang"] = saved["src_lang"]
    if saved.get("tgt_lang") in LANGUAGES:
        settings["tgt_lang"] = saved["tgt_lang"]
    if saved.get("purpose") in PURPOSES:
        settings["purpose"] = saved["purpose"]
    if isinstance(saved.get("settings_open"), bool):
        settings["settings_open"] = saved["settings_open"]
    return settings


def save_settings(src_lang, tgt_lang, purpose, settings_open):
    """Write the current settings to disk. Best effort: write errors are ignored
    so a read-only working directory never breaks the app."""
    data = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "purpose": purpose,
        "settings_open": settings_open,
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

# Each purpose maps a UI label to a style instruction. The instructions are
# language-neutral: they describe *how* to translate, while the target language
# is supplied separately. To change a mode's behaviour, edit its text here;
# the UI does not need to change.
PURPOSES = {
    "plain": "Use a simplified style: a limited, common vocabulary, short "
             "sentences (one idea each), active voice, and no idioms or rare words.",
    "universal": "Use a clear, natural, neutral style.",
    "chat": "Use a casual, conversational style suitable for a chat; "
            "contractions are fine.",
    "letter": "Use a polite, well-structured style suitable for a message or letter.",
    "formal": "Use a formal, professional style suitable for business "
              "correspondence; no slang or contractions.",
}

# How many past translations to keep in the History panel.
HISTORY_SIZE = 20

# Most-recent-first log of translations. A deque with maxlen drops old entries
# automatically, so the history never grows without bound.
history = collections.deque(maxlen=HISTORY_SIZE)


# --- Model calls ----------------------------------------------------------

def _chat(system_prompt, user_text):
    """Send a single-turn request to LM Studio and return the reply text.

    A low temperature is used so translations stay faithful rather than creative.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def translate_text(text, src_lang, tgt_lang, purpose):
    """Translate `text` from src_lang into tgt_lang using the selected style."""
    style = PURPOSES.get(purpose, "")
    system = (
        f"You are a translator. Translate the user's text from {src_lang} into "
        f"{tgt_lang}. {style} Output only the translation, with no notes or "
        f"explanations."
    )
    return _chat(system, text)


def back_translate(text, tgt_lang, src_lang):
    """Translate the target text back into the source language for verification.

    This is deliberately a separate, neutral request with no access to the
    original source text. That way the back-translation is an honest check of
    meaning, not a copy of the original wording.
    """
    system = (
        f"You are a translator. Translate the user's text from {tgt_lang} into "
        f"{src_lang}. Use a neutral, faithful style. Output only the translation."
    )
    return _chat(system, text)


# --- UI event handlers ----------------------------------------------------

def _history_rows():
    """Format the history deque as rows for the History table.

    Only the columns a reader actually needs are shown: a compact direction
    code (e.g. "RU-EN"), the source text, and the translation.
    """
    return [
        [f'{_lang_code(h["src"])}-{_lang_code(h["tgt"])}', h["source"], h["translation"]]
        for h in history
    ]


def run_translation(text, src_lang, tgt_lang, purpose):
    """Handle the Translate action: translate, back-translate, and log history.

    Returns the three panel values plus the refreshed history table.
    """
    text = (text or "").strip()
    if not text:
        return "", "", _history_rows()

    # Same language selected on both sides: nothing to translate, skip the model.
    if src_lang == tgt_lang:
        return text, text, _history_rows()

    translation = translate_text(text, src_lang, tgt_lang, purpose)
    check = back_translate(translation, tgt_lang, src_lang)

    history.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "src": src_lang,
        "tgt": tgt_lang,
        "purpose": purpose,
        "source": text,
        "translation": translation,
        "back": check,
    })
    return translation, check, _history_rows()


def _panel_labels(src_lang, tgt_lang):
    """Build label updates that show each panel's current language.

    The language selectors are hidden in a collapsed Accordion, so the panel
    titles are what tells the user which language is which. They are updated
    whenever the languages change or are swapped.
    """
    return (
        gr.update(label=f"Source ({src_lang})"),
        gr.update(label=f"Translation ({tgt_lang})"),
        gr.update(label=f"Back-translation ({src_lang})"),
    )


def swap_languages(src_lang, tgt_lang):
    """Swap the source and target languages and refresh the panel titles."""
    new_src, new_tgt = tgt_lang, src_lang
    return (new_src, new_tgt, *_panel_labels(new_src, new_tgt))


def relabel_panels(src_lang, tgt_lang):
    """Refresh the panel titles after a language dropdown changes."""
    return _panel_labels(src_lang, tgt_lang)


def clear_fields():
    """Empty the source, translation, and back-translation panels."""
    return "", "", ""


def persist_settings(src_lang, tgt_lang, purpose, settings_open):
    """Save the current settings after any of them change. No UI output."""
    save_settings(src_lang, tgt_lang, purpose, settings_open)


def on_settings_expand(src_lang, tgt_lang, purpose):
    """Remember that the settings panel was opened, and store True in state."""
    save_settings(src_lang, tgt_lang, purpose, True)
    return True


def on_settings_collapse(src_lang, tgt_lang, purpose):
    """Remember that the settings panel was collapsed, and store False in state."""
    save_settings(src_lang, tgt_lang, purpose, False)
    return False


# --- Interface ------------------------------------------------------------

with gr.Blocks(title="Chat Translator") as demo:
    gr.Markdown("# Chat Translator")

    # Restore the languages, style, and panel state saved on the last run.
    settings = load_settings()

    # Tracks whether the settings panel is expanded so its state can be saved
    # alongside the other settings when languages or style change.
    settings_open_state = gr.State(settings["settings_open"])

    # Languages and purpose are configured rarely, so they live in a collapsible
    # Accordion. The panel titles below show the current languages, so the user
    # still knows the direction without expanding this section.
    with gr.Accordion("Languages & purpose", open=settings["settings_open"]) as settings_panel:
        with gr.Row():
            src_lang = gr.Dropdown(
                LANGUAGES, value=settings["src_lang"], label="Source language"
            )
            tgt_lang = gr.Dropdown(
                LANGUAGES, value=settings["tgt_lang"], label="Target language"
            )
        purpose = gr.Dropdown(
            list(PURPOSES.keys()), value=settings["purpose"], label="Purpose"
        )

    # Swap flips the direction and is used often, so it stays visible alongside
    # Paste and Clear even when the settings above are collapsed.
    with gr.Row():
        swap_btn = gr.Button("Swap languages", scale=0)
        paste_btn = gr.Button("Paste", scale=0)
        clear_btn = gr.Button("Clear", scale=0)

    with gr.Row():
        source_box = gr.Textbox(
            label=f"Source ({settings['src_lang']})", lines=8,
            placeholder="Type text to translate...",
        )
        # The built-in "copy" button gives one-click "copy the translation".
        translation_box = gr.Textbox(
            label=f"Translation ({settings['tgt_lang']})", lines=8, buttons=["copy"]
        )
        back_box = gr.Textbox(label=f"Back-translation ({settings['src_lang']})", lines=8)

    translate_btn = gr.Button("Translate", variant="primary")

    gr.Markdown("### History")
    # Fixed relative column widths keep the table within the page width, so the
    # long Source/Translation columns wrap instead of forcing a horizontal
    # scrollbar. The narrow direction code needs only a sliver of space.
    history_table = gr.Dataframe(
        headers=["Dir", "Source", "Translation"],
        column_widths=["10%", "45%", "45%"],
        interactive=False,
        wrap=True,
    )

    # Wiring: connect buttons to their handlers.
    translate_btn.click(
        run_translation,
        inputs=[source_box, src_lang, tgt_lang, purpose],
        outputs=[translation_box, back_box, history_table],
    )
    swap_btn.click(
        swap_languages,
        inputs=[src_lang, tgt_lang],
        outputs=[src_lang, tgt_lang, source_box, translation_box, back_box],
    ).then(
        persist_settings,
        inputs=[src_lang, tgt_lang, purpose, settings_open_state],
    )
    # Paste runs entirely in the browser: the clipboard can only be read on the
    # client (navigator.clipboard), not by the Python backend. The JS reply
    # fills the Source box directly, so no Python handler is needed.
    paste_btn.click(
        None, None, source_box, js="() => navigator.clipboard.readText()"
    )
    clear_btn.click(
        clear_fields,
        outputs=[source_box, translation_box, back_box],
    )

    # Keep the panel titles in sync, and persist settings, when a language is
    # picked from a dropdown.
    for selector in (src_lang, tgt_lang):
        selector.change(
            relabel_panels,
            inputs=[src_lang, tgt_lang],
            outputs=[source_box, translation_box, back_box],
        ).then(
            persist_settings,
            inputs=[src_lang, tgt_lang, purpose, settings_open_state],
        )

    # Persist when the style changes.
    purpose.change(
        persist_settings,
        inputs=[src_lang, tgt_lang, purpose, settings_open_state],
    )

    # Persist the expanded/collapsed state of the settings panel.
    settings_panel.expand(
        on_settings_expand,
        inputs=[src_lang, tgt_lang, purpose],
        outputs=[settings_open_state],
    )
    settings_panel.collapse(
        on_settings_collapse,
        inputs=[src_lang, tgt_lang, purpose],
        outputs=[settings_open_state],
    )


if __name__ == "__main__":
    demo.launch()
