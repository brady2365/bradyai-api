import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

from model import BradyAI
from research import research
from tokenizer import BPETokenizer


# ==========================================
# SETTINGS
# ==========================================

MODEL_FILE = "bradyai_v3.pt"

MEMORY_FILE = Path("brady_memory.json")

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.25
TOP_K = 10

PUBLIC_MODE = (
    os.environ.get("PUBLIC_MODE", "false").lower() == "true"
)


# ==========================================
# DEVICE
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BradyAI API on", device)


# ==========================================
# FLASK
# ==========================================

app = Flask(__name__)


# ==========================================
# MEMORY
# ==========================================

def load_memory():

    if not MEMORY_FILE.exists():
        return {}

    try:

        with MEMORY_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            value = json.load(file)

        return value if isinstance(value, dict) else {}

    except (
        OSError,
        json.JSONDecodeError
    ):

        return {}


session_memory = (
    {}
    if PUBLIC_MODE
    else load_memory()
)


def save_memory():

    if PUBLIC_MODE:
        return

    with MEMORY_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            session_memory,
            file,
            indent=2
        )


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    print("Loading model:", MODEL_FILE)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False
    )

    tokenizer = BPETokenizer(
        vocab_size=len(
            checkpoint["vocab"]
        )
    )

    tokenizer.vocab = checkpoint["vocab"]

    tokenizer.token_to_id = (
        checkpoint["token_to_id"]
    )

    tokenizer.id_to_token = {
        int(key): value
        for key, value
        in checkpoint["id_to_token"].items()
    }

    tokenizer.merges = [
        tuple(merge)
        for merge in checkpoint["merges"]
    ]

    config = checkpoint["config"]

    model = BradyAI(
        vocab_size=len(tokenizer),
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print("BradyAI model loaded.")

    return (
        model,
        tokenizer,
        config["block_size"]
    )


model, tokenizer, block_size = load_model()

EOS_ID = tokenizer.token_to_id.get(
    "<EOS>"
)


# ==========================================
# RESEARCH DETECTION
# ==========================================

def needs_research(user_text):

    text = user_text.lower().strip()

    commands = (
        "research ",
        "look up ",
        "look it up ",
        "search for ",
        "search the web ",
        "search online ",
        "find information about ",
        "find info about ",
    )

    current_words = (
        "latest",
        "today",
        "tonight",
        "yesterday",
        "this week",
        "this month",
        "currently",
        "current",
        "recent",
        "newest",
        "recently",
    )

    return (
        text.startswith(commands)
        or any(
            word in text
            for word in current_words
        )
    )


def clean_research_query(user_text):

    text = user_text.strip()

    prefixes = (
        "research ",
        "look up ",
        "look it up ",
        "search for ",
        "search the web ",
        "search online ",
        "find information about ",
        "find info about ",
    )

    for prefix in prefixes:

        if text.lower().startswith(prefix):

            return text[
                len(prefix):
            ].strip()

    return text


# ==========================================
# RESEARCH RESPONSE
# ==========================================

def build_research_answer(result):

    sources = result.get(
        "sources",
        []
    )

    if not sources:

        return (
            "I could not find reliable "
            "sources for that. Try different wording."
        )

    parts = [
        "Here is what I found:"
    ]

    for source in sources:

        snippet = re.sub(
            r"https?://\S+",
            "",
            source.get(
                "snippet",
                ""
            )
        ).strip()

        if not snippet:
            continue

        lower = snippet.lower()

        teaser = (
            snippet.count("?") > 0
            and not any(
                word in lower
                for word in (
                    "uses",
                    "qubit",
                    "superposition",
                    "entanglement",
                    "process",
                    "information",
                )
            )
        )

        if teaser:
            continue

        snippet = snippet[:500]

        last_stop = max(
            snippet.rfind("."),
            snippet.rfind("!"),
            snippet.rfind("?")
        )

        if last_stop > 120:

            snippet = snippet[
                :last_stop + 1
            ]

        parts.append(
            "- " + snippet
        )

        if len(parts) >= 4:
            break

    if len(parts) <= 1:

        return (
            "I found sources but no "
            "useful summary."
        )

    return "\n\n".join(parts)


# ==========================================
# MEMORY
# ==========================================

def memory_reply(user_text):

    text = user_text.strip()
    lower = text.lower()

    memory_phrases = (
        "my name is",
        "call me",
        "favorite color",
        "i am learning",
        "remember ",
        "forget ",
        "show my notes",
        "list my notes",
        "what notes do you remember",
        "what do you remember",
        "what is my name",
        "remember my name",
        "what am i learning",
    )

    if (
        PUBLIC_MODE
        and any(
            phrase in lower
            for phrase in memory_phrases
        )
    ):

        return (
            "Personal memory is available "
            "in the local BradyAI app, but "
            "is disabled on this public demo "
            "for privacy."
        )

    # ------------------------------
    # NAME
    # ------------------------------

    name_match = re.search(
        r"^(?:my name is|call me)\s+"
        r"([A-Za-z][A-Za-z'-]*)[.!]?$",
        text,
        re.I
    )

    if name_match:

        name = name_match.group(1)

        session_memory["name"] = name

        save_memory()

        return (
            f"Nice to meet you, {name}. "
            "I will remember your name."
        )

    # ------------------------------
    # FAVORITE COLOR
    # ------------------------------

    color_match = re.search(
        r"^my favorite colo[u]?r is\s+"
        r"([A-Za-z]+)[.!]?$",
        text,
        re.I
    )

    if color_match:

        color = (
            color_match.group(1)
            .lower()
        )

        session_memory[
            "favorite_color"
        ] = color

        save_memory()

        return (
            "I will remember that your "
            f"favorite color is {color}."
        )

    # ------------------------------
    # LEARNING
    # ------------------------------

    learning_match = re.search(
        r"^i am learning\s+"
        r"(.+?)[.!]?$",
        text,
        re.I
    )

    if learning_match:

        subject = (
            learning_match
            .group(1)
            .strip()
        )

        session_memory[
            "learning"
        ] = subject

        save_memory()

        return (
            "I will remember that you "
            f"are learning {subject}."
        )

    # ------------------------------
    # REMEMBER NOTE
    # ------------------------------

    remember_match = re.search(
        r"^remember\s+(.+)$",
        text,
        re.I
    )

    if remember_match:

        note = (
            remember_match
            .group(1)
            .strip()
        )

        if note:

            notes = session_memory.setdefault(
                "notes",
                []
            )

            if note not in notes:

                notes.append(note)

                save_memory()

            return (
                "I will remember: "
                + note
            )

    # ------------------------------
    # FORGET
    # ------------------------------

    forget_match = re.search(
        r"^forget\s+(.+)$",
        text,
        re.I
    )

    if forget_match:

        requested = (
            forget_match
            .group(1)
            .strip()
        )

        notes = session_memory.get(
            "notes",
            []
        )

        if requested.lower() in (
            "all notes",
            "my notes",
        ):

            session_memory["notes"] = []

            save_memory()

            return (
                "I forgot all of your "
                "saved notes."
            )

        for note in notes:

            if note.lower() == requested.lower():

                notes.remove(note)

                save_memory()

                return (
                    "I forgot: "
                    + note
                )

        return (
            "I could not find that note. "
            "Use 'show my notes' to see "
            "saved notes."
        )

    # ------------------------------
    # NAME QUESTION
    # ------------------------------

    if (
        "what is my name" in lower
        or "remember my name" in lower
    ):

        if "name" in session_memory:

            return (
                "Your name is "
                + session_memory["name"]
                + "."
            )

        return (
            "You have not told me "
            "your name yet."
        )

    # ------------------------------
    # COLOR QUESTION
    # ------------------------------

    if (
        "what is my favorite color" in lower
        or "remember my favorite color" in lower
    ):

        if "favorite_color" in session_memory:

            return (
                "Your favorite color is "
                + session_memory[
                    "favorite_color"
                ]
                + "."
            )

        return (
            "You have not told me "
            "your favorite color yet."
        )

    # ------------------------------
    # LEARNING QUESTION
    # ------------------------------

    if (
        "what am i learning" in lower
        or "remember what i am learning" in lower
    ):

        if "learning" in session_memory:

            return (
                "You are learning "
                + session_memory[
                    "learning"
                ]
                + "."
            )

        return (
            "You have not told me "
            "what you are learning yet."
        )

    # ------------------------------
    # NOTES
    # ------------------------------

    if (
        "show my notes" in lower
        or "list my notes" in lower
        or "what notes do you remember"
        in lower
    ):

        notes = session_memory.get(
            "notes",
            []
        )

        if notes:

            return (
                "Here are your saved notes:\n- "
                + "\n- ".join(notes)
            )

        return (
            "You have no saved notes yet."
        )

    # ------------------------------
    # EVERYTHING
    # ------------------------------

    if "what do you remember" in lower:

        details = []

        if "name" in session_memory:

            details.append(
                "your name is "
                + session_memory["name"]
            )

        if "favorite_color" in session_memory:

            details.append(
                "your favorite color is "
                + session_memory[
                    "favorite_color"
                ]
            )

        if "learning" in session_memory:

            details.append(
                "you are learning "
                + session_memory[
                    "learning"
                ]
            )

        if session_memory.get("notes"):

            details.append(
                "your notes are: "
                + "; ".join(
                    session_memory["notes"]
                )
            )

        if details:

            return (
                "I remember that "
                + "; ".join(details)
                + "."
            )

        return (
            "I do not have saved "
            "details yet."
        )

    return None


# ==========================================
# GENERATE
# ==========================================

@torch.no_grad()
def generate(user_text):

    token_ids = tokenizer.encode(
        "User: "
        + user_text
        + "\nAssistant:",
        add_special_tokens=False,
    )

    token_ids.insert(
        0,
        tokenizer.token_to_id[
            "<BOS>"
        ]
    )

    token_ids = token_ids[
        -block_size:
    ]

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    generated = []

    for _ in range(MAX_NEW_TOKENS):

        logits, _ = model(
            x[:, -block_size:]
        )

        logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        k = min(
            TOP_K,
            logits.size(-1)
        )

        values, indices = torch.topk(
            logits,
            k
        )

        filtered = torch.full_like(
            logits,
            float("-inf")
        )

        filtered.scatter_(
            1,
            indices,
            values
        )

        next_token = torch.multinomial(
            F.softmax(
                filtered,
                dim=-1
            ),
            num_samples=1
        )

        token_id = next_token.item()

        if token_id == EOS_ID:
            break

        generated.append(
            token_id
        )

        x = torch.cat(
            [x, next_token],
            dim=1
        )

        decoded = tokenizer.decode(
            generated
        )

        if "User:" in decoded:
            break

    response = tokenizer.decode(
        generated
    )

    return (
        response
        .split("User:")[0]
        .split("\nAssistant:")[0]
        .strip()
    )


# ==========================================
# API ROUTES
# ==========================================

@app.get("/")
def home():

    return jsonify({
        "name": "BradyAI",
        "status": "online",
        "message": "BradyAI API is running."
    })


@app.get("/api")
def api_info():

    return jsonify({
        "name": "BradyAI API",
        "status": "online",
        "endpoints": [
            "/",
            "/api",
            "/api/chat",
            "/api/health"
        ]
    })


@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "model": MODEL_FILE,
        "device": device
    })


@app.post("/api/chat")
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    message = str(
        data.get(
            "message",
            ""
        )
    ).strip()

    if not message:

        return jsonify({
            "error":
                "Enter a message first."
        }), 400

    # ------------------------------
    # CLEAR MEMORY
    # ------------------------------

    if message.lower() == "clear memory":

        if PUBLIC_MODE:

            return jsonify({
                "reply":
                    "Personal memory is disabled "
                    "on this public demo.",
                "sources": []
            })

        session_memory.clear()

        save_memory()

        return jsonify({
            "reply":
                "Memory cleared.",
            "sources": []
        })

    # ------------------------------
    # RESEARCH
    # ------------------------------

    if needs_research(message):

        result = research(
            clean_research_query(message)
        )

        return jsonify({
            "reply":
                build_research_answer(
                    result
                ),
            "sources":
                result.get(
                    "sources",
                    []
                )
        })

    # ------------------------------
    # MEMORY
    # ------------------------------

    saved_reply = memory_reply(
        message
    )

    if saved_reply is not None:

        reply = saved_reply

    else:

        reply = generate(
            message
        )

    return jsonify({
        "reply":
            reply
            or "I do not know how to respond to that yet.",
        "sources": []
    })


# ==========================================
# LOCAL RUN
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
