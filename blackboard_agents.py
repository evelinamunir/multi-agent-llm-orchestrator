# Imports
import argparse
import json
import os
import queue
import re
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional


# LLM Backend Layer
class LLMBackend:
    """Base interface for Large Language Model execution engines."""

    def generate(self, system: str, user: str, max_tokens: int = 512,
                 temperature: float = 0.4) -> str:
        """Generates text from the LLM given a system and user prompt."""
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    """Runs a local GGUF model in-process via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        """Initialises the llama.cpp backend with the specified model and context size."""
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(self, system, user, max_tokens=512, temperature=0.7):
        """Executes a chat completion using the local llama.cpp engine."""
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()


class TransformersBackend(LLMBackend):
    """Fallback: loads a Hugging Face model + tokeniser directly."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """ Initialises the Hugging Face tokeniser and model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate(self, system, user, max_tokens=512, temperature=0.7):
        """Executes a chat completion using the Hugging Face transformers pipeline."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True
            )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


def build_backend(model_path: str | None, hf_fallback: str) -> LLMBackend:
    """Try llama.cpp first, fall back to transformers if unavailable."""
    if model_path:
        try:
            return LlamaCppBackend(model_path)
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"[backend] llama-cpp-python unavailable ({e}); falling back to transformers.")
    return TransformersBackend(hf_fallback)


@dataclass
class Blackboard:
    """Shared memory space for all agents to read from and write to."""
    goal: str
    research_notes: str = ""
    draft: str = ""
    critique: list[str] = field(default_factory=list)
    revisions: int = 0
    transcript: list[dict] = field(default_factory=list)
    on_log: Optional[Callable[[dict], None]] = field(default=None, repr=False, compare=False)

    def log(self, agent: str, content: str):
        """Records an agent's action to the transcript and prints a summary."""
        entry = {"agent": agent, "content": content, "t": time.time()}
        self.transcript.append(entry)
        print(f"\n[{agent}]\n{content}")
        if self.on_log:
            try:
                self.on_log(entry)
            except Exception as e:
                print(f"[live] failed to stream entry to browser: {e}")

    def to_json(self, path: str = "transcript.json"):
        """Exports the goal and full transcript to JSON."""
        with open(path, "w") as f:
            json.dump({"goal": self.goal, "transcript": self.transcript}, f, indent=2)


def _extract_json(text: str) -> str:
    """Cleans up markdown code blocks around JSON outputs from the LLM."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return text


def _repair_json(text: str) -> str:
    """Best-effort repair for near-valid JSON emitted by small local models."""
    text = text.strip()
    text = re.sub(r'([}\]])\s*"+\s*([}\]])', r'\1\2', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    opens = {'{': '}', '[': ']'}
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in opens:
            stack.append(opens[ch])
        elif ch in (']', '}') and stack and stack[-1] == ch:
            stack.pop()
    if in_string:
        text += '"'
    text += ''.join(reversed(stack))
    return text


def safe_json_list(text: str) -> list[dict]:
    """Attempts to parse a JSON list; provides a robust fallback if the LLM hallucinates."""
    cleaned = _extract_json(text)
    for candidate in (cleaned, _repair_json(cleaned)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, TypeError):
            continue
    return [
        {"step": i + 1, "owner": "researcher", "task": line.strip("-* ")}
        for i, line in enumerate(text.splitlines()) if line.strip()
    ]


def safe_json_obj(text: str, default: dict) -> dict:
    """Attempts to parse a JSON object, returning a default dictionary on failure."""
    cleaned = _extract_json(text)
    for candidate in (cleaned, _repair_json(cleaned)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return default


class Agent:
    """Base agent class handling basic initialisation and execution."""
    name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def act(self, user_prompt: str, bb: Blackboard, max_tokens: int = 512) -> str:
        """Calls the LLM, logs the result to the blackboard, and returns the output."""
        out = self.backend.generate(self.system_prompt, user_prompt, max_tokens=max_tokens)
        bb.log(self.name, out)
        return out


class PlannerAgent(Agent):
    """Responsible for breaking down the main goal into individual steps."""
    name = "Planner"
    system_prompt = (
        "You are a project planner. Break the user's goal into 3-5 concrete, "
        "ordered subtasks for a research agent and a writer agent to execute. "
        'Respond ONLY with valid JSON: a list of objects like '
        '[{"step": 1, "owner": "researcher", "task": "..."}]. '
        "No prose and no markdown fences, just the JSON array."
    )

    def plan(self, bb: Blackboard) -> list[dict]:
        """Generates the plan based on the overarching goal."""
        raw = self.act(f"Goal: {bb.goal}", bb, max_tokens=400)
        return safe_json_list(raw)


class ResearcherAgent(Agent):
    """Responsible for executing 'researcher' tasks and gathering facts."""
    name = "Researcher"
    system_prompt = (
        "You are a careful researcher. Given a subtask, produce concise, "
        "factual bullet-point notes (5-8 bullets). Flag any claim you're "
        "unsure about with [UNVERIFIED]. Do not invent citations or sources."
    )

    def research(self, task: str, bb: Blackboard) -> str:
        """Gathers notes and appends them to the blackboard's existing research."""
        notes = self.act(
            f"Research subtask: {task}\n\nContext gathered so far:\n{bb.research_notes or '(none yet)'}",
            bb, max_tokens=400,
        )
        bb.research_notes += f"\n\n## {task}\n{notes}"
        return notes


class WriterAgent(Agent):
    """Responsible for turning notes into a full draft and applying edits."""
    name = "Writer"
    system_prompt = (
        "You are a clear, concise technical writer. Write or revise a draft "
        "that directly satisfies the goal, using the supplied research notes "
        "and any critique. Prefer plain language, short paragraphs, and "
        "concrete, concise detail. Return the draft text only, nothing else."
        "You must ONLY use the facts provided in the research notes. "
        "Never invent, fabricate, or hallucinate statistics, quotes, or external sources."
    )

    def draft(self, bb: Blackboard) -> str:
        """Generates the initial text based on the goal and gathered research."""
        prompt = f"Goal: {bb.goal}\n\nResearch notes:\n{bb.research_notes}\n\nWrite the first draft."
        bb.draft = self.act(prompt, bb, max_tokens=1800)
        return bb.draft

    def revise(self, bb: Blackboard) -> str:
        """Rewrites the current draft addressing the latest critic feedback."""
        critique_txt = "\n".join(f"- {c}" for c in bb.critique[-5:]) or "(none)"
        prompt = (
            f"Goal: {bb.goal}\n\nCurrent draft:\n{bb.draft}\n\n"
            f"Critique to address:\n{critique_txt}\n\nProduce a revised draft."
        )
        bb.draft = self.act(prompt, bb, max_tokens=1800)
        bb.revisions += 1
        return bb.draft


class CriticAgent(Agent):
    """Responsible for reviewing the draft and giving feedback for the the revision loop."""
    name = "Critic"
    system_prompt = (
        "You are a precise evaluator. Your ONLY job is to check whether the draft "
        "satisfies the user's original goal using ONLY the provided research notes. "
        "Only evaluate the draft against facts explicitly listed in the research notes. "
        "If the original goal asks for information not found in the notes, a draft that correctly states the information is unavailable is considered a perfect success. "
        "List AT MOST 3 issues, ranked most important first. "
        'Respond ONLY with raw JSON, on a single line, in this exact shape: '
        '{"score": 1-10, "verdict": "accept"|"revise", "issues": ["..."]}. '
        "No markdown, no prose, no extra keys, and make sure every bracket you open is closed."
    )

    def review(self, bb: Blackboard) -> dict:
        """Scores the draft and outputs structured revision issues."""
        prompt = (
            f"Goal: {bb.goal}\n\n"
            f"Research notes:\n{bb.research_notes}\n\n"
            f"Draft:\n{bb.draft}\n\n"
            "Remember: Evaluate only using the provided research notes. Do not ask the writer to add missing facts or concrete data if they are not in the notes."
        )
        raw = self.act(prompt, bb, max_tokens=600)
        verdict = safe_json_obj(
            raw,
            default={"score": 5, "verdict": "revise", "issues": ["Could not parse critique JSON"]},
        )
        bb.critique.extend(verdict.get("issues", []))
        return verdict

class LiveServer:
    """Serves blackboard_visualiser.html and streams transcript entries to it."""
    def __init__(self, html_path: str, goal: str, port: int = 8765):
        self.html_path = html_path
        self.goal = goal
        self.port = port
        self.entries: list[dict] = []
        self.finished_stats: Optional[dict] = None
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.httpd: Optional[ThreadingHTTPServer] = None

    def broadcast(self, entry: dict):
        with self.lock:
            self.entries.append(entry)
            for q in self.clients:
                q.put(entry)

    def finish(self, stats: dict):
        with self.lock:
            self.finished_stats = stats
            for q in self.clients:
                q.put(("__done__", stats))

    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass 

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    with open(server.html_path, "rb") as f:
                        self.wfile.write(f.read())
                    return

                if self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    q: queue.Queue = queue.Queue()
                    with server.lock:
                        server.clients.append(q)
                        backlog = list(server.entries)
                        already_done = server.finished_stats

                    try:
                        self._send_event("init", {"goal": server.goal})
                        for entry in backlog:
                            self._send_event("entry", entry)
                        if already_done is not None:
                            self._send_event("done", already_done)
                        else:
                            while True:
                                item = q.get()
                                if item is None:
                                    break
                                if isinstance(item, tuple) and item[0] == "__done__":
                                    self._send_event("done", item[1])
                                    continue
                                self._send_event("entry", item)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        with server.lock:
                            if q in server.clients:
                                server.clients.remove(q)
                    return

                self.send_response(404)
                self.end_headers()

            def _send_event(self, event: str, data: dict):
                payload = json.dumps(data)
                self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode())
                self.wfile.flush()

        last_err = None
        for port in range(self.port, self.port + 20):
            try:
                self.httpd = ThreadingHTTPServer(("localhost", port), Handler)
                self.port = port
                break
            except OSError as e:
                last_err = e
        else:
            raise RuntimeError(f"Could not bind a local port for --live: {last_err}")

        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()

    def stop(self):
        if self.httpd:
            with self.lock:
                for q in self.clients:
                    q.put(None)
            self.httpd.shutdown()


def run_task(goal: str, backend: LLMBackend, max_revisions: int = 4,
             on_log: Optional[Callable[[dict], None]] = None) -> Blackboard:
    """Joins the multi-agent workflow together."""
    bb = Blackboard(goal=goal, on_log=on_log)
    planner = PlannerAgent(backend)
    researcher = ResearcherAgent(backend)
    writer = WriterAgent(backend)
    critic = CriticAgent(backend)

    plan = planner.plan(bb)

    for step in plan:
        if str(step.get("owner", "")).lower() == "researcher":
            researcher.research(step.get("task", ""), bb)

    writer.draft(bb)

    for i in range(max_revisions):
        verdict = critic.review(bb)
        print(f"\n[orchestrator] round {i + 1} critic verdict: {verdict}")
        if verdict.get("score", 0) >= 8:
            break
        writer.revise(bb)

    return bb


def main():
    """Parses command-line arguments and starts the orchestration layer."""
    ap = argparse.ArgumentParser(description="Local multi-agent task planner.")
    ap.add_argument("goal", type=str, help="The task/goal for the agents to accomplish.")
    ap.add_argument("--model-path", type=str, default=None,
                     help="Path to a local .gguf model file, used with llama-cpp-python.")
    ap.add_argument("--hf-fallback", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                     help="Hugging Face model id used if llama-cpp-python/model-path isn't available.")
    ap.add_argument("--max-revisions", type=int, default=4)
    ap.add_argument("--transcript-out", type=str, default="transcript.json",
                     help="Where to save the run's transcript as JSON for blackboard_visualiser.html.")
    ap.add_argument("--live", action="store_true",
                     help="Open a live blackboard in your browser and stream each agent's output.")
    ap.add_argument("--port", type=int, default=8765,
                     help="Port for the live server (used with --live).")
    args = ap.parse_args()

    backend = build_backend(args.model_path, args.hf_fallback)

    server = None
    on_log = None
    if args.live:
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blackboard_visualiser.html")
        if not os.path.exists(html_path):
            print(f"[live] couldn't find blackboard_visualiser.html next to this script; continuing without --live.")
        else:
            server = LiveServer(html_path, args.goal, port=args.port)
            server.start()
            url = f"http://localhost:{server.port}/"
            print(f"[live] blackboard running at {url}")
            webbrowser.open(url)
            on_log = server.broadcast
            time.sleep(1.5)

    bb = run_task(args.goal, backend, max_revisions=args.max_revisions, on_log=on_log)

    print("\n\nFinal Draft\n")
    print(bb.draft)
    print(f"\n[stats] revisions: {bb.revisions}, transcript entries: {len(bb.transcript)}")

    if args.transcript_out:
        bb.to_json(args.transcript_out)
        print(f"[export] transcript saved to {args.transcript_out}.")

    if server:
        server.finish({"revisions": bb.revisions})
        print("[live] run complete — leaving the server up for a couple seconds.")
        time.sleep(2)
        server.stop()


if __name__ == "__main__":
    main()
