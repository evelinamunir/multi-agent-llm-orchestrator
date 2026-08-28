# Imports
import argparse
import json
import re
import time
from dataclasses import dataclass, field


# LLM Backend Layer
# Agents don't call the model directly, means that underlying engine can be swapped out easily
# Supports two engines with automatic fallback too
class LLMBackend:
    """Base interface for Large Language Model execution engines."""

    def generate(self, system: str, user: str, max_tokens: int = 512,
                 temperature: float = 0.4) -> str:
        """Generates text from the LLM given a system and user prompt."""
        raise NotImplementedError


# Implementation for running local .gguf models
class LlamaCppBackend(LLMBackend):
    """Runs a local GGUF model in-process via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        """Initialises the llama.cpp backend with the specified model and context size."""

        # Lazy import so that the rest of the script can still run if llama cpp isn't installed
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            # 0 = CPU only; raise this to offload to GPU
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(self, system, user, max_tokens=512, temperature=0.7):
        """Executes a chat completion using the local llama.cpp engine."""

        # Format the system and user prompts into a standard chat structure
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Extract and return just the generated text content
        return out["choices"][0]["message"]["content"].strip()


# Fallback implementation using standard Hugging Face transformers
class TransformersBackend(LLMBackend):
    """Fallback: loads a Hugging Face model + tokeniser directly (no GGUF
    file needed). However, heavier on RAM/VRAM than llama.cpp.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """ Initialises the Hugging Face tokeniser and model."""

        # Lazy imports again so that the rest of the script can still run if transformers isn't installed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        # Loads tokeniser and model, automatically routing to GPU if available
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate(self, system, user, max_tokens=512, temperature=0.7):
        """Executes a chat completion using the Hugging Face transformers pipeline."""

        # Constructs chat messages
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Applies the model's specific chat template to format the messages into a single prompt string
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenises the prompt and moves tensors to the correct device (CPU/GPU) for the model
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        # Generates the response without calculating gradients (saves memory)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True
            )
        # Decodes the output, stripping away the input prompt and special tokens
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

# Factory function to build the appropriate backend
def build_backend(model_path: str | None, hf_fallback: str) -> LLMBackend:
    """Try llama.cpp first (runs a local .gguf file),
    fall back to transformers if that's not installed or no model path given.
    """
    if model_path:
        try:
            return LlamaCppBackend(model_path)
        except (ImportError, ValueError, RuntimeError) as e:
            # Informs user if llama-cpp-python is unavailable or the model path is invalid, and falls back to transformers
            print(f"[backend] llama-cpp-python unavailable ({e}); falling back to transformers.")
    return TransformersBackend(hf_fallback)

# Blackboard: shared memory for all agents to read/write
@dataclass
class Blackboard:
    """Shared memory space for all agents to read from and write to."""
    goal: str
    # Holds subtasks
    plan: list[dict] = field(default_factory=list)
    # Aggregated research notes from the ResearcherAgent
    research_notes: str = ""
    # The working doc
    draft: str = ""
    # Critique from the CriticAgent
    critique: list[str] = field(default_factory=list)
    # Number of revisions made to the draft
    revisions: int = 0
    # Transcript of all agent interactions for debugging/logging
    transcript: list[dict] = field(default_factory=list)

    def log(self, agent: str, content: str):
        """Records an agent's action to the transcript and prints a summary."""
        self.transcript.append({"agent": agent, "content": content, "t": time.time()})
        print(f"\n[{agent}]\n{content}")


# Function to clean up code blocks around LLM outputs
def _extract_json(text: str) -> str:
    """Cleans up markdown code blocks around JSON outputs from the LLM."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return text


def safe_json_list(text: str) -> list[dict]:
    """Attempts to parse a JSON list; provides a robust fallback if the LLM hallucinates."""
    try:
        data = json.loads(_extract_json(text))
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, TypeError):
        # Fallback, manually parse each line if JSON decoding fails
        return [
            {"step": i + 1, "owner": "researcher", "task": line.strip("-* ")}
            for i, line in enumerate(text.splitlines()) if line.strip()
        ]


def safe_json_obj(text: str, default: dict) -> dict:
    """Attempts to parse a JSON object, returning a default dictionary on failure."""
    try:
        return json.loads(_extract_json(text))
    except (json.JSONDecodeError, TypeError):
        return default


# Agents

class Agent:
    """Base agent class handling basic initialisation and execution."""
    name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, backend: LLMBackend):
        """ Initialises the agent with a shared LLM backend."""

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
        """Generates the plan and updates the blackboard."""
        raw = self.act(f"Goal: {bb.goal}", bb, max_tokens=400)
        bb.plan = safe_json_list(raw)
        return bb.plan


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
        # Only considers the last 5 critique points to avoid overwhelming the model
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
        "satisfies the user's original goal. "
        "Do NOT invent requirements that aren't in the goal. "
        "Do NOT penalise for missing citations unless the goal asked for them. "
        "Do NOT raise a 'possible' issue unless it is clearly present in the draft. "
        'Respond ONLY with raw JSON: '
        '{"score": 1-10, "verdict": "accept"|"revise", "issues": ["..."]}. '
        "No markdown, no prose, no extra keys."
    )

    def review(self, bb: Blackboard) -> dict:
        """Scores the draft and outputs structured revision issues."""
        prompt = f"Goal: {bb.goal}\n\nResearch notes:\n{bb.research_notes}\n\nDraft:\n{bb.draft}"
        raw = self.act(prompt, bb, max_tokens=300)
        verdict = safe_json_obj(
            raw,
            default={"score": 5, "verdict": "revise", "issues": ["Could not parse critique JSON"]},
        )
        # Stores the issues on the blackboard for the writer to address in the next revision
        bb.critique.extend(verdict.get("issues", []))
        return verdict


# Orchestrator
def run_task(goal: str, backend: LLMBackend, max_revisions: int = 4) -> Blackboard:
    """
    Joins the multi-agent workflow together.
    Runs the pipeline: Plan, research, draft, (back and forth between critique and revise), finish
    """
    bb = Blackboard(goal=goal)

    # Initialise all agents with the shared backend
    planner = PlannerAgent(backend)
    researcher = ResearcherAgent(backend)
    writer = WriterAgent(backend)
    critic = CriticAgent(backend)

    # Step 1: Planner breaks the goal into subtasks
    plan = planner.plan(bb)

    # Step 2: Researcher executes each subtask and gathers notes
    for step in plan:
        if str(step.get("owner", "")).lower() == "researcher":
            researcher.research(step.get("task", ""), bb)

    # Step 3: Writer produces the initial draft based on the research notes
    writer.draft(bb)

    # Step 4: Loop of critic reviewing the draft and providing feedback, then writer revising as needed
    for i in range(max_revisions):
        verdict = critic.review(bb)
        print(f"\n[orchestrator] round {i + 1} critic verdict: {verdict}")

        # Break the loop early if the draft is accepted with a high enough score
        if verdict.get("verdict") == "accept" and verdict.get("score", 0) >= 8:
            break

        # Otherwise, run another revision cycle
        writer.revise(bb)

    return bb

# CLI execution

def main():
    """Parses command-line arguments and starts the orchestration layer."""
    ap = argparse.ArgumentParser(description="Local multi-agent task planner.")
    ap.add_argument("goal", type=str, help="The task/goal for the agents to accomplish.")
    ap.add_argument("--model-path", type=str, default=None,
                     help="Path to a local .gguf model file, used with llama-cpp-python.")
    ap.add_argument("--hf-fallback", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                     help="Hugging Face model id used if llama-cpp-python/model-path isn't available.")
    ap.add_argument("--max-revisions", type=int, default=4)
    args = ap.parse_args()

    # Determines which backend engine to use based on provided arguments and availability
    backend = build_backend(args.model_path, args.hf_fallback)

    # Runs the core multi-agent task planning loop
    bb = run_task(args.goal, backend, max_revisions=args.max_revisions)

    # Prints the final draft and execution statistics
    print("\n\nFinal Draft\n")
    print(bb.draft)
    print(f"\n[stats] revisions: {bb.revisions}, transcript entries: {len(bb.transcript)}")


if __name__ == "__main__":
    main()
    def generate(self, system, user, max_tokens=512, temperature=0.7):
        """Executes a chat completion using the local llama.cpp engine."""

        # Format the system and user prompts into a standard chat structure
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Extract and return just the generated text content
        return out["choices"][0]["message"]["content"].strip()


# Fallback implementation using standard Hugging Face transformers
class TransformersBackend(LLMBackend):
    """Fallback: loads a Hugging Face model + tokeniser directly (no GGUF
    file needed). However, heavier on RAM/VRAM than llama.cpp.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """ Initialises the Hugging Face tokeniser and model."""

        # Lazy imports again so that the rest of the script can still run if transformers isn't installed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        # Loads tokeniser and model, automatically routing to GPU if available
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate(self, system, user, max_tokens=512, temperature=0.7):
        """Executes a chat completion using the Hugging Face transformers pipeline."""

        # Constructs chat messages
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Applies the model's specific chat template to format the messages into a single prompt string
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenises the prompt and moves tensors to the correct device (CPU/GPU) for the model
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        # Generates the response without calculating gradients (saves memory)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True
            )
        # Decodes the output, stripping away the input prompt and special tokens
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

# Factory function to build the appropriate backend
def build_backend(model_path: str | None, hf_fallback: str) -> LLMBackend:
    """Try llama.cpp first (runs a local .gguf file),
    fall back to transformers if that's not installed or no model path given.
    """
    if model_path:
        try:
            return LlamaCppBackend(model_path)
        except (ImportError, ValueError, RuntimeError) as e:
            # Informs user if llama-cpp-python is unavailable or the model path is invalid, and falls back to transformers
            print(f"[backend] llama-cpp-python unavailable ({e}); falling back to transformers.")
    return TransformersBackend(hf_fallback)

# Blackboard: shared memory for all agents to read/write
@dataclass
class Blackboard:
    """Shared memory space for all agents to read from and write to."""
    goal: str
    # Holds subtasks
    plan: list[dict] = field(default_factory=list)
    # Aggregated research notes from the ResearcherAgent
    research_notes: str = ""
    # The working doc
    draft: str = ""
    # Critique from the CriticAgent
    critique: list[str] = field(default_factory=list)
    # Number of revisions made to the draft
    revisions: int = 0
    # Transcript of all agent interactions for debugging/logging
    transcript: list[dict] = field(default_factory=list)

    def log(self, agent: str, content: str):
        """Records an agent's action to the transcript and prints a summary."""
        self.transcript.append({"agent": agent, "content": content, "t": time.time()})
        print(f"\n* [{agent}] *n{content}")


# Function to clean up code blocks around LLM outputs
def _extract_json(text: str) -> str:
    """Cleans up markdown code blocks around JSON outputs from the LLM."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return text


def safe_json_list(text: str) -> list[dict]:
    """Attempts to parse a JSON list; provides a robust fallback if the LLM hallucinates."""
    try:
        data = json.loads(_extract_json(text))
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, TypeError):
        # Fallback, manually parse each line if JSON decoding fails
        return [
            {"step": i + 1, "owner": "researcher", "task": line.strip("-* ")}
            for i, line in enumerate(text.splitlines()) if line.strip()
        ]


def safe_json_obj(text: str, default: dict) -> dict:
    """Attempts to parse a JSON object, returning a default dictionary on failure."""
    try:
        return json.loads(_extract_json(text))
    except (json.JSONDecodeError, TypeError):
        return default


# Agents

class Agent:
    """Base agent class handling basic initialisation and execution."""
    name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, backend: LLMBackend):
        """ Initialises the agent with a shared LLM backend."""

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
        """Generates the plan and updates the blackboard."""
        raw = self.act(f"Goal: {bb.goal}", bb, max_tokens=400)
        bb.plan = safe_json_list(raw)
        return bb.plan


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
        # Only considers the last 5 critique points to avoid overwhelming the model
        critique_txt = "\n".join(f"- {c}" for c in bb.critique[-5:]) or "(none)"
        prompt = (
            f"Goal: {bb.goal}\n\nCurrent draft:\n{bb.draft}\n\n"
            f"Critique to address:\n{critique_txt}\n\nProduce a revised draft."
        )
        bb.draft = self.act(prompt, bb, max_tokens=1800)
        bb.revisions += 1
        return bb.draft


class CriticAgent(Agent):
    """Responsible for reviewing the draft and gatekeeping the revision loop."""
    name = "Critic"
    system_prompt = (
        "You are a rigorous but fair critic. Evaluate the draft against the "
        "goal and research notes. Respond ONLY with valid JSON: "
        '{"score": 1-10, "verdict": "accept"|"revise", "issues": ["...", "..."]}. '
        "A score of 8+ with no major issues means accept."
        "Do not output any conversational text, greetings, markdown fences."
        "Output raw JSON only."
    )

    def review(self, bb: Blackboard) -> dict:
        """Scores the draft and outputs structured revision issues."""
        prompt = f"Goal: {bb.goal}\n\nResearch notes:\n{bb.research_notes}\n\nDraft:\n{bb.draft}"
        raw = self.act(prompt, bb, max_tokens=300)
        verdict = safe_json_obj(
            raw,
            default={"score": 5, "verdict": "revise", "issues": ["Could not parse critique JSON"]},
        )
        # Stores the issues on the blackboard for the writer to address in the next revision
        bb.critique.extend(verdict.get("issues", []))
        return verdict


# Orchestrator
def run_task(goal: str, backend: LLMBackend, max_revisions: int = 4) -> Blackboard:
    """
    Joins the multi-agent workflow together.
    Runs the pipeline: Plan, research, draft, (back and forth between critique and revise), finish
    """
    bb = Blackboard(goal=goal)

    # Initialise all agents with the shared backend
    planner = PlannerAgent(backend)
    researcher = ResearcherAgent(backend)
    writer = WriterAgent(backend)
    critic = CriticAgent(backend)

    # Step 1: Planner breaks the goal into subtasks
    plan = planner.plan(bb)

    # Step 2: Researcher executes each subtask and gathers notes
    for step in plan:
        if str(step.get("owner", "")).lower() == "researcher":
            researcher.research(step.get("task", ""), bb)

    # Step 3: Writer produces the initial draft based on the research notes
    writer.draft(bb)

    # Step 4: Loop of critic reviewing the draft and providing feedback, then writer revising as needed
    for i in range(max_revisions):
        verdict = critic.review(bb)
        print(f"\n[orchestrator] round {i + 1} critic verdict: {verdict}")

        # Break the loop early if the draft is accepted with a high enough score
        if verdict.get("verdict") == "accept" and verdict.get("score", 0) >= 8:
            break

        # Otherwise, run another revision cycle
        writer.revise(bb)

    return bb

# CLI execution

def main():
    """Parses command-line arguments and starts the orchestration layer."""
    ap = argparse.ArgumentParser(description="Local multi-agent task planner.")
    ap.add_argument("goal", type=str, help="The task/goal for the agents to accomplish.")
    ap.add_argument("--model-path", type=str, default=None,
                     help="Path to a local .gguf model file, used with llama-cpp-python.")
    ap.add_argument("--hf-fallback", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                     help="Hugging Face model id used if llama-cpp-python/model-path isn't available.")
    ap.add_argument("--max-revisions", type=int, default=4)
    args = ap.parse_args()

    # Determines which backend engine to use based on provided arguments and availability
    backend = build_backend(args.model_path, args.hf_fallback)

    # Runs the core multi-agent task planning loop
    bb = run_task(args.goal, backend, max_revisions=args.max_revisions)

    # Prints the final draft and execution statistics
    print("\n\nFinal Draft\n")
    print(bb.draft)
    print(f"\n[stats] revisions: {bb.revisions}, transcript entries: {len(bb.transcript)}")


if __name__ == "__main__":
    main()
