import json
import re


class AssemblyAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert deterministic planner. "
            "Return only the final action sequence requested by the user."
        )

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        """
        Qwen3-8B plantea el plan. El codigo solo controla el prompt y normaliza
        la salida al formato canonico requerido por el evaluador.
        """
        response = llm_engine_func(
            prompt=self._build_prompt(scenario_context),
            system=self.system_prompt,
            temperature=0.0,
            do_sample=False,  # Asegurate de esto
            enable_thinking=False,
        )

        return self._parse_actions(response)

    def _build_prompt(self, scenario_context: str) -> str:
        domain = "blocks" if "set of blocks" in scenario_context else "objects"
        initial_state, goal = self._extract_target_problem(scenario_context)
        symbolic_initial = self._symbolic_facts(domain, initial_state)
        symbolic_goal = self._symbolic_facts(domain, goal)

        if domain == "blocks":
            domain_rules = """
Domain blocks. Facts: handempty, clear(X), ontable(X), on(X,Y), holding(X).
Ops:
engage_payload X: pre handempty ontable(X) clear(X); add holding(X); del handempty ontable(X) clear(X)
release_payload X: pre holding(X); add handempty ontable(X) clear(X); del holding(X)
mount_node X Y: pre holding(X) clear(Y); add handempty on(X,Y) clear(X); del holding(X) clear(Y)
unmount_node X Y: pre handempty on(X,Y) clear(X); add holding(X) clear(Y); del handempty on(X,Y) clear(X)
Rules: make on(X,Y) with mount_node X Y; before that hold X and clear Y. Simulate facts after each action.
""".strip()
            examples = """
Examples:
I: handempty, ontable(red), ontable(blue), clear(red), clear(blue)
G: on(red,blue)
A: ["(engage_payload red)", "(mount_node red blue)"]
I: handempty, on(red,blue), ontable(blue), clear(red), ontable(orange), clear(orange)
G: on(blue,red)
A: ["(unmount_node red blue)", "(release_payload red)", "(engage_payload blue)", "(mount_node blue red)"]
""".strip()
        else:
            domain_rules = """
Domain objects. Facts: harmony, planet(X), province(X), pain(X), craves(X,Y).
Ops:
attack X: pre province(X) planet(X) harmony; add pain(X); del province(X) planet(X) harmony
succumb X: pre pain(X); add province(X) planet(X) harmony; del pain(X)
feast X Y: pre craves(X,Y) province(X) harmony; add pain(X) province(Y); del craves(X,Y) province(X) harmony
overcome X Y: pre province(Y) pain(X); add harmony province(X) craves(X,Y); del province(Y) pain(X)
Rules: make craves(X,Y) with overcome X Y, never feast. Before overcome X Y need pain(X), province(Y). Make pain(X) by attack X or feast X Z when craves(X,Z) exists. Feast consumes craves; simulate facts after each action.
""".strip()
            examples = """
Examples:
I: harmony, planet(a), planet(b), planet(c), province(a), province(b), province(c)
G: craves(a,c), craves(b,a)
A: ["(attack a)", "(overcome a c)", "(attack b)", "(overcome b a)"]
I: craves(a,b), craves(b,c), harmony, planet(c), province(a)
G: craves(b,a)
A: ["(feast a b)", "(succumb a)", "(feast b c)", "(overcome b a)"]
I: craves(a,c), harmony, planet(b), planet(c), province(a), province(b)
G: craves(c,a)
A: ["(feast a c)", "(succumb a)", "(attack c)", "(overcome c a)"]
""".strip()

        return f"""
{domain_rules}

{examples}

Solve only this current problem. Think internally; output only JSON.
Do not copy an example. Simulate facts; every precondition must hold; all goals true at end. Prefer shortest valid plan.
I:
{symbolic_initial}

G:
{symbolic_goal}

[FINAL]
["(action arg)", "(action arg arg)"]

No explanations. No words object/block inside actions.
""".strip()

    def _symbolic_facts(self, domain: str, text: str) -> str:
        facts = []
        for item in [x.strip() for x in re.split(r", | and ", text) if x.strip()]:
            if domain == "objects":
                if item == "harmony":
                    facts.append("harmony")
                elif match := re.fullmatch(r"(planet|province) object ([a-e])", item):
                    facts.append(f"{match.group(1)}({match.group(2)})")
                elif match := re.fullmatch(r"object ([a-e]) craves object ([a-e])", item):
                    facts.append(f"craves({match.group(1)},{match.group(2)})")
            else:
                if item == "the hand is empty":
                    facts.append("handempty")
                elif match := re.fullmatch(r"the (\w+) block is unobstructed", item):
                    facts.append(f"clear({match.group(1)})")
                elif match := re.fullmatch(r"the (\w+) block is on the table", item):
                    facts.append(f"ontable({match.group(1)})")
                elif match := re.fullmatch(r"the (\w+) block is on top of the (\w+) block", item):
                    facts.append(f"on({match.group(1)},{match.group(2)})")
        return ", ".join(facts)

    def _extract_target_problem(self, scenario_context: str) -> tuple[str, str]:
        parts = scenario_context.split("[STATEMENT]")
        if len(parts) < 3:
            raise ValueError("No se encontro el segundo problema.")

        statement = parts[2].split("My plan is as follows:")[0].strip()
        match = re.search(
            r"As initial conditions I have that, (.*?)\.\nMy goal is to have that (.*?)\.",
            statement,
            re.S,
        )
        if not match:
            raise ValueError("No se pudo extraer estado inicial y meta.")
        return match.group(1), match.group(2)

    def _parse_actions(self, text: str) -> list:
        final_text = self._final_text(text)
        return (
            self._parse_json_actions(final_text)
            or self._parse_parenthesized_actions(final_text)
            or self._parse_english_actions(final_text)
        )

    def _final_text(self, text: str) -> str:
        match = re.search(r"\[FINAL\]\s*([\s\S]*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else text

    def _parse_json_actions(self, text: str) -> list:
        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char != "[":
                continue
            try:
                values, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if not isinstance(values, list):
                continue

            actions = [self._normalize_action(value) for value in values if isinstance(value, str)]
            if actions and all(actions):
                return actions
        return []

    def _parse_parenthesized_actions(self, text: str) -> list:
        actions = []
        for match in re.finditer(r"\([^()]+\)", text.lower()):
            action = self._normalize_action(match.group(0))
            if action:
                actions.append(action)
        return actions

    def _parse_english_actions(self, text: str) -> list:
        patterns = [
            (r"pick up the (\w+) block", "(engage_payload {0})"),
            (r"put down the (\w+) block", "(release_payload {0})"),
            (r"(?:stack|mount_node) the (\w+) block on top of the (\w+) block", "(mount_node {0} {1})"),
            (r"unmount_node the (\w+) block from on top of the (\w+) block", "(unmount_node {0} {1})"),
            (r"attack object ([a-e])", "(attack {0})"),
            (r"feast object ([a-e]) from object ([a-e])", "(feast {0} {1})"),
            (r"succumb object ([a-e])", "(succumb {0})"),
            (r"overcome object ([a-e]) from object ([a-e])", "(overcome {0} {1})"),
        ]

        actions = []
        for raw_line in text.lower().splitlines():
            line = re.sub(r"^[\-\*\d\.\)\s]+", "", raw_line.strip()).strip("` ,;")
            for pattern, template in patterns:
                match = re.search(pattern, line)
                if match:
                    actions.append(template.format(*match.groups()))
                    break
        return actions

    def _normalize_action(self, action: str) -> str | None:
        action = action.strip().lower().strip("()` ,;")
        tokens = re.sub(r"[^a-z_ ]+", " ", action).split()
        if not tokens:
            return None

        verb = tokens[0]
        args = [
            token
            for token in tokens[1:]
            if token not in {"object", "block", "the", "from", "on", "top", "of"}
        ]

        arity = {
            "engage_payload": 1,
            "release_payload": 1,
            "mount_node": 2,
            "unmount_node": 2,
            "attack": 1,
            "feast": 2,
            "succumb": 1,
            "overcome": 2,
        }
        if verb not in arity or len(args) != arity[verb]:
            return None

        return f"({verb} {' '.join(args)})"
