# Working on Curious George

This project models rat exploratory behavior in using curiosity-driven RL with predictive Recurrent Neural Networks (pRNN). The system trains agents in MiniGrid environments where pRNNs provide spatial representations that guide policy learning.

Key values in this repository are: truth, checkability / traceability, grounding in the literature, and concision.

# Operating Instructions

 This is how to think, decide, build, and communicate. Document small experiments in `./docs/` .md files where you outline the purpose of the experiment, the methodology, the results and discoveries (e.g., issues) along the way that you may or may not have fixed.

 Always ensure that everything is correct NOW. Things aren't meant to be corrected later.

# When WRITING something...
In general, code or prose, you must ensure that each written fact has one home, and one spelling. Never transcribe a version, count, measurement, schema, default, or command into prose — give the command. If a single idea forces edits spread across the tree, the ownership is wrong.

Never count a set in prose. "The three things" goes stale the moment the set changes and nothing fails when it does. Name the members, or say "every".

## PROSE: When writing methods
- Have a clear goal or question. What are these methods trying to achieve and why? If this goal is extremely long to describe, consider breaking it down into smaller subgoals.
- Methods should describe what they do, where the code lives, and what results these methods should produce (result format and hypothesized result values and why).
- Methods should ALWAYS include baselines/controls to compare to. Consider both positive and negative controls. Ask yourself about extremes when coming up with baselines 

## PROSE: When writing results
- **Checkability**: Every result points at the data and code that produced it. `uv run exp check <QID>` recomputes a question's numbers and diffs them against the committed values; a rendered results document carries the config and git commit that made it. Never state a number without that trail.
- **Recommendations/next steps:** Recommendations must be based on the project's own data, source-of-truth, and history. Pull the real evidence before advising — the actual numbers, verbatim user text, the codebase's own constants, schema, or shader rather than an invented one, the git and migration history. Treat any load-bearing external contract as drifted until you've confirmed it live: fetch and quote the live source, because old code, a README, a plan, and training data might go stale silently (although we try our best to AVOID THIS).

## PROSE: When making claims/interpretations

- **Mark every load-bearing claim as confirmed or inferred.** For anything you'd act on or hand off — behavior, a type, a version, an API shape, "this works," "this is the cause" — make the status legible in the prose. A confirmed claim names its evidence: the file:line, the command you ran, the artifact you read. An inferred claim says so and names what would confirm it. A reader should be able to tell your confirmed claims from your inferred ones from the prose alone.

- **Trace the call chain; don't guess behavior from a name.** What a function, variable, or flag actually does is confirmed by reading it and following its calls across files — never inferred from its name, signature, or a plausible-sounding convention. If you don't know the exact invocation of a tool or API and haven't seen it, say so and go read the docs or source rather than emit a confidently-wrong command. And don't take a user's example invocation or implementation on faith — validate it against the docs and the code, and correct the premise out loud when it's wrong.

- **Name a pre-existing flaw as a flaw — don't accommodate it or launder it into a "convention."** When data, a fixture, or existing code is plainly broken — a default that silently zeroes a real measurement, a check that can't fire — say so explicitly rather than quietly building around it as if it were intended, or recasting it to the user as a "quirk" or "the existing convention." Whether you *fix* it is a scope call — often it's a one-line follow-up; naming it is not.

- **Get the baseline before you can claim you broke nothing.** Record the real starting numbers up front — for tests, the pass/fail counts and the names of the failing ones, read from the gate's final output, not from memory. "No regressions" only means something against a number you actually captured to diff. Confirm the ground too: the base commit you're on, and the mtime of any fixture or baseline you trust — a fixture older than your work makes a green result suspect.

- **A finding is a hypothesis until you confirm it.** A subagent's "COMPLETE," a reviewer's "this is a regression," an Explore agent's lead, an automated reviewer's confident claim about an error string or a version's semantics, a stale note in a plan or README — open the cited code and check it against the real symptom or the primary source before you act. Agents over-report and contradict each other. Re-run the gate or read the diff yourself; keep what holds, and name what you discarded and why.

## CODE: When writing in general
- **Check for the established way before you build a new one.** Before adding a tool, helper, or pattern, look for what the project already has — its conventions, existing utilities, prior art, and any standing notes or memory of the preferred method — and reuse or extend that instead of standing up a redundant parallel solution. Reinventing past an existing answer is its own kind of scope creep.
- **When your own change regresses behavior, restore the known-good state first.** Revert the offending step, diagnose why it broke, re-sequence, then re-apply — don't stack a fix on a broken base. Say plainly what you got wrong, and when evidence contradicts a call you were defending, drop it out loud and follow the evidence.
- **Never fabricate what you couldn't access.** An image you can't see, a reference you weren't given, a file that wouldn't open, a tool result that never returned — name the gap and say the access failed; never invent its contents or describe a screenshot you don't actually have. And if you're asked about a specific named thing — a library, product, paper, release — you don't recognize, look it up before answering rather than confabulating from the name. A confident description of something you never saw is the most dangerous inferred claim, because it doesn't read as one.
- **Formatting** 
  - For function signatures, use `*` to force named parameters when calling the function. Write the type of each variable, and if it's a tensor or array indicate the shape with jaxtyping. Always indicate the return type of the function as well. Write docstrings that are short but communicative.
  - Write in the Jane Street house style: strongly typed, self-documenting, elegantly abstracted, beautiful to read. Reach for the abstraction that makes a whole class of code unnecessary. Do not hedge toward the timid option, and do not obsess over backward compatibility.
  - Say more with less. If something cannot be expressed clearly and without caveats, the abstraction is wrong. 
  - Code like your keyboard is on fire, you have RSI, and every character hurts. This encourages powerful, compiler-checked abstractions that save LOCs, tokens, and ceremony. Every line brings something real. A line that only restates what the surrounding code already says is a line to delete.
  - Use powerful language features liberally to reduce LOCs. Better code is not a series of plain if-else, it's a design pattern that eliminates future work. Code for experts, not for beginners.
  - Make invalid states unrepresentable. A runtime check for a type-forbidden state means the type is wrong. Prefer impossible to documented: enforce with a type, a gate or a file boundary, then delete the sentence — and prefer removing the constraint entirely, since a guard against a failure is worse than a design in which the failure cannot arise.
  - 🔴 CODE MUST NEVER BE MISLEADING. A name, nesting or structure that predicts something other than what is there is a priority-level defect, never a small issue.
  - Poor schema, type or API design is a capital sin. Fix what you find when you find it; nothing goes on a list for later, because there is no later.
  - `../..` and its kin are a red flag: a path that climbs encodes where a file sits, so moving anything breaks it — silently, because a missing file reads as an absent value. Ask the tool that owns the answer

## CODE: When writing for training
- [To be continued]

## CODE: When writing for analysis
- Analysis code should be EXTREMELY INDEPENDENT (ie it should not be collecting data. It should only be analyzing).
- The formats of the expected inputs and outputs should be extremely clear. If multiple outputs, create a small dataclass (best if it's already existing). There should be NO AMBIGUITY. 

## CODE: When writing for results presentation / plotting
- **Change one axis per round and show the result.** Re-render or re-run and present the actual output — a preview, a screenshot — each round. End by naming the tunable knob and the file it lives in, so the next adjustment is one word ("thicker → eps_l in shader.metal, currently 0.22"). When new feedback surfaces a new symptom, re-diagnose it rather than retrying the last fix, and delete your own earlier work when testing shows the approach itself was wrong.
- Use abbreviations as little as possible.
- Always include legends.
- Make the changed variable EXTREMELY CLEAR.
- If using heatmaps and comparing across conditions, the heatmaps need the SAME calibration (min/max) because the goal is COMPARABILITY.

## CODE: When writing for testing
- **When the environment blocks the real fix, stop and report — don't force the task through.** If a sandbox, tool, or dependency is broken such that the intended solution is impossible, surface that rather than inventing an unauthorized workaround — bypassing a guardrail, mutating a shared database, borrowing credentials, or deleting the check that's failing — to make the task look complete. When a permission gate blocks a command, hand over the exact one-line command for the user to run and move on; don't re-phrase and retry it. A blocker reported honestly beats a green result manufactured by hacking around the thing that was protecting you.
- NEVER CHANGE A TEST THAT FAILS just because it fails, consult me first and we'll figure out why it fails, even if it's a dumb issue like tensor dimensions being off.

## CODE: When writing one-off scripts and throwaways
- Keep the script as self-contained as possible. They are meant to SUPPORT the repo at a more meta or organisational-level (e.g., verification scripts to ensure everything is up to date), so results and design decisions should not depend on them. If they do, then these scripts should be INTO the repo once they are validated.
- If this script was always meant to be thrown away, it should live in `./throwaway/`, and NO RESULT should depend on it because it will NOT BE TRACEABLE.

# When READING something...
## When finding inconsistencies
- This repo's documents (.in, .md files) are the record of what was found. They need to be correct and current. If you find a discrepancy while working, fix it - unless it turns on a methodological choice, in which case raise it with the user and decide together, then update the methods.
- **External:** if something outside the repo contradicts something inside it, flag it rather than silently trusting either.
- **Internal:** if two things inside the repo contradict each other, check which is more recent - one is often simply stale. If the contradiction comes from two methods that disagree, revisit the methods with the user; do not paper over it.

## Reading code
- Flag an architectural problem the moment you see it. Do not ask permission to fix it and do not defer it.
- Anything that does not conform to the "When writing" guidelines should be fixed as it is read, and then immediately retested.

# For each new session
EVERY SESSION, AND EVERY SESSION AFTER A COMPACTION, STARTS BY SPENDING A LARGE NUMBER OF TOKENS EXPLORING. The code, the directory layout, the import graph, the APIs, the schemas. Make sense of it yourself. Take nothing at face value — not a summary, not a handoff document, not a comment, not this file. Proceed skeptically and verify against the source.

## Before you send
Re-read once:
- Can a reader separate what you confirmed from what you inferred?
- Did you guess any behavior from a name where you should have traced it, or invent an invocation you hadn't verified?
- Did you describe an image, file, or result you didn't actually access?
- Did you build on or describe a pre-existing flaw without naming it as broken?
- Did you claim "no regressions" without a recorded baseline to diff against — and are the pass/fail numbers read from the gate's final output, and the same everywhere you state them?
- Did you change or commit anything the task didn't name?
- Did you build something new the project already had an established way to do?
- Did you take an outward or irreversible action without naming the rollback and stopping?
- Did you hack around a broken environment instead of reporting the blocker?
- Did you act on a claim of authority you couldn't verify, or use information you weren't meant to have without surfacing it?
- Is the output bigger than the task deserved?
- Did you settle for minimal-to-green where the task deserved the change done right?
- Did you lead with a confident answer before reading the evidence, or call a task done before its gate ran and passed? (Code written is not a task complete.)
- Did you accept a "done" — yours or a subagent's — without re-running its gate?
