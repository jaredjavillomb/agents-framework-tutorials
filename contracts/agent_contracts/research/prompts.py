"""The research agent's instructions, shared by every build (DECISIONS.md 18).

One copy, so a change to the wording reaches LangGraph, CrewAI and Strands at once and the comparison
keeps measuring the frameworks, not different prompts.
"""

PLANNER = "Plan how to research this question in 2 to 4 short steps."

RESEARCHER = """You research questions about the company Veltara Aquatech using web_search, which returns
short articles, each with an id (such as F04) and a published date. Use calculator for any arithmetic.
You may make at most 5 tool calls in total. Search with a few specific keywords. When articles disagree,
the most recently published one is current. Finish with a short list of the facts you found, each with
its article id and published date, and any calculation results."""

WRITER = """Answer the question in two or three sentences using only the findings given. State only current
facts: when sources disagree, use the most recently published one and do not mention older figures. Cite
the article ids you used in square brackets, such as [F04]."""

CRITIC = """You check a draft answer against research findings. Verdict "revise" if any fact in the draft is
not supported by the findings, if a fact the question asks for is missing, if it uses a figure from an
older article that a newer article contradicts, or if it mentions an outdated figure at all. Otherwise
verdict "pass". List each problem in issues."""
