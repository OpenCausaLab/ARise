decompose_prompt = """Your task is to decompose the original question into one smaller sub-question based on the Intermediate answer and Observation. The decomposed process is encouraged to be done from multiple perspectives.
Output a thought to reason the original question, and output one sub-question that you think is appropriate to solve next. DO NOT REPEAT the question and DO NOT try to answer the question.
The output format is limited to:
Thought: ...
Sub-question: ...
Here, the "..." indicates omitted output information that you need to fill in.

{state}

Observation: {knowledge}

Output:
"""


answer_prompt = """Your task is to answer the following question using provided supporting facts.
The output answer should be a complete declarative sentence, rather than directly outputting phrases or words. Do not use pronouns in the sentence.
Specially, if no provided supporting facts, just output "No directly relevant facts found." and nothing else.

Question:
{question}

Supporting facts:
{external_knowledge}

Output:
"""


final_answer_prompt = """Your task is to answer the original question based on the intermediate answers.
Output the final answer directly and nothing else.

Here are some examples:

Example 1:
Original question: Who lived longer, Muhammad Ali or Alan Turing?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Example 2:
Original question: When was the founder of craigslist born?
Intermediate answer: Craigslist was founded by Craig Newmark.
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

{state}So the final answer is: """


zero_shot_proposal_prompt = """
Your task is to answer the following question using provided supporting facts.
Output the final answer directly and nothing else.

Question:
{question}

Supporting facts:
{external_knowledge}

Output:
"""


tot_value_prompt = """Given a question, your task is to evaluate the quality of the current reasoning step.
Directly output JUST A NUMBER between 0 and 10 to represent the quality score. Do not output anything else.

Original question:
{question}

Current reasoning step:
{state}

Output:
"""


tot_thought_prompt = """Given a question and the existing steps, your task is to continue completing the remaining reasoning steps based on the provided supporting facts.
Each complete reasoning process consists of THREE steps, and the output format is limited to:
First: ...
Second: ...
Third: ...

If there are no existing steps, you need to start reasoning from the beginning, and output:
First: ...
Second: ...
Third: ...
If the first step is given, you need to output:
Second: ...
Third: ...

If both the first and second steps are given, you need to output:
Third: ...

Question:
{question}

Existing steps:
{state}

Supporting facts:
{external_knowledge}

Output:
"""


tot_answer_prompt = """Given a question, the existing steps and supporting facts, your task is to output the final answer directly and nothing else.

Question:
{question}

Existing steps:
{state}

Supporting facts:
{external_knowledge}

Output:
"""


star_value_prompt = """Given a question, your task is to determine the consistency score of its decomposition sub-questions and corresponding intermediate answers with the original question. Directly output JUST A NUMBER between 0 and 10 to represent the consistency score. Do not output anything else.

Question:
{question}

Sub-questions:
{sub}

Intermediate answers:
{ans}

Output:
"""


complete_query_from_ans = """Given intermediate answer containing the facts about the original question, which is unknown, your task is to infer what the orginal question might have been.
Output the most likely original question directly and nothing else.

Here are some examples:

Example 1:
Intermediate answer:
Muhammad Ali was 74 years old when he died.
Alan Turing was 41 years old when he died.
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?

Example 2:
Intermediate answer:
Craigslist was founded by Craig Newmark.
Craig Newmark was born on December 6, 1952.
The original question might be:
When was the founder of craigslist born?

Intermediate answer:
{answer}
The original question might be:
"""


complete_query_from_subquery = """Given sub-question derived from the original question, which is unknown, your task is to infer what the original question might have been.
Output the most likely original question directly and nothing else.

Here are some examples:

Example 1:
Sub-question:
How old was Muhammad Ali when he died?
How old was Alan Turing when he died?
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?

Example 2:
Sub-question:
Who was the mother of George Washington?
The original question might be:
Who was the maternal grandfather of George Washington?

Example 3:
Sub-question:
Who is the director of Jaws?
Where is Steven Spielberg from?
Who is the director of Casino Royale?
Where is Martin Campbell from?
The original question might be:
Are both the directors of Jaws and Casino Royale from the same country?

Sub-question:
{query}
The original question might be:
"""
