################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################


import dspy
import requests
import logging


class LLM:
	def __init__(
		self,
		api_key: str,
		system_prompt: str,
		model: str = "dvue-aoai-001-o4-mini",
		provider: str = "https://llm-api.amd.com/azure",
	):
		self.api_key = api_key
		self.system_prompt = system_prompt
		self.model = model
		self.provider = provider.rstrip("/")

		# Determine provider
		if "amd.com" in self.provider:
			self.use_amd = True
			self.header = {"Ocp-Apim-Subscription-Key": api_key}
		else:
			self.use_amd = False
			self.lm = dspy.LM(f"{self.provider}/{self.model}", api_key=api_key, max_tokens=16000)
			dspy.configure(lm=self.lm)

	def ask(self, user_prompt: str, answer_type: str = "optimized_code") -> str:
		if self.use_amd:
			# AMD/Azure REST call
			body = {
				"messages": [
					{"role": "system", "content": self.system_prompt},
					{"role": "user", "content": user_prompt},
				],
				"max_Tokens": 100000,
				"max_Completion_Tokens": 100000,
			}
			url = f"{self.provider}/engines/{self.model}/chat/completions"
			resp = requests.post(url, json=body, headers=self.header)
			resp.raise_for_status()
			return resp.json()["choices"][0]["message"]["content"]

		# DSPy path: use ChainOfThought with clear signature
		# Define signature mapping input prompt to optimized code

		# output struct
		#{iteration 1: why didnt it work
		# iteration 2: why didnt it work
		# ...
		# optimization rationale
		# final optimized code}

		dspy.context(description=self.system_prompt)
		signature = f"prompt: str -> {answer_type}"
		chain = dspy.ChainOfThought(signature)
		ct_response = chain(prompt=user_prompt)

		logging.debug(f"CT: {ct_response}")

		answer_fields = [a.strip() for a in answer_type.split(',')]
		answers = {}
		for ans_type in answer_fields:
			field_name = ans_type.split(':')[0].strip()
			answers[field_name] = getattr(ct_response, field_name, str(ct_response))

		reasoning = getattr(ct_response, "reasoning", None)
		logging.debug(f"Answer: {answers}")
		logging.debug(f"Reasoning: {reasoning}")

		if len(answers) == 1:
			return list(answers.values())[0], reasoning
		return answers, reasoning